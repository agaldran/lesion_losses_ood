import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from . import paired_transforms_tv as p_tr
from . import triplet_transforms_tv as triplet_tr
from skimage import io, img_as_ubyte
from skimage.metrics import structural_similarity as ssim, variation_of_information, mean_squared_error
import os, os.path as osp

from skimage.segmentation import morphological_chan_vese
from skimage.color import rgb2hsv

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def mutual_information(im1, im2):
    # assumes images contain integer values in [0,255]
    X = np.array(im1).astype(float)
    Y = np.array(im2).astype(float)
    hist_2d, _, _ = np.histogram2d(X.ravel(), Y.ravel(), bins=255)
    pxy = hist_2d / float(np.sum(hist_2d))  # joint probability distribution

    px = np.sum(pxy, axis=1)  # marginal distribution for x over y
    py = np.sum(pxy, axis=0)  # marginal distribution for y over x

    Hx = - sum(px * np.log(px + (px == 0)))  # Entropy of X
    Hy = - sum(py * np.log(py + (py == 0)))  # Entropy of Y
    Hxy = np.sum(-(pxy * np.log(pxy + (pxy == 0))).ravel())  # Joint Entropy

    M = Hx + Hy - Hxy  # mutual information
    nmi = 2 * (M / (Hx + Hy))  # normalized mutual information
    return nmi

class ActiveLabels(Dataset):
    def __init__(self, csv_path, transforms=None, mean=None, std=None, test=False, active_labels=True):
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image_path
        self.test = test
        if not self.test:
            self.target_list = df.mask_path
        else:
            self.target_list = None
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)
        self.active_labels = active_labels

    def get_random_params(self):
        # num_iters = np.random.randint(1, 5)
        num_iters = np.random.randint(1, 10)
        smoothing = np.random.randint(1, 4)
        lambda1 = np.random.normal(loc=1.0)
        return num_iters, smoothing, lambda1

    def __getitem__(self, index):
        # load image and labels
        im_name = self.im_list[index]
        img = Image.open(self.im_list[index])
        orig_size = img.size
        if not self.test:
            target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
            if self.active_labels:
                # apply active labels now
                sat = rgb2hsv(np.array(img))[:, :, 1]
                n, s, l1 = self.get_random_params()
                target = morphological_chan_vese(sat, init_level_set=target, num_iter=n, smoothing=s, lambda1=l1)
            # continue
            target = Image.fromarray(target)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            img = self.normalize(img)
            return img, target / 255
        else:
            if self.transforms is not None:
                img = self.transforms(img)
            img = self.normalize(img)
            return img, im_name, orig_size

    def __len__(self):
        return len(self.im_list)

class PolypDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mean=None, std=None, test=False):
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image_path
        self.test=test
        if not self.test:
            self.target_list = df.mask_path
        else:
            self.target_list = None
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        im_name = self.im_list[index]
        img = Image.open(self.im_list[index])
        orig_size = img.size
        if not self.test:
            target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
            target = Image.fromarray(target)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            img = self.normalize(img)
            return img, target/255
        else:
            if self.transforms is not None:
                img = self.transforms(img)
            img = self.normalize(img)
            return img, im_name, orig_size

    def __len__(self):
        return len(self.im_list)

class InferenceDataset(Dataset):
    def __init__(self, data_source, transforms=None, mean=None, std=None):
        self.data_source = data_source
        self.target_list = None

        if self.data_source.endswith('.csv'):
            df = pd.read_csv(self.data_source)
            self.im_list = df.image_path
            if len(df.columns) == 2:
                self.target_list = df.mask_path
        else:
            self.im_list = os.listdir(self.data_source)

        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)
        self.denormalize = tr.Normalize((-np.array(mean)/np.array(std)).tolist(), (1.0 / np.array(std)).tolist())

    def __getitem__(self, index):
        # load image and labels
        if self.data_source.endswith('.csv'):
            im_name = self.im_list[index]
            img = Image.open(im_name)
        else:
            im_name = osp.join(self.data_source, self.im_list[index])
            img = Image.open(im_name)
        orig_size = img.size
        if self.transforms is not None:
            if self.target_list is not None:
                target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
                target = Image.fromarray(target)
                img, target = self.transforms(img, target)
                img = self.normalize(img)
                return img, target, im_name, orig_size
            else:
                img = self.transforms(img)
                img = self.normalize(img)
                return img, im_name, orig_size



    def __len__(self):
        return len(self.im_list)


class RegDataset(Dataset):
    def __init__(self, path_soft_seg, sim_method='mutual_info', transforms=None, mean=None, std=None):
        path_im = 'data/images/'
        path_seg = 'data/masks_binary/'
        path_soft_seg = osp.join('data/results_val_soft', path_soft_seg)

        print('Using soft segs from {}'.format(path_soft_seg))
        self.im_list = os.listdir(path_im)
        self.im_list = sorted([osp.join(path_im, n) for n in self.im_list])

        self.seg_list = os.listdir(path_seg)
        self.seg_list = sorted([osp.join(path_seg, n) for n in self.seg_list])

        self.soft_seg_list = sorted(os.listdir(path_soft_seg))
        self.soft_seg_list = sorted([osp.join(path_soft_seg, n) for n in self.soft_seg_list])

        self.sim_method = sim_method
        self.transforms = transforms
        self.mean, self.std = mean, std
        self.normalize = tr.Normalize(self.mean, self.std)
        self.tensorize = tr.ToTensor()

    def threshold_soft_seg(self, s, t):
        s = np.array(s) / 255.0
        s = s > t
        return Image.fromarray(img_as_ubyte(s))

    def compute_similarity(self, im, im_deg, sim_method='mutual_info'):
        # sim_method = 'mutual_info', 'dice', 'ssim', 'var_info'
        im = np.array(im)
        im_deg = np.array(im_deg)
        if sim_method == 'mutual_info':
            return mutual_information(im, im_deg)
        elif sim_method == 'dice':
            return dice_score(im, im_deg)
        elif sim_method == 'ssim':
            return ssim(im.astype(bool), im_deg.astype(bool))
        elif sim_method == 'var_info':
            under_seg, over_seg = variation_of_information(im.astype(bool), im_deg.astype(bool))
            return 1 - (under_seg + over_seg)
        elif sim_method == 'mse':
            return 1 - mean_squared_error(im.astype(bool).ravel(), im_deg.astype(bool).ravel())

    def __getitem__(self, index):
        img = Image.open(self.im_list[index])

        seg = Image.open(self.seg_list[index]).convert('L')
        soft_seg = Image.open(self.soft_seg_list[index]).convert('L')

        if self.transforms is not None:
            # need to compute similarity on transformed images
            img, seg, soft_seg = self.transforms(img, seg, soft_seg)

        random_tr = np.random.randint(low=10, high=245) / 255.
        bin_soft_seg = self.threshold_soft_seg(soft_seg, random_tr)

        sim = self.compute_similarity(seg, bin_soft_seg, sim_method=self.sim_method)

        # only return img and bin soft seg, with sim to seg
        img = self.tensorize(img)
        bin_soft_seg = self.tensorize(bin_soft_seg)
        if self.mean is not None:
            img = self.normalize(img)
            # bin_soft_seg = tr.Normalize(self.mean[1], self.std[1])(bin_soft_seg)

        return img, bin_soft_seg, sim

    def __len__(self):
        return len(self.im_list)


def get_train_val_reg_datasets(path_soft_seg, mean=None, std=None, tg_size=(480, 640), sim_method='mutual_info'):
    dataset = RegDataset(path_soft_seg=path_soft_seg, mean=mean, std=std, sim_method=sim_method)
    ll = len(dataset)
    ll_t, ll_v = int(0.9 * ll), int(0.1 * ll)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [ll_t, ll_v],
                                                               generator=torch.Generator().manual_seed(42))

    # transforms definition
    # required transforms
    resize = triplet_tr.Resize(tg_size)
    # geometric transforms
    h_flip = triplet_tr.RandomHorizontalFlip()
    v_flip = triplet_tr.RandomVerticalFlip()
    rotate = triplet_tr.RandomRotation(degrees=15, fill=0, fill_tg=0)

    scale = triplet_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = triplet_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = triplet_tr.RandomChoice([scale, transl, rotate])

    # intensity transforms
    brightness, contrast, saturation, hue = 0.10, 0.10, 0.10, 0.01
    jitter = triplet_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = triplet_tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip])

    val_transforms = triplet_tr.Compose([resize])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset


def get_train_val_reg_loaders(path_soft_seg, batch_size=4, tg_size=(480, 640), mean=None, std=None,
                              sim_method='mutual_info', num_workers=0):
    train_dataset, val_dataset = get_train_val_reg_datasets(path_soft_seg, tg_size=tg_size,
                                                            mean=mean, std=std, sim_method=sim_method)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader

def get_train_val_seg_datasets(csv_path_train, csv_path_val, mean=None, std=None, tg_size=(512, 512), active_labels=False):

    train_dataset = ActiveLabels(csv_path=csv_path_train, mean=mean, std=std, active_labels=active_labels)
    val_dataset = ActiveLabels(csv_path=csv_path_val, mean=mean, std=std, active_labels=False)
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=15, fill=0, fill_tg=0)

    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])

    # intensity transforms
    brightness, contrast, saturation, hue = 0.10, 0.10, 0.10, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    # train_transforms = p_tr.Compose([resize, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_seg_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), mean=None, std=None,
                              num_workers=0, active_labels=False):
    train_dataset, val_dataset = get_train_val_seg_datasets(csv_path_train, csv_path_val, tg_size=tg_size,
                                                            mean=mean, std=std, active_labels=active_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader

def build_pseudo_dataset(train_csv_path, path_test_ims, path_test_preds):
    # assumes predictions are in path_to_preds and have the same name as images in path_test_ims
    # image extension does not matter
    train_df = pd.read_csv(train_csv_path)
    test_im_list = sorted(os.listdir(path_test_ims))
    test_im_list = [osp.join(path_test_ims, n) for n in test_im_list]
    test_gt_list = sorted(os.listdir(path_test_preds))
    test_gt_list = [osp.join(path_test_preds, n) for n in test_gt_list]

    assert len(test_im_list) == len(test_gt_list)

    # If there are more pseudo-segmentations than training segmentations
    # we bootstrap training images to get same numbers
    missing = len(test_im_list) - train_df.shape[0]
    if missing > 0:
        extra_segs = train_df.sample(n=missing, replace=True, random_state=42)
        train_df = pd.concat([train_df, extra_segs])

    train_im_list = list(train_df.image_path)
    train_gt_list = list(train_df.mask_path)

    train_im_list.extend(test_im_list)
    train_gt_list.extend(test_gt_list)

    return train_im_list, train_gt_list

def get_test_seg_dataset(csv_path_test, mean=None, std=None, tg_size=(512, 512)):
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    test_transforms = tr.Compose([resize, tensorizer])

    test_dataset = PolypDataset(csv_path=csv_path_test, mean=mean, std=std, test=True)
    test_dataset.transforms = test_transforms

    return test_dataset

def get_test_seg_loader(csv_path_test, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    test_dataset = get_test_seg_dataset(csv_path_test, tg_size=tg_size, mean=mean, std=std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader

def get_inference_seg_dataset(data_source, mean=None, std=None, tg_size=(512, 512)):
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    test_transforms = p_tr.Compose([resize, tensorizer])

    test_dataset = InferenceDataset(data_source=data_source, mean=mean, std=std)
    test_dataset.transforms = test_transforms

    return test_dataset

def get_inference_seg_loader(data_source, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    test_dataset = get_inference_seg_dataset(data_source, tg_size=tg_size, mean=mean, std=std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader

