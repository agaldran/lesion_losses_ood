import sys, json, os, time, argparse
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch

from utils.get_loaders import get_train_val_seg_loaders
from utils.model_saving_loading import save_model, str2bool
from utils.reproducibility import set_seeds
from utils.evaluation import dice_score
from torch.optim.lr_scheduler import CosineAnnealingLR


# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='data/train_f1.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='fpnet_resnext50_W_imagenet', help='architecture')
parser.add_argument('--loss1', type=str, default='ce', help='first loss: ce, dice, topk, focal loss')
parser.add_argument('--loss2', type=str, default='dice', help='second ')
parser.add_argument('--mixture', type=str, default='linear', help='mechanism to combine loss1 and loss2: linear, fine_tune_loss2, only_loss1')
parser.add_argument('--batch_dice', type=str2bool, nargs='?', const=True, default=False, help='per-sample & averaged or per_batch dice loss')
parser.add_argument('--log_dice', type=str2bool, nargs='?', const=True, default=False, help='-log(dice) or 1-dice as the loss')
parser.add_argument('--top_k', type=int, default=10, help='percentage of considered pixels in topk loss')
parser.add_argument('--focal_alpha', type=float, default=-1, help='alpha weight for focal loss if used, -1 ignores it')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--min_lr', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--max_lr', type=float, default=3e-4, help='max learning rate')
parser.add_argument('--cycle_lens', type=str, default='1/12', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--optimizer', type=str, default='nadam', help='optimizer choice')
parser.add_argument('--metric', type=str, default='dice', help='which metric to use for monitoring progress (loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 480,640', type=str, default='480,640')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: 8]')


def get_loss(loss_name, batch_dice, log_dice, top_k, focal_alpha):
    from utils.losses import BinaryDiceLoss, TopKLoss, FocalLoss
    if loss_name == 'ce':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return BinaryDiceLoss(log_loss=log_dice, batch_dice=batch_dice)
    elif loss_name == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=2)
    elif loss_name == 'top_k':
        return TopKLoss(k=top_k)
    else:
        raise NotImplementedError("loss {} is not available".format(loss_name))


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run_one_epoch(loader, model, criterion1, criterion2, alpha_loss1, optimizer=None, scheduler=None, assess=False):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here
    if train:
        model.train()
    else:
        model.eval()
    if assess:
        dices = []
        alpha_loss1 = 1  # use only loss1 for evaluation

    print('loss = {:.2f}*{} + {:.2f}*{}'.format(alpha_loss1, type(criterion1).__name__,
                                                1-alpha_loss1, type(criterion2).__name__))
    with trange(len(loader)) as t:
        n_elems, running_loss1, running_loss2 = 0, 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits_aux, logits = model(inputs)

            loss_1 = criterion1(logits_aux.squeeze(), labels.squeeze().float()) + \
                     criterion1(logits.squeeze(), labels.squeeze().float())
            loss_2 = criterion2(logits_aux.squeeze(), labels.squeeze().float()) + \
                     criterion2(logits.squeeze(), labels.squeeze().float())
            loss = alpha_loss1 * loss_1 + (1-alpha_loss1)*loss_2

            if train:  # only in training mode
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if assess:
                # evaluation
                for i in range(len(logits)):
                    prediction = logits[i].sigmoid().detach().cpu().numpy()[-1]
                    target = labels[i].cpu().numpy()
                    thresh = 0.5
                    bin_pred = prediction > thresh
                    dice = dice_score(target.ravel(), bin_pred.ravel())
                    dices.append(dice)


            # Compute running loss
            n_elems += inputs.size(0)
            running_loss1 += loss_1.item() * inputs.size(0)/n_elems
            running_loss2 += loss_2.item() * inputs.size(0)/n_elems
            if train:
                t.set_postfix(tr_loss1_2_lr="{:.4f}/{:.4f}/{:.6f}".format(float(running_loss1), float(running_loss2), get_lr(optimizer)))
            else:
                t.set_postfix(vl_loss1_2="{:.4f}/{:.4f}".format(float(running_loss1), float(running_loss2)))
            t.update()

    if assess: return np.mean(dices), np.std(dices), running_loss1, running_loss2
    return None, None, None, None

def train_one_cycle(train_loader, model, criterion1, criterion2, mixture, optimizer=None, scheduler=None, cycle=0):
    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle + 1, epoch + 1, cycle_len))
        if mixture == 'linear':
            alpha_loss1 = 1-epoch/(cycle_len - 1)
        elif mixture == 'combo':
            alpha_loss1 = 0.5
        elif mixture == 'fine_tune_loss2':
            alpha_loss1 = 1 if epoch <= 0.9*cycle_len else 0
        elif mixture == 'only_loss1':
            alpha_loss1 = 1  # only loss1
        else:
            raise NotImplementedError("mixture strategy {} is not available".format(mixture))

        tr_mean_dice, tr_std_dice, tr_loss1, tr_loss2 = \
            run_one_epoch(train_loader, model, criterion1, criterion2, alpha_loss1, optimizer=optimizer, scheduler=scheduler, assess=False)

    return tr_mean_dice, tr_std_dice, tr_loss1, tr_loss2

def train_model(model, optimizer, criterion1, criterion2, mixture, train_loader, val_loader, scheduler, metric, exp_path):
    n_cycles = len(scheduler.cycle_lens)
    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle + 1, n_cycles))
        # prepare next cycle:
        # reset iteration counter
        scheduler.last_epoch = -1
        # update number of iterations
        scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader)

        # train one cycle
        _, _, _, _ = train_one_cycle(train_loader, model, criterion1, criterion2, mixture, optimizer, scheduler, cycle)

        with torch.no_grad():
            tr_mean_dice, tr_std_dice, tr_loss1, tr_loss2 = run_one_epoch(train_loader, model, criterion1, criterion2, mixture, assess=True)
            vl_mean_dice, vl_std_dice, vl_loss1, vl_loss2 = run_one_epoch(val_loader, model, criterion1, criterion2, mixture, assess=True)

        print('Train|Val Loss: {:.4f}/{:.4f}|{:.4f}/{:.4f} - Train|Val DICE: {:.2f}+-{:.2f}||{:.2f} +- {:.2f}'.format(
            tr_loss1, tr_loss2, vl_loss1, vl_loss2, 100*tr_mean_dice, 100*tr_std_dice, 100*vl_mean_dice, 100*vl_std_dice))
        # check if performance was better than anyone before and checkpoint if so
        if exp_path is not None:
            print(15 * '-', ' Checkpointing ', 15 * '-')
            save_model(exp_path, model, optimizer)
    del model
    torch.cuda.empty_cache()
    return tr_loss1, tr_loss2, vl_loss1, vl_loss2, tr_mean_dice, tr_std_dice, vl_mean_dice, vl_std_dice



if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    max_lr, min_lr, bs= args.max_lr, args.min_lr, args.batch_size

    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens) == 2:  # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path = osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path, 'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path = None

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=1, pretrained=True)

    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_seg_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs,
                                                         tg_size=tg_size, mean=mean, std=std, num_workers=args.num_workers)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
    elif optimizer_choice == 'sgd_mom':
        optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.6)

    else:
        sys.exit('please choose between adam or sgd optimizers')
    scheduler = CosineAnnealingLR(optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=min_lr)

    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    criterion1 = get_loss(args.loss1, args.batch_dice, args.log_dice, args.top_k, args.focal_alpha)
    criterion2 = get_loss(args.loss2, args.batch_dice, args.log_dice, args.top_k, args.focal_alpha)
    mixture = args.mixture
    print('* Instantiating loss functions {}/{}'.format(args.loss1, args.loss2))
    print('* Starting to train\n', '-' * 10)
    start = time.time()
    tr_loss1, tr_loss2, vl_loss1, vl_loss2, tr_mean_dice, tr_std_dice, vl_mean_dice, vl_std_dice = \
        train_model(model, optimizer, criterion1, criterion2, mixture, train_loader, val_loader, scheduler, metric, experiment_path)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
    print('Dice: %f' % vl_mean_dice)

    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('DICE={:.2f}+/-{:.2f}|{:.2f}+/-{:.2f}, Loss={:.2f}|{:.2f}/{:.2f}|{:.2f}'
                  .format(100*tr_mean_dice, 100*tr_std_dice, 100*vl_mean_dice, 100*vl_std_dice, tr_loss1, tr_loss2, vl_loss1, vl_loss2), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)
