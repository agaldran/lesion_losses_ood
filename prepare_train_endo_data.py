import os, os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image

tg_size = (640,480)

path_ims = 'data_endotect/segmented-images/images/'
path_masks = 'data_endotect/segmented-images/masks/'
im_list = os.listdir(path_ims)
mask_list = os.listdir(path_masks)

im_list = sorted([osp.join(path_ims, n) for n in im_list])
mask_list = sorted([osp.join(path_masks, n) for n in mask_list])
n_samples = len(im_list)

path_ims_out = 'data_endotect/train/images/'
path_masks_out = 'data_endotect/train/masks_binary/'

os.makedirs(path_ims_out, exist_ok=True)
os.makedirs(path_masks_out, exist_ok=True)


im_list_out, m_list_out = [],[]

for i in tqdm(range(len(im_list))):
    im_name = im_list[i]
    m_name = mask_list[i]

    im_name_out = osp.join(path_ims_out, im_name.split('/')[-1])
    m_name_out = osp.join(path_masks_out, m_name.split('/')[-1])


    img = Image.open(im_name)
    mask = Image.open(m_name)
    mask = Image.fromarray(255*(np.array(mask)>127).astype(np.uint8)) # they are jpg with more than two values, why oh why?
    
    img = img.resize(tg_size, Image.BICUBIC)
    mask = mask.resize(tg_size, Image.NEAREST)

    im_list_out.append(im_name_out)
    m_list_out.append(m_name_out)

    img.save(im_name_out)
    mask.save(m_name_out)
    
        
data_tuples = list(zip(im_list_out, m_list_out))
df = pd.DataFrame(data_tuples, columns=['image_path','mask_path'])

df_other, df_val1 = train_test_split(df, test_size=n_samples//5, random_state=0)
df_other, df_val2 = train_test_split(df_other, test_size=n_samples//5, random_state=0)
df_other, df_val3 = train_test_split(df_other, test_size=n_samples//5, random_state=0)
df_val4, df_val5 = train_test_split(df_other, test_size=n_samples//5, random_state=0)

df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5])
df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5])
df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5])
df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5])
df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4])

df_train1.to_csv('data_endotect/train_f1.csv', index=None)
df_val1.to_csv('data_endotect/val_f1.csv', index=None)

df_train2.to_csv('data_endotect/train_f2.csv', index=None)
df_val2.to_csv('data_endotect/val_f2.csv', index=None)

df_train3.to_csv('data_endotect/train_f3.csv', index=None)
df_val3.to_csv('data_endotect/val_f3.csv', index=None)

df_train4.to_csv('data_endotect/train_f4.csv', index=None)
df_val4.to_csv('data_endotect/val_f4.csv', index=None)

df_train5.to_csv('data_endotect/train_f5.csv', index=None)
df_val5.to_csv('data_endotect/val_f5.csv', index=None)
