from scipy.io import loadmat, savemat
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
seed = np.random.RandomState(112311)



def plot_clean_noisy_img(filepath):
    Dt = loadmat(filepath)
    clean_img = Dt['clean']
    clean_img = np.sqrt(clean_img)
    plt.imshow(clean_img*255, cmap='gray', vmin=0, vmax=255)
    noisy_img = Dt['noisy']
    noisy_img = np.sqrt(noisy_img)
    plt.imshow(noisy_img*255, cmap='gray', vmin=0, vmax=255)
    

def generate_syn_dataset(savepath):
    save_path_train = savepath + 'train/'
    save_path_val = savepath + 'val/'

    path1 = './BSR_bsds500/BSR/BSDS500/data/images/train'  #path to BSD500 train images
    i=0
    for file in os.listdir(path1):
            if file.endswith(".jpg"):
                i= i+1
                print(i)
                im_file = os.path.join(path1, file)
                img = cv2.imread(im_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(gray.shape)
                data_save_path = save_path_train +'trn_' + file[:-4] + '.mat'
                
                dat = dict()
                dat['clean'] = ((np.float32(gray)+1.0)/256.0)**2
                dat['noisy'] = dat['clean'] * seed.gamma(size=dat['clean'].shape, shape=1.0, scale=1.0).astype(dat['clean'].dtype)


                savemat(data_save_path, dat)

    path3 = './BSR_bsds500/BSR/BSDS500/data/images/test'  # path to BSD500 test images
    for file in os.listdir(path3):
            if file.endswith(".jpg"):
                i= i+1
                print(i)
                
                im_file = os.path.join(path3, file)
                print(im_file)
                img = cv2.imread(im_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(gray.shape)
                data_save_path = save_path_train + 'tst_'+ file[:-4] + '.mat'
                
                dat = dict()
                dat['clean'] = ((np.float32(gray)+1.0)/256.0)**2
                dat['noisy'] = dat['clean'] * seed.gamma(size=dat['clean'].shape, shape=1.0, scale=1.0).astype(dat['clean'].dtype)


                savemat(data_save_path, dat)

    path2 = './BSR_bsds500/BSR/BSDS500/data/images/val' # path to BSD500 val images
    L_file = [];
    for file in os.listdir(path2):
            if file.endswith(".jpg"):
                L_file.append(file)

    val_list = random.sample(L_file, 50)
    
    for file in L_file:
        if file in val_list:
            im_file = os.path.join(path2, file)
            img = cv2.imread(im_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            data_save_path = save_path_val + 'val_'+ file[:-4] + '.mat'
            
            dat = dict()
            dat['clean'] = ((np.float32(gray)+1.0)/256.0)**2
            dat['noisy'] = dat['clean'] * seed.gamma(size=dat['clean'].shape, shape=1.0, scale=1.0).astype(dat['clean'].dtype)


            savemat(data_save_path, dat)
        else:
            i= i+1
            print(i)
            im_file = os.path.join(path2, file)
            img = cv2.imread(im_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray.shape)
            data_save_path = save_path_train + 'val_'+ file[:-4] + '.mat'
            
            dat = dict()
            dat['clean'] = ((np.float32(gray)+1.0)/256.0)**2
            dat['noisy'] = dat['clean'] * seed.gamma(size=dat['clean'].shape, shape=1.0, scale=1.0).astype(dat['clean'].dtype)


            savemat(data_save_path, dat)

                


if __name__ == '__main__':
    
    ### uncomment below lines if a new dataset generation is needed

    savepath = 'Path to save the dataset'
    # generate_syn_dataset(savepath)