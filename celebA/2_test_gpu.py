from __future__ import print_function
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
# import tensorflow as tf
#print('Tensorflow version', tf.__version__)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pandas as pd
#from __future__ import print_function
import os
import sys
import gzip
import json
import shutil
import zipfile
import argparse
import subprocess
from six.moves import urllib
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import pandas as pd

#pickle_file_path = 'dec11_mask.pkl'
#df = pd.read_pickle(pickle_file_path)
#df.to_csv(pickle_file_path[:-4]+'.csv', index=False)

import os
import pickle
import shutil
# import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import downscale_local_mean
import sys
# import glob
# from torchvision import transforms, datasets
# import torch
from PIL import Image
# import torch.nn.functional as F
from skimage.io import imread, imsave

from argparse import ArgumentParser


### uncertainty22 based variance ###

def normalize(a):
    return((a-np.min(a))/(np.max(a)-np.min(a)))

def main(n_top_=128*128, num_mea_=10, ad_hoc_=0):
    # dir = '/home/sunyang/code-cs-fairness-main/test_images/celebA/imgs'
    dir = '/home/sunyang/code-cs-fairness-main/datasets/celeba'
    file_list = os.listdir(dir)
    n_sample_ = len(file_list)
    n_sample_ = 2000
    images = {i: imread(os.path.join(dir, file_list[i]))[64:192, 64:192, :] for i in range(n_sample_)}
    
    x_mean, x2_mean = np.zeros((128,128,3)), np.zeros((128,128,3))
    for i in range(n_sample_):
        sample = images[i] / 255.
        x_mean = x_mean + sample #(128,128,3)
        x2_mean = x2_mean + np.square(sample) #(128,128,3)

    print('mean check', np.max(x_mean))
    print('mean_square check', np.max(x2_mean))
    x_mean = x_mean/n_sample_
    x2_mean = x2_mean/n_sample_
    var = x2_mean - np.square(x_mean) #(128,128,3)
    var_m_ = np.mean(var, axis=-1) #(128,128)


    def color_map(img, path):
        plt.figure(figsize=(8,5))
        plt.imshow(img)#, cmap='YlOrRd')
        plt.colorbar()
        plt.savefig(path)

    def mask_generation(n_top, num_mea, ad_hoc=0, var_m=var_m_):
        if n_top != 128*128:
            #find tops
            d1_idx = var_m.ravel().argsort()[-n_top:][::-1]
            d2idx = [(idx//128, idx%128) for idx in d1_idx]

            if ad_hoc == 0: #mnist
                #gausian_block
                mask = np.zeros((128,128,3))
                for (i,j) in d2idx:
                    mask[i,j,:] = 1

                # output
                color_map(mask[:,:,0], '../uncertainty2/mnist_var_top_{}_mask_color.jpg'.format(n_top))
                mask = mask.reshape((1,-1))
                np.save('../uncertainty2/mnist_var_top_{}_mask.npy'.format(n_top), mask)

            else: #celebA                
                if n_top == 1: #gaussian
                    A = np.zeros((128*128*3, num_mea))
                    for i in range(128*128):
                        A[3*i:3*i+3,:] = np.random.randn(num_mea) / np.sqrt(num_mea) 
                    np.save('../uncertainty2/celeb_gaussian_A_mea_{}.npy'.format(num_mea), A)

                else:
                    A = np.zeros((128*128*3, num_mea)) #extreme_block
                    adhoc_1_2 = np.load('../uncertainty2/celeb_var_top_1_2_adhoc.npy') #128,128
                    for i in range(128*128):
                        if adhoc_1_2.ravel()[i] == 0:
                            A[3*i:3*i+3,:] = np.random.randn(num_mea) / np.sqrt(num_mea) 
                        else:
                            if adhoc.ravel()[i] != 0:
                                A[3*i:3*i+3,:] = np.sqrt(128*128/(2*n_top)) * np.random.randn(num_mea) / np.sqrt(num_mea) 
                            else:
                                A[3*i:3*i+3,:] = 0 * ( np.random.randn(num_mea) + 3)
                    np.save('../uncertainty2/celeb_extreme_A_top_1_{}_mea_{}.npy'.format(int(128*128/n_top), num_mea), A)
                

            

        else:
            if ad_hoc == 0: #mnist
                # A
                mask = np.zeros((128,128))
                unit = (1/num_mea) * (128*128) / np.sum(var_m)
                for i in range(var_m.shape[0]):
                    for j in range(var_m.shape[1]):
                        mask[i,j] = np.sqrt(var_m[i,j] * unit)
                mask_reshape = np.reshape(mask, (-1))
                A = np.zeros((128*128*3,num_mea))
                for i in range(128*128):
                    A[3*i:3*i+3,:] = mask_reshape[i] * np.random.randn(num_mea)
                
                # output
                color_map(normalize(np.expand_dims(var_m, axis=-1)), '../uncertainty2/mnist_propotional_var_color.jpg')  
                np.save('../uncertainty2/mnist_propotional_A_mea_{}.npy'.format(num_mea), A)

            else: #celebA
                mask = np.zeros((128,128))
                adhoc = np.load('../uncertainty/celeb_var_top_1_2_adhoc.npy')[64:192,64:192]
                # var_m = var_m*adhoc
                # unit = (128*128*0.5/num_mea) / np.sum(var_m)
                var_m_norm = normalize(var_m*adhoc)
                var_m_norm = (var_m_norm+0.15)/(1+0.15) #aug14
                propotion = np.sum(adhoc)/(128*128)
                unit = (128*128*propotion/num_mea) / np.sum(var_m_norm*adhoc) #sep24: was  (128*128*0.5/num_mea) / np.sum(var_m_norm*adhoc)
                for i in range(var_m.shape[0]):
                    for j in range(var_m.shape[1]):
                        # if var_m[i,j] != 0:
                        #     mask[i,j] = np.sqrt(var_m[i,j] * unit)
                        # else:
                        #     mask[i,j] = 1 / np.sqrt(num_mea)
                        if adhoc[i,j] != 0:
                            mask[i,j] = np.sqrt(var_m_norm[i,j] * unit)
                        else:
                            mask[i,j] = 1 / np.sqrt(num_mea)

                mask_reshape = np.reshape(mask, (-1))
                A = np.zeros((128*128*3,num_mea))
                for i in range(128*128):
                    A[3*i:3*i+3,:] = mask_reshape[i] * np.random.randn(num_mea)

                # output
                color_map(adhoc, '../uncertainty2/celeb_adhoc_color.jpg')
                # color_map(normalize(var_m*adhoc)*adhoc, '../uncertainty2/celeb_propotional_var_color_unscale.jpg')
                color_map(var_m_norm*adhoc, '../uncertainty2/celeb_propotional_energy_map_scale_0_15.jpg') #aug14
                np.save('../uncertainty2/celeb_propotional_A_mea_{}.npy'.format(num_mea), A)
                np.save('../uncertainty2/celeb_propotional_variance_900.npy', (normalize(var_m)+0.15)/(1+0.15)/900)
                np.save('../uncertainty2/celeb_propotional_variance_2700.npy', (normalize(var_m)+0.15)/(1+0.15)/2700)
                np.save('../uncertainty2/celeb_propotional_variance_3900.npy', (normalize(var_m)+0.15)/(1+0.15)/3900)
                np.save('../uncertainty2/celeb_propotional_variance_1600.npy', (normalize(var_m)+0.15)/(1+0.15)/1600)


        
    mask_generation(n_top_, num_mea_, ad_hoc_)

    print('COMPLETED: n_top_: {}, num_mea_: {}, ad_hoc_: {}'.format(n_top_, num_mea_, ad_hoc_))



def block(i, j, num_mea, block_size=16):
    mask = np.ones((128,128)) * np.sqrt( 0.5*16384*(1/num_mea) / (16384-block_size**2) )
    mask[block_size*i:block_size*(i+1), block_size*j:block_size*(j+1)] = np.sqrt( 0.5*16384*(1/num_mea) / (block_size**2) )
    mask_reshape = np.reshape(mask, (-1))
    A = np.zeros((128*128*3,num_mea))
    for ele in range(128*128):
        A[3*ele:3*ele+3,:] = mask_reshape[ele] * np.random.randn(num_mea)
    np.save('../uncertainty2/celeb_block_{}_data_driven_A_i_{}_j_{}_mea_{}.npy'.format(block_size, i, j, num_mea), A)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pickle', type=str, default='nan')
    hparams = parser.parse_args()

    if hparams.pickle == 'nan':
        for mea in [6300]:
            # #gaussian
            # main(n_top_=1, num_mea_=mea, ad_hoc_=1)           
            #general
            main(n_top_=16384, num_mea_=mea, ad_hoc_=1)


        # for mea in [2700, 3900, 6300]:
        #     block_size = 16
        #     for i in range(int(128/block_size)):
        #         for j in range(int(128/block_size)):
        #             block(i, j, mea, block_size)
        #             print(f'({i},{j}) done')

    else:
        df = pd.read_pickle(hparams.pickle) #'to_produce_table_chart_aug19.pkl'
        csv_name = 'chart_'+'_'.join(hparams.pickle[:-4].split('_')[4:])+'.csv'
        df.to_csv(csv_name, index=False)
        print('csv done')


    # def get_l2_loss(image1, image2):
    #     """Get L2 loss between the two images"""
    #     assert image1.shape == image2.shape
    #     return np.mean((image1 - image2)**2)

    # def get_l2_loss_region(image1, image2): #aug9
    #     """Get L2 loss between regions"""
    #     mask_ = np.load('../uncertainty/celeb_var_top_1_2_adhoc.npy')[64:192,64:192] #(256,256)
    #     mask = np.repeat(mask_[:, :, np.newaxis], 3, -1)
    #     center_loss = np.mean((image1*mask - image2*mask)**2)
    #     corner_loss = np.mean((image1*np.where(mask == 1, 0, 1) - image2*np.where(mask == 1, 0, 1))**2)
    #     return center_loss, corner_loss

    # mea_img_path = 'C:/Users/User/Desktop/code-cs-fairness-main/3/full_gaussian/noise_8_0/measurement_2700/output_FULL_GAUSSIAN_measurement_2700_noise_8_0.jpg'
    # mea_img = imread(mea_img_path)[64:192,64:192,:]/256.

    # org_img_path = 'C:/Users/User/Desktop/code-cs-fairness-main/3/celebA_img_3_original_256.png'
    # org_img = imread(org_img_path)[64:192,64:192,:]/256.


    # print(get_l2_loss(mea_img,org_img))
    # print(get_l2_loss_region(mea_img,org_img))
