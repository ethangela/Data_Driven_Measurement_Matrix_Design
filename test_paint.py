import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from collections import defaultdict
# from argparse import ArgumentParser
# from skimage.io import imread, imsave
# import torch.nn as nn 
# import torch.optim as optim
# import random

def kernel(x1, x2, l=0.5, sigma_f=0.2):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

def sample_from_2D_Gaussian(size):
    m = size**2
    test_d1 = np.linspace(-10, 10, m)
    test_d2 = np.linspace(-10, 10, m)
    
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
    test_X = np.asarray(test_X)
    mu = np.zeros_like(test_d1)
    cov = kernel(test_X, test_X)
    print('parameter set done')
    
    gp_samples = np.random.multivariate_normal(
            mean = mu.ravel(), 
            cov = cov, 
            size = 1)
    z = gp_samples.reshape(test_d1.shape)
    print('sampling done')
    
    #scale to range(0,1)
    z = (z - np.min(z))/np.ptp(z)
    np.save('2D.npy', z)
    #print(z)
    test_d1 = (test_d1 - np.min(test_d1))/np.ptp(test_d1)
    test_d2 = (test_d2 - np.min(test_d2))/np.ptp(test_d2)
    print('scaling done')
    
    fig = plt.figure(figsize=(5, 5))
    plt.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=1)
    #ax.set_title("with optimization l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    #plt.show()
    plt.savefig('Jun25.jpg')
    
def excel(path):
    df = pd.read_pickle(path)
    csv_title = path[:-3] + 'csv'
    df.to_csv(csv_title,index=0)
    pass








def plot_lay(lay_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))
 
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 256) & (df['input_size'] == 768) & (df['filter_size'] == 4)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='layers')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(lay_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('layers')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('layers for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/layer/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/layer_for_{}_on_{}_signal.png'.format(mask_inf, type))

def plot_chn(chn_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path) 
    plt.figure(figsize=(16,10))
    
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']    

    if mask_inf != 'circulant':
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='channels')
    y_avg = []
    for sgl in sgl_list:
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(chn_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(chn_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    
    
    plt.xlabel('channels')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('channels for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()
    
    path_dir = 'figures/channel/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/channel_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_ipt(ipt_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']#, '1D_rbf_3.0_4.npy', '1D_rbf_3.0_5.npy'] ##########

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['channels'] == 6) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='input_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(ipt_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(ipt_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('input_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('input_size for {} on signal {}'.format(mask_inf, type))    
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/input_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/input_size_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_fit(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6  )] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='filter_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        if len(psnr) > 10:
            p = []
            l = int(len(psnr)/3)
            for i in range(3):
                p.append(psnr[i*l:(i+1)*l])
            psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(fit_size, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(fit_size, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('filter_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('filter_size for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/filter_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/filter_size_for_{}_on_{}_signal.png'.format(mask_inf, type))





def plot_activation(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10)] #a set of channels on 3 signals THREE TIMES?#########

    act_list = ['relu', 'leaky_relu', 'sigmoid']
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(act_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(act_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('act_functions')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('act_function for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/act_function/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/act_function{}_on_{}_signal.png'.format(mask_inf, type))



def picture(path, mask=None, type='block'):
    plt.figure(figsize=(80,10))
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)
    fs = np.load(path)
    print('signal',fs.shape)
    title = path[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    
    if mask:
        m = np.load('mask/1D_mask_'+type+'_4096_'+str(mask)+'_1.npy')
        fs = fs * m
        fs[fs == 0.] = np.nan
        title += '_masked_'+type+'_'+str(mask)
    
    plt.plot(xs, fs, 'gray') # Plot the samples
    plt.savefig(tn + '.jpg')
    plt.close()


def group_pic(org, msk, path, title_m):
    org1, org2, org3 = org
    path1, path2, path3 = path

    plt.figure(figsize=(240,30))
    
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)

    # original
    plt.subplot(331)
    fs = np.load(org1)
    plt.title(org1[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(334)
    fs = np.load(org2)
    plt.title(org2[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(337)
    fs = np.load(org3)
    plt.title(org3[:-4])
    plt.plot(xs, fs, 'gray')


    # masked
    plt.subplot(332)
    m = np.load(msk)
    fs = np.load(org1)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(335)
    m = np.load(msk)
    fs = np.load(org2)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(338)
    m = np.load(msk)
    fs = np.load(org3)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')


    # recovered
    plt.subplot(333)
    fs = np.load(path1)
    title = path1[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(336)
    fs = np.load(path2)
    title = path2[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(339)
    fs = np.load(path3)
    title = path3[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.savefig(title_m + '.jpg')
    print('saved')
    plt.close()


def ipt(hparams):
    o = 4096 / (pow(hparams.f, hparams.l))
    print(o)



def plot_jan(csv_path, mea_type, mea_range, sig_type, mask_inf):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,5))
 
    if sig_type == 'exp':
        sgl_list = ['1D_exp_0.25_4096_10.npy', '1D_exp_0.25_4096_11.npy', '1D_exp_0.25_4096_12.npy', '1D_exp_0.25_4096_13.npy', '1D_exp_0.25_4096_14.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_4096_10.npy', '1D_rbf_3.0_4096_11.npy', '1D_rbf_3.0_4096_12.npy', '1D_rbf_3.0_4096_13.npy', '1D_rbf_3.0_4096_14.npy'] 

    df_sel = df

    # if mea_type == 'layers':
    #     #lay_list = [1,2,3,4,5,6,7,8,9,10,12,16,20,25,30,35,40,50,60,70,85,100,120,140,180,240] #3,3,7
    # elif mea_type == 'channels':
    #     #lay_list = [50,80,120,140,160,180,200,220,260,300,350,400,450,500,600,700,850,1000]#160,180,160
    # elif mea_type == 'input_size':
    #     #lay_list = [30,50,70,80,90,95,100,105,110,120,130,140,150,180,200,240,330,384,420,480,520]#120,200,384

    lay_list = mea_range

    df_sel = df_sel.sort_values(by = mea_type)##### e.g.'layers'
    y_avg = []
    for i,sgl in enumerate(sgl_list):
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        label_title = 'testing signal {}'.format(int(i+1))
        plt.plot(lay_list, psnr, '--', label = label_title, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z+e+f)/5 for x,y,z,e,f in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel(mea_type)#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    mask_info_ls = mask_inf.split('_') #'random_8', 'block_2', 'denoise_0.05', 'circulant_100'
    if mask_info_ls[0] == 'random' or mask_info_ls[0] == 'block':
        mask_info = '1_{} of entries are {}-masked'.format(int(mask_info_ls[1]), mask_info_ls[0])
    elif mask_info_ls[0] == 'denoise':
        mask_info = 'medium noise added'
    elif mask_info_ls[0] == 'circulant':
        mask_info = 'compressing at ratio of {}_4096'.format(int(mask_info_ls[1]))
    plot_title =  mea_type + ' for '+ sig_type +' signals with measurement as '+ mask_info
    #plt.title(plot_title)
    # show a legend on the plot
    plt.legend()

    path_dir = 'feb6_hyper_figures/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    img_name = plot_title +'.jpg'
    final = os.path.join(path_dir, img_name)
    plt.savefig(final)


def qplot(sig_type, plot_title, plot_path=None): 
    plt.figure(figsize=(14,4))
    xs = np.linspace(-10,10,4096) 
    xs = (xs - np.min(xs))/np.ptp(xs)
    if sig_type == 'exp':
        path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
    elif sig_type == 'rbf':
        path = 'Gaussian_signal/1D_rbf_3.0_4096_30.npy'
    elif sig_type == 'recover':
        path = 'result/' + plot_path
    fs = np.load(path)
    plt.plot(xs, fs, 'gray')
    plt.savefig('feb5_figures/{}.jpg'.format(plot_title))
    print('single plot saved')
    plt.close()

def oplot(sig_type, mea_path, plot_title): 
    plt.figure(figsize=(14,4))
    xs = np.linspace(-10,10,4096) 
    xs = (xs - np.min(xs))/np.ptp(xs)
    if sig_type == 'exp':
        path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
    elif sig_type == 'rbf':
        path = 'Gaussian_signal/1D_rbf_3.0_4096_30.npy'
    fs = np.load(path)
    if mea_path != 'denoise':
        ms = np.load(mea_path)
        ms = ms.reshape(4096)
        ys = fs * ms
        ys[ys == 0.] = np.nan
    else:
        ys = fs + 0.05 * np.random.randn(4096)
    plt.plot(xs, ys, 'gray')
    plt.savefig('feb5_figures/{}.jpg'.format(plot_title))
    print('single plot saved')
    plt.close()


def main(hparams):
    if hparams.plot == 'arch':
        plot_jan(hparams.csv, 
        hparams.mea_type, 
        hparams.mea_range, 
        hparams.sig_type,
        hparams.mask_inf)
    elif hparams.plot == 'original':
        qplot(hparams.sig_type,
        hparams.title)
    elif hparams.plot == 'observe':
        oplot(hparams.sig_type,
        hparams.mea_path,
        hparams.title)
    elif hparams.plot == 'recover':
        qplot(hparams.sig_type,
        hparams.title,
        hparams.recover_path)

    print('ploting completed')



if __name__ == '__main__':    
    # PARSER = ArgumentParser()
    # PARSER.add_argument('--csv', type=str, default='inpaint_lay_rbf_block_14.pkl', help='path stroing pkl')
    # PARSER.add_argument('--mea_type', type=str, default='layers', help='layers/channels/input_size/filter_size/step_size')
    # PARSER.add_argument('--mea_range', type=float, nargs='+', default=[1,2,3,4,5,6,7,8,9,10,12,16,20], help='a list of xs')
    # PARSER.add_argument('--sig_type', type=str, default='exp', help='exp/rbf')
    # PARSER.add_argument('--mask_inf', type=str, default='random_8', help='mask info, e.g., block_2, denoise_0.05, circulant_100') 
    # PARSER.add_argument('--plot', type=str, default='arch') 
    # PARSER.add_argument('--mea_path', type=str, default='Masks/1D_mask_random_64_2_1.npy') 
    # PARSER.add_argument('--title', type=str, default='original_exp') 
    # PARSER.add_argument('--recover_path', type=str, default='recover_path') 
    
    # HPARAMS = PARSER.parse_args()
    
    # main(HPARAMS)

    # a = np.random.randint(5, size=(5,2))
    # print('a:')
    # print(a)
    # dim = 2
    # for axis in range(dim):
    #     print('axis is {}'.format(axis))
    #     zero_shape = list(a.shape) #[5,1]
    #     zero_shape[axis] = 1
    #     d1 = np.diff(a, axis=axis)
    #     print('d1:')
    #     print(d1)
    #     d2 = np.zeros(zero_shape)
    #     print('d2:')
    #     print(d2)
    #     diff = np.concatenate((d1,d2),axis=axis)
    #     print('diff:')
    #     print(diff)


    # def load_img(path, img_name):
    #     img_path = os.path.join(path, img_name)
    #     img = imread(img_path)
    #     img = np.transpose(img, (1, 0, 2))
    #     img = np.pad(img, ((23,23), (3,3), (0,0)), 'constant', constant_values=0)
    #     #img = img[None, :, :, :]
    #     img_clean = img / 255.
    #     return img_clean

    # img = load_img('Celeb_signal', '182649.jpg')
    # print(img.shape)

    # img = img * 255.
    # img = np.transpose(img, (1, 0, 2))
    # imsave(os.path.join('Celeb_signal', 'test.jpg'), img.astype(np.uint8))


    #generate basis
    #def generate_basis():
    #    """generate the basis"""
    #    x = np.zeros((224, 224)) ##########################!!!!!!!!!!!!!!!!!#########
    #    coefs = pywt.wavedec2(x, 'db1')
    #    n_levels = len(coefs)
    #    basis = []
    #    for i in range(n_levels):
    #        coefs[i] = list(coefs[i])
    #        n_filters = len(coefs[i])
    #        for j in range(n_filters):
    #            for m in range(coefs[i][j].shape[0]):
    #                try:
    #                    for n in range(coefs[i][j].shape[1]):
    #                        coefs[i][j][m][n] = 1
    #                        temp_basis = pywt.waverec2(coefs, 'db1')
    #                        basis.append(temp_basis)
    #                        coefs[i][j][m][n] = 0
    #                except IndexError:
    #                    coefs[i][j][m] = 1
    #                    temp_basis = pywt.waverec2(coefs, 'db1')
    #                    basis.append(temp_basis)
    #                    coefs[i][j][m] = 0
    #
    #    basis = np.array(basis)
    #    return basis
        
    # #basis = generate_basis()
    # #np.save('./wavelet_basis.npy', basis) #
    # #print(basis.shape)

    # #gt1
    # predict1 =[28.05742073059082, 28.18284797668457, 27.38011360168457, 29.36186408996582, 28.98558235168457, 28.15776252746582, 29.43712043762207, 31.84532356262207, 31.39378547668457, 28.50895881652832, 26.57737922668457, 26.85331916809082, 26.92857551574707, 26.92857551574707, 27.45536994934082, 28.83506965637207, 27.78148078918457, 27.58079719543457, 25.87498664855957, 28.05742073059082, 25.034624099731445, 25.360734939575195, 27.95707893371582, 25.32310676574707, 25.27293586730957, 24.006120681762695, 22.78947639465332, 23.404069900512695, 24.01866340637207, 24.934282302856445, 23.21592903137207, 23.353899002075195, 22.21251106262207, 22.18742561340332, 22.91490364074707, 28.23301887512207, 24.708513259887695, 25.059709548950195, 24.031206130981445, 26.07567024230957, 27.17943000793457, 26.85331916809082, 27.98216438293457, 29.41203498840332, 30.16459846496582, 31.77006721496582, 30.89207649230957, 31.46904182434082, 30.96733283996582, 30.16459846496582, 27.50554084777832, 26.85331916809082, 24.382402420043945, 25.134965896606445, 27.40519905090332, 29.21135139465332, 29.01066780090332, 28.33336067199707, 30.76664924621582, 30.41545295715332, 29.26152229309082, 29.76323127746582, 27.85673713684082, 25.67430305480957, 29.18626594543457, 31.77006721496582, 30.96733283996582, 29.66288948059082, 28.55912971496582, 26.92857551574707, 24.09391975402832, 24.11900520324707, 23.94340705871582, 26.90349006652832, 28.35844612121582, 28.63438606262207, 28.55912971496582, 28.23301887512207, 28.38353157043457, 28.13267707824707, 28.58421516418457, 31.24327278137207, 32.17143630981445, 32.6480598449707, 33.22502517700195, 34.73015213012695, 32.87382888793945, 32.8487434387207, 30.66630744934082, 27.07908821105957, 25.97532844543457, 26.57737922668457, 26.45195198059082, 25.360734939575195, 25.19767951965332, 24.89665412902832, 21.15892219543457, 22.88981819152832, 22.325395584106445, 26.30143928527832, 23.16575813293457, 21.898942947387695, 23.84306526184082, 21.347063064575195, 21.798601150512695, 21.66063117980957, 23.353899002075195, 21.83622932434082, 21.698259353637695, 20.55687141418457, 21.13383674621582, 21.648088455200195, 21.76097297668457, 23.755266189575195, 25.09733772277832, 24.47020149230957, 23.04033088684082, 23.44169807434082, 22.526079177856445, 21.999284744262695, 24.532915115356445, 26.30143928527832, 24.64579963684082, 24.006120681762695, 21.73588752746582, 25.29802131652832, 21.798601150512695, 20.60704231262207, 26.65263557434082, 25.461076736450195, 23.89323616027832, 22.802019119262695, 20.368730545043945, 24.407487869262695, 23.41661262512207, 21.096208572387695, 20.795183181762695, 25.39836311340332, 22.225053787231445, 20.092790603637695, 22.53862190246582, 22.877275466918945, 24.608171463012695, 25.67430305480957, 28.96049690246582, 28.58421516418457, 31.21818733215332, 29.06083869934082, 27.93199348449707, 25.84990119934082, 23.59221076965332, 22.63896369934082, 22.977617263793945, 22.400651931762695, 22.425737380981445, 23.64238166809082, 20.820268630981445, 21.38469123840332, 21.848772048950195, 24.77122688293457, 24.36985969543457, 24.34477424621582, 24.833940505981445, 24.683427810668945, 27.20451545715332, 26.72789192199707, 25.59904670715332, 23.94340705871582, 26.90349006652832, 27.70622444152832, 24.733598709106445, 25.37327766418457, 22.174882888793945, 21.71080207824707, 21.547746658325195, 22.31285285949707, 20.48161506652832, 20.920610427856445, 21.05858039855957, 22.475908279418945, 21.98674201965332, 20.00499153137207, 20.88298225402832, 23.153215408325195, 21.798601150512695, 19.70396614074707, 21.43486213684082, 21.798601150512695, 19.17717170715332, 18.888689041137695, 19.57853889465332, 19.22734260559082, 18.65037727355957, 17.634416580200195, 18.412065505981445, 18.09849739074707, 17.207963943481445, 18.17375373840332, 17.67204475402832, 17.333391189575195, 18.060869216918945, 19.239885330200195, 18.55003547668457, 20.33110237121582, 19.02665901184082, 18.90123176574707, 19.85447883605957, 17.935441970825195, 18.14866828918457, 20.20567512512207, 19.92973518371582, 21.61046028137207, 19.62870979309082, 20.042619705200195, 17.684587478637695, 18.713090896606445, 22.08708381652832, 24.14409065246582, 22.601335525512695, 20.35618782043457, 18.90123176574707, 21.91148567199707, 21.572832107543945, 20.73246955871582, 20.40635871887207, 19.85447883605957, 19.05174446105957, 18.537492752075195, 18.32426643371582]
    # target1 =[29.25, 29.149999618530273, 30.559999465942383, 29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727, 19.100000381469727]
    # date =['20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401', '20210405', '20210406', '20210407', '20210408', '20210409', '20210412', '20210413', '20210414', '20210415', '20210416', '20210419', '20210420', '20210421', '20210422', '20210423', '20210426', '20210427', '20210428', '20210429', '20210430', '20210503', '20210504', '20210505', '20210506', '20210507', '20210510', '20210511', '20210512', '20210513', '20210514', '20210517', '20210518', '20210519', '20210520', '20210521', '20210524', '20210525', '20210526', '20210527']

    # #gt2
    # predict2 = [28.388551712036133, 30.76468849182129, 31.758729934692383, 32.19815444946289, 29.736597061157227, 31.003854751586914, 28.733064651489258, 31.835603713989258, 30.776430130004883, 27.2210636138916, 26.087984085083008, 27.05094337463379, 30.045167922973633, 29.34593391418457, 26.924951553344727, 29.306665420532227, 27.603540420532227, 24.95157814025879, 26.279695510864258, 25.749635696411133, 26.550302505493164, 26.33403205871582, 25.558870315551758, 24.674448013305664, 23.520723342895508, 23.5073184967041, 23.3260440826416, 22.446840286254883, 25.709062576293945, 23.5073184967041, 23.165056228637695, 22.939294815063477, 23.13006019592285, 22.40757179260254, 23.439577102661133, 24.547151565551758, 23.4650821685791, 22.50212287902832, 23.8276309967041, 24.961069107055664, 27.365678787231445, 27.187013626098633, 26.72991371154785, 26.156312942504883, 27.485734939575195, 29.822603225708008, 31.260339736938477, 31.154993057250977, 29.279138565063477, 27.970003128051758, 25.860559463500977, 25.7077579498291, 25.606325149536133, 27.28880500793457, 28.952302932739258, 26.535234451293945, 28.85869789123535, 28.478471755981445, 29.587709426879883, 30.202829360961914, 28.743860244750977, 27.970361709594727, 27.057466506958008, 30.577836990356445, 29.54583168029785, 31.92754554748535, 29.17901039123535, 28.377038955688477, 28.29850196838379, 24.817758560180664, 23.776620864868164, 24.391740798950195, 26.570589065551758, 26.170076370239258, 27.05710792541504, 26.60392189025879, 26.94487953186035, 26.33201026916504, 26.745927810668945, 27.471025466918945, 27.656572341918945, 30.143632888793945, 30.177324295043945, 32.00050735473633, 31.053560256958008, 30.860185623168945, 32.791683197021484, 31.91117286682129, 29.612268447875977, 29.740869522094727, 28.00571632385254, 29.811220169067383, 28.426156997680664, 26.075525283813477, 27.06565284729004, 24.689516067504883, 22.71092414855957, 23.090791702270508, 23.991228103637695, 24.21995735168457, 23.693811416625977, 23.285470962524414, 23.051523208618164, 21.126256942749023, 21.45279884338379, 21.161073684692383, 21.957353591918945, 21.485185623168945, 22.134355545043945, 21.44200325012207, 21.492067337036133, 20.098344802856445, 20.1307315826416, 20.200429916381836, 21.40012550354004, 22.089509963989258, 22.704988479614258, 22.477922439575195, 20.683866500854492, 21.050508499145508, 24.043901443481445, 23.145715713500977, 23.12151527404785, 22.12486457824707, 20.63872718811035, 22.40757179260254, 22.104578018188477, 20.861520767211914, 23.612306594848633, 22.802148818969727, 23.145715713500977, 20.489187240600586, 19.921995162963867, 22.145151138305664, 21.29442024230957, 19.940797805786133, 20.564756393432617, 23.6126651763916, 21.957353591918945, 20.304651260375977, 21.02138328552246, 22.169710159301758, 24.519983291625977, 23.95456886291504, 29.521272659301758, 28.270029067993164, 28.168596267700195, 29.3076114654541, 26.670717239379883, 24.470277786254883, 23.428781509399414, 23.622156143188477, 23.49913215637207, 23.49094581604004, 22.783525466918945, 21.885698318481445, 20.573774337768555, 21.249101638793945, 20.88738441467285, 22.994577407836914, 22.695497512817383, 22.57377815246582, 24.667207717895508, 23.539705276489258, 26.79765510559082, 26.30650520324707, 24.909700393676758, 23.927759170532227, 24.973169326782227, 26.923288345336914, 24.68132972717285, 24.838045120239258, 22.79728889465332, 22.26426124572754, 21.8421573638916, 21.92069435119629, 20.616777420043945, 22.427858352661133, 21.316011428833008, 23.28416633605957, 22.652315139770508, 21.66124153137207, 21.473085403442383, 22.384675979614258, 21.893884658813477, 20.46498680114746, 22.126169204711914, 21.221460342407227, 19.501489639282227, 19.011350631713867, 19.627954483032227, 19.72315788269043, 19.126840591430664, 18.312410354614258, 18.893545150756836, 18.165006637573242, 17.685483932495117, 18.025136947631836, 18.25926399230957, 17.747175216674805, 18.37315559387207, 18.634450912475586, 18.64720344543457, 19.759523391723633, 20.041399002075195, 19.783723831176758, 19.579553604125977, 18.9858455657959, 18.54084587097168, 19.95485496520996, 20.35435676574707, 19.72956657409668, 20.23394203186035, 19.59165382385254, 17.869905471801758, 18.25748634338379, 20.173261642456055, 22.8496036529541, 21.858171463012695, 20.558053970336914, 19.333505630493164, 20.313962936401367, 22.401994705200195, 20.1408748626709, 20.34718132019043, 19.768362045288086, 19.44834327697754, 19.49022102355957, 18.799467086791992]
    # target2 = [29.149999618530273, 30.559999465942383, 29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727, 19.100000381469727, 19.459999084472656]
    # date= ['20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401', '20210405', '20210406', '20210407', '20210408', '20210409', '20210412', '20210413', '20210414', '20210415', '20210416', '20210419', '20210420', '20210421', '20210422', '20210423', '20210426', '20210427', '20210428', '20210429', '20210430', '20210503', '20210504', '20210505', '20210506', '20210507', '20210510', '20210511', '20210512', '20210513', '20210514', '20210517', '20210518', '20210519', '20210520', '20210521', '20210524', '20210525', '20210526', '20210527']

    # #gt3
    # predict3 = [29.350664138793945, 31.620187759399414, 30.968957901000977, 34.09701156616211, 28.31956672668457, 28.24033546447754, 27.44041633605957, 28.431360244750977, 32.36366653442383, 27.01060676574707, 26.273630142211914, 28.64408302307129, 32.16829299926758, 30.539148330688477, 30.088716506958008, 30.1071720123291, 27.76926612854004, 24.39592933654785, 25.81016731262207, 24.438264846801758, 26.229127883911133, 27.555452346801758, 29.08257484436035, 27.288454055786133, 24.543542861938477, 24.992883682250977, 24.624940872192383, 24.174509048461914, 27.789888381958008, 22.416208267211914, 20.34856605529785, 20.849462509155273, 23.729501724243164, 23.76206398010254, 25.386857986450195, 25.087297439575195, 22.052583694458008, 22.98710060119629, 23.196596145629883, 23.51676368713379, 26.221521377563477, 26.665437698364258, 26.87925148010254, 26.4885196685791, 28.145891189575195, 31.60282325744629, 32.06953048706055, 31.52250099182129, 30.51310157775879, 29.21282386779785, 25.488908767700195, 25.664735794067383, 26.478761672973633, 28.471513748168945, 27.156038284301758, 27.411096572875977, 27.602121353149414, 28.57353401184082, 28.95232582092285, 27.59453010559082, 26.398454666137695, 28.15349769592285, 26.777246475219727, 28.37491798400879, 30.310136795043945, 30.640092849731445, 27.667261123657227, 27.277605056762695, 26.71647071838379, 23.999773025512695, 23.628572463989258, 24.97551918029785, 28.42917823791504, 30.87778663635254, 30.073518753051758, 29.675195693969727, 27.932077407836914, 26.951982498168945, 26.71971321105957, 27.17014503479004, 27.59995460510254, 31.189287185668945, 31.0058536529541, 34.62667465209961, 32.80867385864258, 31.893701553344727, 32.97256088256836, 32.20737075805664, 29.355012893676758, 29.341981887817383, 29.550371170043945, 31.095949172973633, 28.47477149963379, 26.333330154418945, 28.442209243774414, 24.58262062072754, 22.804773330688477, 23.009904861450195, 24.949472427368164, 23.780519485473633, 25.068857192993164, 22.885095596313477, 20.824506759643555, 22.06779670715332, 22.4617862701416, 22.764604568481445, 24.309091567993164, 24.896272659301758, 23.207429885864258, 21.4963436126709, 20.98349952697754, 20.167844772338867, 19.72608757019043, 20.991098403930664, 21.268407821655273, 21.909318923950195, 22.401010513305664, 22.209985733032227, 21.971200942993164, 23.563447952270508, 25.458498001098633, 26.017465591430664, 27.192934036254883, 24.540285110473633, 21.782350540161133, 20.668203353881836, 22.171998977661133, 21.741106033325195, 21.761720657348633, 22.069963455200195, 25.64085578918457, 21.849641799926758, 20.32740592956543, 21.669466018676758, 22.93284034729004, 21.462705612182617, 22.465044021606445, 22.05802345275879, 21.02095603942871, 19.802621841430664, 21.39215660095215, 24.463220596313477, 25.53556251525879, 24.098520278930664, 29.25297737121582, 29.28879737854004, 30.719324111938477, 29.9269962310791, 29.391908645629883, 25.994691848754883, 23.5819034576416, 23.281251907348633, 24.00736427307129, 24.24614906311035, 26.707773208618164, 22.945871353149414, 22.821046829223633, 24.427400588989258, 25.491060256958008, 21.848543167114258, 21.533784866333008, 22.09166145324707, 23.550416946411133, 23.36699867248535, 24.559816360473633, 25.28483772277832, 24.94730567932129, 24.084428787231445, 24.933183670043945, 26.899873733520508, 24.988550186157227, 24.837678909301758, 23.795717239379883, 23.001222610473633, 23.39195442199707, 23.44188117980957, 23.203096389770508, 23.603601455688477, 22.770029067993164, 23.82718849182129, 23.62639045715332, 21.66838264465332, 21.8322696685791, 22.997949600219727, 22.36844825744629, 20.732789993286133, 21.25810432434082, 20.31382942199707, 19.996896743774414, 19.716867446899414, 19.987123489379883, 20.018056869506836, 18.90825843811035, 18.38673210144043, 18.467592239379883, 18.67923927307129, 18.280363082885742, 19.0623722076416, 18.57667350769043, 18.52945899963379, 19.090593338012695, 18.446969985961914, 16.83680534362793, 19.13834571838379, 19.40860939025879, 20.172178268432617, 20.55531883239746, 19.507394790649414, 18.559309005737305, 19.50412940979004, 21.55494499206543, 20.59004783630371, 20.205286026000977, 19.922006607055664, 18.953306198120117, 19.8123722076416, 19.883466720581055, 23.527612686157227, 22.193696975708008, 20.70456886291504, 19.294126510620117, 19.367372512817383, 20.46903419494629, 20.887449264526367, 22.060205459594727, 20.472299575805664, 21.130029678344727, 20.645952224731445, 20.77836036682129]
    # target3 = [30.559999465942383, 29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727, 19.100000381469727, 19.459999084472656, 19.049999237060547]
    # date= ['20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401', '20210405', '20210406', '20210407', '20210408', '20210409', '20210412', '20210413', '20210414', '20210415', '20210416', '20210419', '20210420', '20210421', '20210422', '20210423', '20210426', '20210427', '20210428', '20210429', '20210430', '20210503', '20210504', '20210505', '20210506', '20210507', '20210510', '20210511', '20210512', '20210513', '20210514', '20210517', '20210518', '20210519', '20210520', '20210521', '20210524', '20210525', '20210526', '20210527']

    # #gt4
    # predict4 = [29.255207061767578, 30.21633529663086, 30.213077545166016, 30.508838653564453, 29.528522491455078, 28.464221954345703, 28.669116973876953, 30.192081451416016, 30.734729766845703, 28.769031524658203, 27.938587188720703, 29.236743927001953, 30.47842788696289, 29.68020248413086, 28.735363006591797, 29.024608612060547, 28.631107330322266, 26.886592864990234, 25.565265655517578, 23.528972625732422, 24.98098373413086, 23.84572982788086, 25.098636627197266, 23.586170196533203, 22.88315200805664, 22.18990707397461, 22.097957611083984, 21.766719818115234, 24.017681121826172, 21.773235321044922, 21.407245635986328, 21.382991790771484, 21.670787811279297, 21.312763214111328, 21.760929107666016, 23.617664337158203, 23.862743377685547, 23.183979034423828, 22.01578140258789, 22.665584564208984, 23.03084945678711, 23.300182342529297, 25.061710357666016, 25.189136505126953, 25.988086700439453, 29.286334991455078, 28.876544952392578, 29.587528228759766, 28.937725067138672, 28.920711517333984, 27.184162139892578, 27.25113296508789, 27.13094711303711, 26.854373931884766, 27.902385711669922, 27.663822174072266, 28.104022979736328, 27.496212005615234, 27.87197494506836, 27.247150421142578, 26.266834259033203, 26.188640594482422, 25.399463653564453, 25.744457244873047, 27.033565521240234, 27.374576568603516, 26.37398910522461, 26.32077407836914, 25.82663345336914, 24.43362808227539, 23.549243927001953, 24.589290618896484, 24.81735610961914, 26.12094497680664, 27.108501434326172, 26.391727447509766, 26.781970977783203, 25.96854019165039, 26.344303131103516, 24.504581451416016, 23.965190887451172, 27.334392547607422, 28.677440643310547, 31.532230377197266, 31.81351089477539, 31.292583465576172, 31.405529022216797, 31.348331451416016, 30.680789947509766, 29.78989028930664, 28.745861053466797, 29.71169662475586, 29.019901275634766, 27.983837127685547, 28.90768051147461, 26.485851287841797, 25.118549346923828, 24.90170669555664, 24.543682098388672, 25.96347427368164, 24.767040252685547, 23.307788848876953, 22.870845794677734, 22.397701263427734, 21.672237396240234, 21.426433563232422, 21.75296401977539, 21.87459945678711, 22.062480926513672, 20.808849334716797, 19.457477569580078, 20.09280014038086, 21.246517181396484, 20.653911590576172, 22.276065826416016, 23.34109115600586, 21.77468490600586, 21.21755599975586, 21.768169403076172, 21.302989959716797, 23.609699249267578, 23.531505584716797, 21.691783905029297, 20.858081817626953, 20.446842193603516, 21.84238052368164, 20.08230209350586, 20.856273651123047, 23.97641372680664, 22.659069061279297, 21.63277816772461, 20.159770965576172, 19.147235870361328, 21.192577362060547, 19.978404998779297, 19.927722930908203, 20.258235931396484, 22.381771087646484, 20.574268341064453, 20.364665985107422, 21.011211395263672, 20.850841522216797, 22.66232681274414, 22.740520477294922, 27.73440933227539, 27.049854278564453, 29.612865447998047, 27.490779876708984, 26.155696868896484, 24.07089614868164, 23.765361785888672, 23.15103530883789, 22.77780532836914, 22.253620147705078, 23.61440658569336, 22.456707000732422, 22.10519790649414, 22.038951873779297, 22.390460968017578, 23.370052337646484, 22.532367706298828, 23.231403350830078, 22.723506927490234, 22.24782943725586, 23.752330780029297, 24.484310150146484, 23.20895767211914, 22.538158416748047, 23.042797088623047, 25.503360748291016, 23.29837417602539, 22.307559967041016, 21.555309295654297, 21.917316436767578, 21.785907745361328, 21.83188247680664, 21.487613677978516, 22.208370208740234, 22.261585235595703, 22.659069061279297, 22.019763946533203, 20.779888153076172, 19.86075210571289, 20.68142318725586, 20.524311065673828, 20.01315689086914, 19.67685317993164, 20.063114166259766, 20.283214569091797, 20.226741790771484, 19.539470672607422, 18.471187591552734, 18.716266632080078, 18.225566864013672, 18.25253677368164, 17.985736846923828, 17.945735931396484, 18.31570816040039, 18.75192642211914, 18.497074127197266, 20.30312728881836, 19.532413482666016, 19.21438980102539, 20.469287872314453, 19.379283905029297, 19.332584381103516, 18.74197006225586, 18.366207122802734, 18.569835662841797, 18.56259536743164, 18.395709991455078, 19.699298858642578, 18.79391860961914, 18.664501190185547, 18.581058502197266, 19.243350982666016, 20.515987396240234, 23.611148834228516, 22.734004974365234, 21.377201080322266, 20.237964630126953, 21.260272979736328, 23.450054168701172, 21.767444610595703, 20.41788101196289, 20.756359100341797, 20.511280059814453, 20.412090301513672, 20.837810516357422]
    # target4 = [29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727, 19.100000381469727, 19.459999084472656, 19.049999237060547, 19.450000762939453]
    # date= ['20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401', '20210405', '20210406', '20210407', '20210408', '20210409', '20210412', '20210413', '20210414', '20210415', '20210416', '20210419', '20210420', '20210421', '20210422', '20210423', '20210426', '20210427', '20210428', '20210429', '20210430', '20210503', '20210504', '20210505', '20210506', '20210507', '20210510', '20210511', '20210512', '20210513', '20210514', '20210517', '20210518', '20210519', '20210520', '20210521', '20210524', '20210525', '20210526', '20210527']

    # #gt5
    # predict5 = [31.04498291015625, 31.061676025390625, 30.888839721679688, 30.389404296875, 29.574081420898438, 29.711166381835938, 29.349990844726562, 30.359603881835938, 29.05914306640625, 28.558517456054688, 28.95306396484375, 30.223709106445312, 30.118820190429688, 29.364303588867188, 27.634735107421875, 28.97332763671875, 27.527435302734375, 26.22222900390625, 27.972061157226562, 26.126861572265625, 27.262832641601562, 27.448776245117188, 26.757431030273438, 26.199569702148438, 25.104141235351562, 25.527297973632812, 25.7227783203125, 25.447433471679688, 24.992095947265625, 24.214920043945312, 23.1492919921875, 22.856063842773438, 23.716659545898438, 22.988372802734375, 22.533035278320312, 24.44378662109375, 22.850082397460938, 21.49718475341797, 21.70220947265625, 21.52936553955078, 23.915740966796875, 24.925338745117188, 25.033828735351562, 24.461654663085938, 26.17095947265625, 29.148544311523438, 31.266677856445312, 30.403701782226562, 29.762405395507812, 28.60382080078125, 28.776641845703125, 28.844589233398438, 28.40594482421875, 27.908889770507812, 29.187881469726562, 28.308197021484375, 27.950592041015625, 27.899337768554688, 28.783798217773438, 28.61810302734375, 29.730224609375, 29.385757446289062, 29.004318237304688, 29.451309204101562, 29.205764770507812, 29.944793701171875, 28.713470458984375, 28.423828125, 27.879074096679688, 27.968475341796875, 28.382095336914062, 28.079345703125, 27.6180419921875, 27.587051391601562, 26.528564453125, 25.870590209960938, 25.543991088867188, 25.769256591796875, 26.114944458007812, 26.477294921875, 28.087677001953125, 28.5048828125, 28.458389282226562, 30.218948364257812, 30.963912963867188, 31.502700805664062, 33.363372802734375, 31.565872192382812, 31.919891357421875, 31.035446166992188, 30.833999633789062, 29.922149658203125, 28.768310546875, 27.457122802734375, 28.305816650390625, 26.701400756835938, 25.17327880859375, 24.911056518554688, 25.249557495117188, 26.324737548828125, 25.48199462890625, 24.528411865234375, 23.728591918945312, 22.826263427734375, 23.096847534179688, 22.512771606445312, 22.722564697265625, 21.608047485351562, 22.302978515625, 21.203964233398438, 21.39110565185547, 20.386856079101562, 21.034706115722656, 20.869613647460938, 22.331588745117188, 21.38634490966797, 21.06926727294922, 19.517295837402344, 21.04424285888672, 21.28144073486328, 21.530563354492188, 21.66405487060547, 23.773880004882812, 21.972793579101562, 22.112258911132812, 23.529525756835938, 22.443618774414062, 21.577056884765625, 22.50799560546875, 22.676040649414062, 22.3935546875, 20.494125366210938, 21.077606201171875, 22.065765380859375, 21.366065979003906, 20.954833984375, 21.323165893554688, 23.388870239257812, 21.432823181152344, 20.73193359375, 22.38165283203125, 22.833404541015625, 23.198150634765625, 21.8511962890625, 25.484390258789062, 26.836074829101562, 26.712127685546875, 29.047210693359375, 28.902984619140625, 26.610809326171875, 25.664382934570312, 26.042236328125, 25.502273559570312, 24.777542114257812, 24.843093872070312, 24.045654296875, 23.0777587890625, 22.658187866210938, 22.36376953125, 23.890701293945312, 22.988357543945312, 22.30535888671875, 23.508071899414062, 22.637924194335938, 24.907470703125, 24.456878662109375, 23.376953125, 22.30059814453125, 24.070693969726562, 24.009902954101562, 23.3817138671875, 23.990829467773438, 22.477005004882812, 23.177886962890625, 23.365036010742188, 22.8751220703125, 22.2469482421875, 22.916854858398438, 21.754669189453125, 23.052734375, 22.237396240234375, 20.46611785888672, 21.878631591796875, 22.227874755859375, 21.126487731933594, 19.35578155517578, 20.234283447265625, 19.897544860839844, 19.598953247070312, 19.466041564941406, 19.5667724609375, 18.80926513671875, 18.731185913085938, 18.19598388671875, 18.54583740234375, 17.92242431640625, 17.523704528808594, 18.34796142578125, 18.34259796142578, 17.860443115234375, 18.83667755126953, 18.839065551757812, 18.455825805664062, 19.253280639648438, 18.303260803222656, 18.850982666015625, 19.116798400878906, 18.384918212890625, 18.833709716796875, 19.286651611328125, 19.06195831298828, 19.9774169921875, 19.466636657714844, 18.33484649658203, 17.80620574951172, 18.928466796875, 19.981582641601562, 21.313629150390625, 21.804718017578125, 20.17169952392578, 20.281967163085938, 20.72418975830078, 20.875564575195312, 20.879737854003906, 20.310562133789062, 19.002357482910156, 18.977333068847656, 19.090560913085938, 18.833099365234375]
    # target5 = [29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727, 19.100000381469727, 19.459999084472656, 19.049999237060547, 19.450000762939453, 18.290000915527344]
    # date= ['20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401', '20210405', '20210406', '20210407', '20210408', '20210409', '20210412', '20210413', '20210414', '20210415', '20210416', '20210419', '20210420', '20210421', '20210422', '20210423', '20210426', '20210427', '20210428', '20210429', '20210430', '20210503', '20210504', '20210505', '20210506', '20210507', '20210510', '20210511', '20210512', '20210513', '20210514', '20210517', '20210518', '20210519', '20210520', '20210521', '20210524', '20210525', '20210526', '20210527']

    # #gt0
    # target = [29.80, 29.25, 29.149999618530273, 30.559999465942383, 29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328, 19.600000381469727, 18.829999923706055, 18.399999618530273, 18.450000762939453, 18.049999237060547, 17.75, 17.850000381469727, 17.1200008392334, 17.110000610351562, 17.5, 18.850000381469727, 19.049999237060547, 21.149999618530273, 20.350000381469727, 20.25, 19.850000381469727, 19.600000381469727, 19.75, 20.450000762939453, 19.760000228881836, 20.700000762939453, 20.079999923706055, 19.899999618530273, 18.399999618530273, 19.59000015258789, 21.450000762939453, 25.290000915527344, 22.549999237060547, 19.100000381469727, 19.690000534057617, 22.149999618530273, 24.299999237060547, 21.989999771118164, 21.850000381469727, 20.770000457763672, 20.600000381469727, 19.709999084472656, 18.600000381469727]


    
    # plt.figure(figsize=(35,10))
    
    # def entry_point_minmax(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=5):
    #     length = len(tar)
    #     entries = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         #if pre[idx] < np.mean(pre[idx-avg:idx]) and pre[idx] < np.mean(pre[idx+1:idx+1+avg]): #and max((pre[idx-1] - pre[idx]),(pre[idx+1] - pre[idx])) > 1.5 :
    #         if ( max([ pre1[idx], pre2[idx], pre3[idx], pre4[idx], pre5[idx] ]) - tar[idx] )/tar[idx] > ratio and ( max(tar[idx-avg:idx]) - tar[idx] )/tar[idx] > ratio : #and max((pre[idx-1] - pre[idx]),(pre[idx+1] - pre[idx])) > 1.5 :
    #             entries[idx+1] = 1#tar[idx+1]
    #     return entries#[np.nan if x == 0 else x for x in entries]
    
    # def exit_point_minmax(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=5):
    #     length = len(tar)
    #     exits = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         if ( tar[idx] - min([ pre1[idx], pre2[idx], pre3[idx], pre4[idx], pre5[idx] ]) )/tar[idx] > ratio and ( tar[idx] - min( tar[idx-avg:idx] ) )/tar[idx] > ratio :
    #             exits[idx] = 1#tar[idx]
    #     return exits#[np.nan if x == 0 else x for x in exits]


    # def entry_point_mean(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=5):
    #     length = len(tar)
    #     entries = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         #if pre[idx] < np.mean(pre[idx-avg:idx]) and pre[idx] < np.mean(pre[idx+1:idx+1+avg]): #and max((pre[idx-1] - pre[idx]),(pre[idx+1] - pre[idx])) > 1.5 :
    #         if ( np.mean([ pre1[idx], pre2[idx], pre3[idx], pre4[idx], pre5[idx] ]) - tar[idx] )/tar[idx] > ratio or ( np.mean(tar[idx-avg:idx]) - tar[idx] )/tar[idx] > ratio : #and max((pre[idx-1] - pre[idx]),(pre[idx+1] - pre[idx])) > 1.5 :
    #             entries[idx+1] = 1#tar[idx+1]
    #     return entries#[np.nan if x == 0 else x for x in entries]
    
    # def exit_point_mean(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=5):
    #     length = len(tar)
    #     exits = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         if ( tar[idx] - np.mean([ pre1[idx], pre2[idx], pre3[idx], pre4[idx], pre5[idx] ]) )/tar[idx] > ratio and ( tar[idx] - np.mean( tar[idx-avg:idx] ) )/tar[idx] > ratio :
    #             exits[idx] = 1#tar[idx]
    #     return exits#[np.nan if x == 0 else x for x in exits]


    # def entry_point_3consecutive(pre1, pre2, pre3, pre4, pre5, tar, dat, avg=3):
    #     length = len(tar)
    #     entries = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         surge = tar[idx] < pre1[idx] and pre1[idx] < pre2[idx] and pre2[idx] < pre3[idx]
    #         plunge = tar[idx-3] > tar[idx-2] and tar[idx-2] > tar[idx-1] and tar[idx-1] > tar[idx]
    #         if plunge and surge:
    #             entries[idx+1] = 1
    #     return entries#[np.nan if x == 0 else x for x in entries]

    # def exit_point_3consecutive(pre1, pre2, pre3, pre4, pre5, tar, dat, avg=3):
    #     length = len(tar)
    #     exits = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         plunge = tar[idx] > pre1[idx] and pre1[idx] > pre2[idx] and pre2[idx] > pre3[idx]
    #         surge = tar[idx-3] < tar[idx-2] and tar[idx-2] < tar[idx-1] and tar[idx-1] < tar[idx]
    #         if plunge and surge:
    #             exits[idx] = 1
    #     return exits#[np.nan if x == 0 else x for x in entries]


    # def entry_point_2consecutive(pre1, pre2, pre3, pre4, pre5, tar, dat, avg=2):
    #     length = len(tar)
    #     entries = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         surge = tar[idx] < pre1[idx] and pre1[idx] < pre2[idx]
    #         plunge = tar[idx-2] > tar[idx-1] and tar[idx-1] > tar[idx]
    #         if plunge and surge:
    #             entries[idx+1] = 1
    #     return entries#[np.nan if x == 0 else x for x in entries]

    # def exit_point_2consecutive(pre1, pre2, pre3, pre4, pre5, tar, dat, avg=2):
    #     length = len(tar)
    #     exits = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         plunge = tar[idx] > pre1[idx] and pre1[idx] > pre2[idx]
    #         surge = tar[idx-2] < tar[idx-1] and tar[idx-1] < tar[idx]
    #         if plunge and surge:
    #             exits[idx] = 1
    #     return exits#[np.nan if x == 0 else x for x in entries]


    # def entry_point_1consecutive_ratio(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=1):
    #     length = len(tar)
    #     entries = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         surge = tar[idx] < pre1[idx] and (pre1[idx] - tar[idx]) / tar[idx] > ratio
    #         plunge = tar[idx-1] > tar[idx] and (tar[idx-1] - tar[idx]) / tar[idx] > ratio
    #         if plunge and surge:
    #             entries[idx+1] = 1
    #     return entries#[np.nan if x == 0 else x for x in entries]

    # def exit_point_1consecutive_ratio(pre1, pre2, pre3, pre4, pre5, tar, dat, ratio, avg=1):
    #     length = len(tar)
    #     exits = [0] * length
    #     for idx in range(avg,length-1):#-avg):
    #         plunge = tar[idx] > pre1[idx] and (tar[idx] - pre1[idx]) / tar[idx] > ratio
    #         surge = tar[idx-1] < tar[idx] and (tar[idx] - tar[idx-1]) / tar[idx] > ratio
    #         if plunge and surge:
    #             exits[idx] = 1
    #     return exits#[np.nan if x == 0 else x for x in entries]
    

    # entries_minmax = entry_point_minmax(predict1, predict2, predict3, predict4, predict5, target, date, 0.07)
    # exits_minmax = exit_point_minmax(predict1, predict2, predict3, predict4, predict5, target, date, 0.12)

    # entries_mean = entry_point_mean(predict1, predict2, predict3, predict4, predict5, target, date, 0.08)
    # exits_mean = exit_point_mean(predict1, predict2, predict3, predict4, predict5, target, date, 0.08)

    # entries_3consecutive = entry_point_3consecutive(predict1, predict2, predict3, predict4, predict5, target, date)
    # exits_3consecutive = exit_point_3consecutive(predict1, predict2, predict3, predict4, predict5, target, date)

    # entries_2consecutive = entry_point_2consecutive(predict1, predict2, predict3, predict4, predict5, target, date)
    # exits_2consecutive = exit_point_2consecutive(predict1, predict2, predict3, predict4, predict5, target, date)

    # entries_1consecutive = entry_point_1consecutive_ratio(predict1, predict2, predict3, predict4, predict5, target, date, 0.03)
    # exits_1consecutive = exit_point_1consecutive_ratio(predict1, predict2, predict3, predict4, predict5, target, date, 0.04)

    # entries_join = [(a+b+c+d+e) / 5 for a,b,c,d,e in zip(entries_minmax, entries_mean, entries_3consecutive, entries_2consecutive, entries_1consecutive)]
    # exits_join = [(a+b+c+d+e) / 5 for a,b,c,d,e in zip(exits_minmax, exits_mean, exits_3consecutive, exits_2consecutive, exits_1consecutive)]

    
    # def points_generate(entry_list, exit_list, en_ratio, ex_ratio):
    #     entries, exits = [0] * len(target), [0] * len(target)
    #     for idx, prob in enumerate(entry_list):
    #         if prob >= en_ratio:
    #             entries[idx] = target[idx]
    #     entries = [np.nan if x == 0 else x for x in entries]

    #     for idx, prob in enumerate(exit_list):
    #         if prob >= ex_ratio:
    #             exits[idx] = target[idx]
    #     exits = [np.nan if x == 0 else x for x in exits]

    #     return entries, exits
    
    # entries3, exits3 = points_generate(entries_join, exits_join, 0.6, 0.6)
    # entries4, exits4 = points_generate(entries_join, exits_join, 0.8, 0.8)

    # for idx, item in enumerate(entries4):
    #     if item > 0:
    #         print(date[idx])

    # print('\t')

    # for idx, item in enumerate(exits4):
    #     if item > 0:
    #         print(date[idx])


    #read
    # df = pd.read_pickle('to_produce_table_chart.pkl')
    # print(df)
    # #df.to_csv(file_name, sep='\t', encoding='utf-8')
    # df.to_csv('chart.csv', index=False)
    # print('csv done')

    #data
    # x_mea = [20,40,60,80,100,120,200,392]
    # gaussian = [2826.36, 629.2, 567.04, 589.44, 561.12, 544.05, 541.84, 532.50]
    # extreme_98 = [5259.71, 3045.75, 3294.04, 3287.61, 2825.86, 2645.92, 2437.27, 2383.59]
    # extreme_196 = [2850.41, 1616.33, 1305.83, 1192.84, 1167.68, 1158.55, 1127.19, 1147.27]
    # extreme_392 = [2261.22, 688.30, 673.26, 612.83, 596.90, 586.75, 568.75, 553.83]
    # general = [2789.52, 685.58, 584.64, 649.43, 607.32, 593.44, 577.40, 543.07]
    # adaptive = [4386.02, 4209.91, 5132.95, 3518.83, 3719.57, 5627.79, 1619.13, 2541.81]
    df = pd.read_csv('C:/Users/User/Desktop/csgm/src/chart_round.csv')
    columns = df.columns.to_list()
    '''
    columns :['dataset', 'seed_no', 'measurement_type', 'measurement_num', 'EXTREME_pixel_observe', 'measurement_loss', 'MSE_loss', 'noise', 'round']
    noise: ['0_0', '0_1', '0_5', '1_5', '5_0', '10_0', '20_0', '25_0', '30_0']
    '''
    mea_adp = [10,20,30,40,50]
    x_mea = [30,60,90,120,150]
    noise_ls = ['0_0', '0_1', '0_5', '1_5', '5_0', '10_0', '20_0', '25_0', '30_0']

    for noise in noise_ls:
        gaussian, extreme_98, extreme_196, general, adaptive_best, adaptive = [], [], [], [], [], []
        adaptive_idx = []
        for num_mea in x_mea:
            gaussian_loss_mean = df[ (df['measurement_type']=='gaussian') & (df['measurement_num']==num_mea) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            gaussian.append(gaussian_loss_mean)
            
            block_98_loss_mean = df[ (df['measurement_type']=='gaussian_block') & (df['EXTREME_pixel_observe']==98) & (df['measurement_num']==num_mea) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            extreme_98.append(block_98_loss_mean)
            
            block_196_loss_mean = df[ (df['measurement_type']=='gaussian_block') & (df['EXTREME_pixel_observe']==196) & (df['measurement_num']==num_mea) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            extreme_196.append(block_196_loss_mean)
            
            general_loss_mean = df[ (df['measurement_type']=='gaussian_block_general') & (df['measurement_num']==num_mea) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            general.append(general_loss_mean)

            adaptive_loss_0 = df[ (df['measurement_type']=='gaussian_block_adaptive') & (df['round']==0) & (df['measurement_num']==num_mea/3) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            adaptive_loss_1 = df[ (df['measurement_type']=='gaussian_block_adaptive') & (df['round']==1) & (df['measurement_num']==num_mea/3) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            adaptive_loss_2 = df[ (df['measurement_type']=='gaussian_block_adaptive') & (df['round']==2) & (df['measurement_num']==num_mea/3) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            adaptive_loss_3 = df[ (df['measurement_type']=='gaussian_block_adaptive') & (df['round']==3) & (df['measurement_num']==num_mea/3) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            adaptive_loss_4 = df[ (df['measurement_type']=='gaussian_block_adaptive') & (df['round']==4) & (df['measurement_num']==num_mea/3) & (df['noise']==noise) ]['MSE_loss'].mean(axis=0)
            #plan 1 best
            adaptive_list = [adaptive_loss_0, adaptive_loss_1, adaptive_loss_2, adaptive_loss_3, adaptive_loss_4]
            adaptive_loss_mean = min(adaptive_list)
            min_index = adaptive_list.index(adaptive_loss_mean)
            if int(num_mea/3*(min_index+1)) <= num_mea: 
                adaptive_best.append(adaptive_loss_mean)
                adaptive_idx.append(int(num_mea/3*(min_index+1)))
            else:
                adaptive_best.append(adaptive_loss_2)
                adaptive_idx.append(num_mea)
            #plan 2 plain
            adaptive.append(adaptive_loss_2)
        
        plt.figure(figsize=(25,15))   
        plot_title = 'noise_{}_mean_MSE_line_chart'.format(noise)
        plt.plot(x_mea, gaussian, color='forestgreen', marker='.', label = 'full_gaussian', alpha=1, markersize=8)#####
        plt.plot(x_mea, extreme_98, color='deepskyblue', marker='.', label = 'extreme_top_98', alpha=1, markersize=8)#####
        plt.plot(x_mea, extreme_196, color='royalblue', marker='.', label = 'extreme_top_196', alpha=1, markersize=8)#####  
        plt.plot(x_mea, general, color='darkviolet', marker='.', label = 'general', alpha=1, markersize=8)#####
        plt.plot(x_mea, adaptive, color='red', marker='.', label = 'adaptive', alpha=1, markersize=8)##### 
        plt.plot(x_mea, adaptive_best, color='darkorange', marker='.', label = 'adaptive_with_dynamic_#mea', alpha=1, markersize=8)##### 
        plt.xlabel('#measurements', fontsize=16)
        plt.xticks(np.array(x_mea), fontsize=16)
        plt.ylabel('MSE', fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(plot_title, fontsize=16)
        plt.legend(prop={'size': 16})
        for a,b,c in zip(x_mea, adaptive_best, adaptive_idx): 
            plt.text(a, b, '#mea='+str(c), fontsize=16)
        path_dir = './'
        img_name = plot_title+'.jpg'
        final = os.path.join(path_dir, img_name)
        plt.savefig(final)







    #plot 
    # plt.figure(figsize=(25,15))   
    # plot_title =  'jul16_mean_MSE_line_chart (pre_pixel*255*255)'
    
    # plt.plot(x_mea, gaussian, 'y--', label = 'full_gaussian', alpha=1)#####
    # plt.plot(x_mea, extreme_98, 'm--', label = 'extreme_top_98', alpha=1)#####
    # plt.plot(x_mea, extreme_196, 'c--', label = 'extreme_top_196', alpha=1)#####
    # plt.plot(x_mea, extreme_392, 'r--', label = 'extreme_top_392', alpha=1)#####    
    # plt.plot(x_mea, general, 'g--', label = 'general', alpha=1)#####
    # plt.plot(x_mea, adaptive, 'b--', label = 'adaptive', alpha=1)##### 

    # plt.xlabel('#measurements', fontsize=16)
    # plt.xticks(np.array(x_mea), fontsize=16)
    # plt.ylabel('MSE', fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.title(plot_title, fontsize=16)
    # plt.legend(prop={'size': 16})
    # path_dir = './'
    # img_name = 'jul16_mean_MSE_line_chart.jpg'
    # final = os.path.join(path_dir, img_name)
    # plt.savefig(final)



    



