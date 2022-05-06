"""main script"""

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils2 as utils
import yaml
from skimage.io import imread, imsave
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def save_image(image, path):
    """Save an image as a png file"""
    x_png = np.uint8(np.clip(image*256,0,255))
    x_png = x_png.transpose(1,2,0)
    if x_png.shape[-1] == 1:
        x_png = x_png[:,:,0]
    x_png = Image.fromarray(x_png).save(path)

def normalize(a):
    return((a-np.min(a))/(np.max(a)-np.min(a)))

def matrix_builder(hparams, var_m):
    signal_shape = var_m.shape
    std = np.zeros(signal_shape)
    
    adhoc = np.load('../../uncertainty2/celeb_var_top_1_2_adhoc.npy')[64:192,64:192] #128,128
    var_m_norm = normalize(var_m*adhoc)
    var_m_norm = (var_m_norm+0.15)/(1+0.15) #aug14
    unit = (0.5/hparams.num_measurements) * np.prod(signal_shape) / np.sum(var_m_norm*adhoc)
    
    for i in range(signal_shape[0]):
        for j in range(signal_shape[1]):
            if adhoc[i,j] != 0:
                std[i,j] = np.sqrt(var_m_norm[i,j] * unit)
            else:
                std[i,j] = 1 / np.sqrt(hparams.num_measurements) #aug14
    std = np.reshape(std, (-1))
    A = np.zeros((3*np.prod(signal_shape), hparams.num_measurements))
    for i in range(std.shape[0]):
        A[3*i:3*i+3,:] = std[i] * np.random.randn(hparams.num_measurements)      
    return A, var_m_norm*adhoc

def matrix_builder_data_driven(hparams, top_idx, bot_idx, lambd, round, matrix=None):
    if round == 0:
        std = np.ones((hparams.image_size, hparams.image_size)) #(128,128)
    else:
        std = matrix
    top_pos = [(idx//128, idx%128) for idx in top_idx]
    bot_pos = [(idx//128, idx%128) for idx in bot_idx]
    for i in range(std.shape[0]):
        for j in range(std.shape[1]):
            if (i,j) in top_pos:
                std[i,j] = std[i,j] * math.exp(-lambd)
            elif (i,j) in bot_pos:
                std[i,j] = std[i,j] * math.exp(lambd)
    std = std / np.sum(std) * 128*128*(1/hparams.num_measurements)
    std_reshape = np.reshape(std, (-1))
    A = np.zeros((3*np.prod(std.shape), hparams.num_measurements))
    for i in range(std_reshape.shape[0]):
        A[3*i:3*i+3,:] = np.sqrt(std_reshape[i]) * np.random.randn(hparams.num_measurements)      
    return std, A

def color_map(img, path):
    plt.figure(figsize=(8,5))
    plt.imshow(img, cmap='YlOrRd')
    plt.colorbar()
    plt.savefig(path)


def main(hparams):

    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict_ = utils.model_input(hparams) 
    xs_dict = {}
    
    img = xs_dict_[hparams.img_no].reshape((3, 256, 256))[:,64:192,64:192] # RESECALE sep27
    xs_dict[0] = img.reshape(-1)
    
    noise_info = '_'.join(str(hparams.noise_std).split('.'))
    save_dir = os.path.join('./', str(hparams.img_no))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_image(img, os.path.join(save_dir,'{}_img_{}_original.png'.format(hparams.dataset, hparams.img_no)))
    # save_image(img_noise, os.path.join(save_dir,'{}_img_{}_with_noise_{}.png'.format(hparams.dataset, hparams.img_no, noise_info)))

    # get estimator
    estimator = utils.get_estimator(hparams, hparams.model_type)

    # set up folders, etc for checkpointing 
    utils.setup_checkpointing(hparams) #'./estimated/{0}/{1}/{2}/{3}/{4}/{5}/annealed_{6}/'...

    # get saved results
    measurement_losses, l2_losses, z_hats, likelihoods = utils.load_checkpoints(hparams) #{'dcgan':{}}, {'dcgan':{}} ...

    x_batch_dict = {}
    x_hats_dict = {}

    A = utils.get_A(hparams)



    for key, x in xs_dict.items():
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before . If yes, then skip this image.
            save_path = utils.get_save_path(hparams, key)
            is_saved = os.path.isfile(save_path) 
            if is_saved:
                continue

        x_batch_dict[key] = x
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #(1,256*256*3)
        x_batch = np.concatenate(x_batch_list) #(1,256*256*3)

        # Construct noise and measurements
        noise_batch = utils.get_noise(hparams) #(1,#mea)
        y_batch = utils.get_measurements(x_batch, A, noise_batch, hparams) #y_batch = np.matmul(x_batch, A) + noise_batch

        # Construct estimates 
        if hparams.measurement_type != 'gaussian_block_adaptive': 
            # x_hat_batch, z_hat_batch, likelihood_batch = estimator(A, y_batch, hparams)
            x_hat_batch, likelihood_batch = estimator(A, y_batch, hparams)
        else:
            #x_hat_batch, x_hat_batch_var, z_hat_batch, likelihood_batch = estimator(A, y_batch, hparams)
            x_hat_batch, x_hat_batch_var, likelihood_batch = estimator(A, y_batch, hparams)

        for i, key in enumerate(x_batch_dict.keys()):
            x = xs_dict[key] #original
            y_train = y_batch[i] #y_mea
            x_hat = x_hat_batch[i] #h_recover

            # Save the estimate
            x_hats_dict[key] = x_hat

            # Compute and store measurement and l2 loss
            measurement_losses[key] = utils.get_measurement_loss(x_hat, A, y_train, hparams)
            l2_losses[key] = utils.get_l2_loss(x_hat, x)
            ###
            l2_pixl_loss_matrix = utils.get_pixl_l2_loss(x_hat, x) #(128,128)
            center_rloss, corner_rloss = utils.get_l2_loss_region(x_hat, x) #(n,) (n,)
            ###
            # z_hats[key] = z_hat_batch[i]
            likelihoods[key] = likelihood_batch[i]

            # log 
            if hparams.measurement_type == 'gaussian_block':
                num_observe_info = hparams.num_observe
            else:
                num_observe_info = 'NA'
            pickle_file_path = hparams.pickle_file_path
            
            if hparams.adaptive_round_count == -1:
                round_info = 'NA' 
            else:
                round_info = hparams.adaptive_round_count
            
            if not os.path.exists(pickle_file_path):
                d = {'dataset':[hparams.dataset], 'img_no':[hparams.img_no], 'measurement_type':[hparams.measurement_type], 'measurement_num':[hparams.num_measurements],
                    'EXTREME_pixel_observe':[num_observe_info], 'block_i':[int(hparams.i)], 'block_j':[int(hparams.j)], 'MSE_loss':[l2_losses[0]], 'Center_loss':[center_rloss], 'Corner_loss':[corner_rloss], 'noise':[noise_info], 'round':[round_info]}
                df = pd.DataFrame(data=d)
                df.to_pickle(pickle_file_path)
            else:
                d = {'dataset':hparams.dataset, 'img_no':hparams.img_no, 'measurement_type':hparams.measurement_type, 'measurement_num':hparams.num_measurements,
                    'EXTREME_pixel_observe':num_observe_info, 'block_i':int(hparams.i), 'block_j':int(hparams.j), 'MSE_loss':l2_losses[0], 'Center_loss':center_rloss, 'Corner_loss':corner_rloss, 'noise':noise_info, 'round':round_info}
                df = pd.read_pickle(pickle_file_path)
                df = df.append(d, ignore_index=True)
                df.to_pickle(pickle_file_path)
            

            # output saveing 
            img = x_hat.reshape(hparams.image_shape) 
            
            if hparams.measurement_type == 'gaussian_block':
                save_dir = './{}/extreme/top_{}/noise_{}/measurement_{}'.format(hparams.img_no, hparams.num_observe, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_image(img, os.path.join(save_dir, 'output_EXTREME_observe_top_{}_measurement_{}_noise_{}.jpg'.format(hparams.num_observe, hparams.num_measurements, noise_info)))
            
            elif hparams.measurement_type == 'gaussian_block_general':
                save_dir = './{}/general/noise_{}/measurement_{}'.format(hparams.img_no, noise_info, hparams.num_measurements)
                npy_dir = './{}/general/noise_{}/measurement_{}/npy'.format(hparams.img_no, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)
                save_image(img, os.path.join(save_dir, 'output_GENERAL_measurement_{}_noise_{}.jpg'.format(hparams.num_measurements, noise_info)))
                np.save( os.path.join(npy_dir, 'l2_pixl_loss_matrix_GENERAL_measurement_{}_noise_{}.npy'.format(hparams.num_measurements, noise_info)), l2_pixl_loss_matrix ) 
            
            elif hparams.measurement_type == 'gaussian':
                save_dir = './{}/full_gaussian/noise_{}/measurement_{}'.format(hparams.img_no, noise_info, hparams.num_measurements)
                npy_dir = './{}/full_gaussian/noise_{}/measurement_{}/npy'.format(hparams.img_no, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)
                save_image(img, os.path.join(save_dir, 'output_FULL_GAUSSIAN_measurement_{}_noise_{}.jpg'.format(hparams.num_measurements, noise_info)))
                np.save( os.path.join(npy_dir, 'l2_pixl_loss_matrix_FULL_GAUSSIAN_measurement_{}_noise_{}.npy'.format(hparams.num_measurements, noise_info)), l2_pixl_loss_matrix ) 

            elif hparams.measurement_type == 'gaussian_block_adaptive': 
                save_dir = './{}/adaptive/noise_{}/measurement_{}'.format(hparams.img_no, noise_info, hparams.num_measurements)
                npy_dir = './{}/adaptive/noise_{}/measurement_{}/npy'.format(hparams.img_no, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)
                
                x_hat_batch_var = np.mean(x_hat_batch_var, axis=-1) #(128,128)
                A_var_m, var_m = matrix_builder(hparams, x_hat_batch_var)
                
                np.save(os.path.join(npy_dir, 'variance_ADAPTIVE_measurement_{}_noise_{}_ROUND_{}.npy'.format(hparams.num_measurements, noise_info, hparams.adaptive_round_count)), A_var_m) 
                np.save(os.path.join(npy_dir, 'variance_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), var_m) 
                color_map(var_m, os.path.join(save_dir, 'variance_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))
                save_image(img, os.path.join(save_dir, 'reconstruction_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))

            elif hparams.measurement_type == 'gaussian_data_driven': 
                save_dir = './{}/data_driven/noise_{}/measurement_{}'.format(hparams.img_no, noise_info, hparams.num_measurements)
                npy_dir = './{}/data_driven/noise_{}/measurement_{}/npy'.format(hparams.img_no, noise_info, hparams.num_measurements)
                load_npy_dir = './{}/data_driven/noise_{}/measurement_{}/npy'.format(hparams.load_img_no, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)
                n_top = int(128*128*hparams.top_percent)
                top_idx = l2_pixl_loss_matrix.ravel().argsort()[:n_top]
                bot_idx = l2_pixl_loss_matrix.ravel().argsort()[-n_top:]
                
                lamb = hparams.lamb * (hparams.lamb_decay_rate ** hparams.adaptive_round_count)
                if hparams.adaptive_round_count == 0:
                    # lamb = hparams.lamb
                    std_out, A = matrix_builder_data_driven(hparams, top_idx, bot_idx, lamb, hparams.adaptive_round_count, matrix=None)
                else:
                    # lamb = hparams.lamb * hparams.lamb_decay_rate
                    std_in = np.load( os.path.join(load_npy_dir, 'energy_map_from_ROUND_{}.npy'.format(hparams.adaptive_round_count-1)) )
                    std_out, A = matrix_builder_data_driven(hparams, top_idx, bot_idx, lamb, hparams.adaptive_round_count, matrix=std_in)
                save_image(img, os.path.join(save_dir, 'Reconstruct_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))
                
                top_mask = np.zeros((128,128))
                bot_mask = np.zeros((128,128))
                top_pos = [(idx//128, idx%128) for idx in top_idx]
                bot_pos = [(idx//128, idx%128) for idx in bot_idx]
                for basis in top_pos:
                    top_mask[basis[0],basis[1]] = 1
                for basis in bot_pos:
                    bot_mask[basis[0],basis[1]] = 1
                top_mask[top_mask==0] = np.nan
                bot_mask[bot_mask==0] = np.nan
                # plt.figure()
                # plt.imshow(img, 'gray', interpolation='none')
                # plt.imshow(top_mask, 'blue', interpolation='none', alpha=0.7)
                # plt.imshow(bot_mask, 'red', interpolation='none', alpha=0.7)
                # plt.savefig(os.path.join(save_dir, 'Reconstruct_energy_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))
                np.save( os.path.join(npy_dir, 'least_error_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), top_mask) 
                np.save( os.path.join(npy_dir, 'most_error_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), bot_mask) 
                color_map(top_mask, os.path.join(save_dir, 'least_error_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))
                color_map(bot_mask, os.path.join(save_dir, 'most_error_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))

                np.save( os.path.join(npy_dir, 'Measurement_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), A) 
                np.save( os.path.join(npy_dir, 'energy_map_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), std_out) 
                np.save( os.path.join(npy_dir, 'l2_pixl_loss_matrix_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), l2_pixl_loss_matrix) 
    

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--checkpoint-path', type=str, default='./checkpoints/glow/graph_unoptimized.pb', help='Path to pretrained model')
    PARSER.add_argument('--net', type=str, default='glow', help='Name of model. options = [glow, stylegan2, ncsnv2, dd]')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--image-size', type=int, default=128, help='size of image') # RESECALE sep27
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--num-input-images', type=int, default=30, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')
    PARSER.add_argument('--cache-dir', type=str, default='cache', help='cache directory for model weights')
    PARSER.add_argument('--ncsnv2-configs-file', type=str, default='./ncsnv2/configs/ffhq.yml', help='location of ncsnv2 config file')


    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=1, help='expected norm of noise')
    PARSER.add_argument('--measurement-noise-type', type=str, default='gaussian', help='type of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--downsample', type=int, default=None, help='downsampling factor')

    # Model
    PARSER.add_argument('--model-type', type=str, default=None, required=True, help='model used for estimation. options=[map, langevin, pulse, dd]')
    PARSER.add_argument('--mloss-weight', type=float, default=-1, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior-weight', type=float, default=-1, help='weight on z prior')
    PARSER.add_argument('--zprior-sdev', type=float, default=1.0, help='standard deviation for target distributon of  z')
    PARSER.add_argument('--zprior-init-sdev', type=float, default=1.0, help='standard deviation to initialize z')
    PARSER.add_argument('--T', type=float, default=-200, help='number of iterations for each level of noise in Langevin annealing')
    PARSER.add_argument('--L', type=float, default=-10, help='number of noise levels for annealing Langevin')
    PARSER.add_argument('--sigma-init', type=float, default=64, help='initial noise level for annealing langevin')
    PARSER.add_argument('--sigma-final', type=float, default=16, help='final noise level for annealing Langevin')
    PARSER.add_argument('--error-threshold', type=float, default=0., help='threshold for measurement error before restart')
    PARSER.add_argument('--num-noise-variables', type=int, default=5, help='STYLEGAN2 : number of noise variables in  to optimize')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='sgd', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0., help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=1000, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=1, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')

    #PULSE arguments
    PARSER.add_argument('--seed', type=int, help='manual seed to use')
    PARSER.add_argument('--loss-str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
    PARSER.add_argument('--pulse-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
    PARSER.add_argument('--noise-type', type=str, default='trainable', help='zero, fixed, or trainable')
    PARSER.add_argument('--tile-latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
    PARSER.add_argument('--lr-schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')

    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--debug', action='store_true', help='debug mode does not save images or stats')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=0,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                       )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')

    PARSER.add_argument('--cuda', dest='cuda', action='store_true')
    PARSER.add_argument('--no-cuda', dest='cuda', action='store_false')
    PARSER.set_defaults(cuda=True)

    PARSER.add_argument('--project', dest='project', action='store_true')
    PARSER.add_argument('--no-project', dest='project', action='store_false')
    PARSER.set_defaults(project=True)


    ## ADDED by Young ##
    #gaussian matrix
    PARSER.add_argument('--top-ratio-inverse', type=int, default=4)
    PARSER.add_argument('--num-observe', type=int, default=16384)
    # PARSER.add_argument('--mnist-propotional-var-mask', type=str, default='./uncertainty2/mnist_propotional_var_mask.npy')

    #adaptive
    PARSER.add_argument('--adaptive-round-count', type=int, default=-1)
    PARSER.add_argument('--sample-frequency', type=int, default=50)
    PARSER.add_argument('--img-no', type=int, default=3)

    #annealed
    PARSER.add_argument('--annealed', dest='annealed', action='store_true')
    PARSER.add_argument('--no-annealed', dest='annealed', action='store_false')
    PARSER.set_defaults(annealed=True)

    #log
    PARSER.add_argument('--pickle-file-path', type=str, default='./result_celeba.pkl')
    PARSER.add_argument('--log-file-path', type=str, default='./log_out_celeba.txt')

    # data driven
    PARSER.add_argument('--i', type=int, default=99, help='x-axis idx of block') 
    PARSER.add_argument('--j', type=int, default=99, help='x-axis idx of block') 
    PARSER.add_argument('--lamb', type=float, default=0.1, help='initial value of lambda') 
    PARSER.add_argument('--lamb-decay-rate', type=float, default=0.95, help='lambda decay rate per round') 
    PARSER.add_argument('--top-percent', type=float, default=0.33, help='choose top performing pixls') 
    PARSER.add_argument('--block-size', type=int, default=16, help='choose top performing pixls') 
    PARSER.add_argument('--load-img-no', type=int, default=256)


    HPARAMS = PARSER.parse_args()
    HPARAMS.input_path = f'./test_images/{HPARAMS.dataset}'
    if HPARAMS.cuda:
        HPARAMS.device='cuda:0' #for torch only
    else:
        HPARAMS.device = 'cpu:0'


    if HPARAMS.net == 'ncsnv2':
        with open(HPARAMS.ncsnv2_configs_file, 'r') as f:
            HPARAMS.ncsnv2_configs = yaml.load(f)
        HPARAMS.ncsnv2_configs['sampling']['step_lr'] = HPARAMS.learning_rate
        HPARAMS.ncsnv2_configs['sampling']['n_steps_each'] = int(HPARAMS.T)
        HPARAMS.ncsnv2_configs['model']['sigma_begin'] = int(HPARAMS.sigma_init)
        HPARAMS.ncsnv2_configs['model']['sigma_end'] = HPARAMS.sigma_final

    HPARAMS.image_shape = (3, HPARAMS.image_size, HPARAMS.image_size)
    HPARAMS.n_input = np.prod(HPARAMS.image_shape)

    if HPARAMS.measurement_type == 'circulant':
        HPARAMS.train_indices = np.random.randint(0, HPARAMS.n_input, HPARAMS.num_measurements )
        HPARAMS.sign_pattern = np.float32((np.random.rand(1,HPARAMS.n_input) < 0.5)*2 - 1.)
    elif HPARAMS.measurement_type == 'superres':
        HPARAMS.y_shape = (HPARAMS.batch_size, HPARAMS.image_shape[0], HPARAMS.image_size//HPARAMS.downsample,HPARAMS.image_size//HPARAMS.downsample)
        HPARAMS.num_measurements = np.prod(HPARAMS.y_shape[1:])
    elif HPARAMS.measurement_type == 'project':
        HPARAMS.y_shape = (HPARAMS.batch_size, HPARAMS.image_shape[0], HPARAMS.image_size, HPARAMS.image_size)
        HPARAMS.num_measurements = np.prod(HPARAMS.y_shape[1:])


    if HPARAMS.mloss_weight < 0:
        HPARAMS.mloss_weight = None
    if HPARAMS.zprior_weight < 0:
        HPARAMS.zprior_weight = None
    if HPARAMS.annealed:
        if HPARAMS.T < 0:
            HPARAMS.T = 200
        if HPARAMS.L < 0:
            HPARAMS.L = 10
        if HPARAMS.sigma_final < 0:
            HPARAMS.sigma_final = HPARAMS.noise_std
        if HPARAMS.sigma_init < 0:
            HPARAMS.sigma_init =  100 * HPARAMS.sigma_final
        HPARAMS.max_update_iter = int(HPARAMS.T * HPARAMS.L)

    
    #ADDED by Young 29Jul
    HPARAMS.celeb_block_mask = './uncertainty2/celeb_var_top_{}_mask.npy'.format(HPARAMS.num_observe)
    HPARAMS.celeb_propotional_A = './uncertainty2/celeb_propotional_A_mea_{}.npy'.format(HPARAMS.num_measurements)
    HPARAMS.celeb_gaussian_A = './uncertainty2/celeb_gaussian_A_mea_{}.npy'.format(HPARAMS.num_measurements)
    HPARAMS.celeb_extreme_A = './uncertainty2/celeb_extreme_A_top_1_{}_mea_{}.npy'.format(HPARAMS.top_ratio_inverse, HPARAMS.num_measurements)
    HPARAMS.celeb_block_data_driven_A = './uncertainty2/celeb_block_{}_data_driven_A_i_{}_j_{}_mea_{}.npy'.format(
        HPARAMS.block_size, HPARAMS.i, HPARAMS.j, HPARAMS.num_measurements)

    main(HPARAMS)


