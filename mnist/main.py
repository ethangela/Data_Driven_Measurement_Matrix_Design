"""main script"""

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
import pandas as pd 
import math


def matrix_builder(hparams, var_m):
    var_m = (var_m+0.1)/(1+0.1)
    signal_shape = var_m.shape
    std = np.zeros(signal_shape)
    unit = 1**2 * np.prod(signal_shape) / np.sum(var_m)
    for i in range(signal_shape[0]):
        for j in range(signal_shape[1]):
            std[i,j] = np.sqrt(var_m[i,j] * unit)
            # var_adjust = (var_m[i,j]+0.25) / (1+0.25) #aug14
            # std[i,j] = np.sqrt(var_adjust * unit)
    std = np.reshape(std, (-1))
    A = np.zeros((np.prod(signal_shape), hparams.num_measurements))
    for i in range(std.shape[0]):
        A[i,:] = std[i] * np.random.randn(hparams.num_measurements)      
    return A

def matrix_builder_data_driven(hparams, top_idx, bot_idx, lambd, round, matrix=None):
    if round == 0:
        std = np.ones(hparams.image_shape[:-1]) #(28,28)
    else:
        std = matrix
    top_pos = [(idx//28, idx%28) for idx in top_idx]
    bot_pos = [(idx//28, idx%28) for idx in bot_idx]
    for i in range(std.shape[0]):
        for j in range(std.shape[1]):
            if (i,j) in top_pos:
                std[i,j] = std[i,j] * math.exp(-lambd)
            elif (i,j) in bot_pos:
                std[i,j] = std[i,j] * math.exp(lambd)
    std = std / np.sum(std) * 784.
    std_reshape = np.reshape(std, (-1))
    A = np.zeros((np.prod(std.shape), hparams.num_measurements))
    for i in range(std_reshape.shape[0]):
        A[i,:] = np.sqrt(std_reshape[i]) * np.random.randn(hparams.num_measurements)      
    return std, A

def color_map(img, path):
    plt.figure(figsize=(8,5))
    plt.imshow(img, cmap='YlOrRd')
    plt.colorbar()
    plt.savefig(path)

def main(hparams):
    ##-- compressed sensing --##
    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape) #64*64*3
    utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams) #{0: img}
    img = xs_dict[0].reshape(hparams.image_shape)
    # print('!!!!!!!!!! img mean {}'.format(np.mean(img)))

    noise_info = '_'.join(str(hparams.noise_std).split('.'))
    img_noise = hparams.noise_std * np.random.randn(hparams.image_shape[0], hparams.image_shape[1], hparams.image_shape[2])
    # print('!!!!!!!!!! noise mean {}'.format(np.mean(img_noise)))
    img_noise += img
    
    if hparams.dataset == 'celebA':
        img = (img+1.)/2.
        img_noise = (img_noise+1.)/2.
    img = img * 255.
    img_noise = img_noise * 255.

    if hparams.mini_batch != 1:
        save_dir = os.path.join('../src', str(hparams.seed_no))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imsave(os.path.join(save_dir,'{}_seed_{}_original.jpg'.format(hparams.dataset, hparams.seed_no)), img.astype(np.uint8))
        #imsave(os.path.join(save_dir,'{}_seed_{}_input_with_noise_{}.jpg'.format(hparams.dataset, hparams.seed_no, noise_info)), img_noise.astype(np.uint8))

    def color_map(img, path): #jul14!!!!
        plt.figure(figsize=(8,5))
        plt.imshow(img, cmap='YlOrRd')
        plt.colorbar()
        plt.savefig(path)
    
    print('input image(s) saved')
        

    # MLE 
    estimators = utils.get_estimators(hparams) #{dcgan_estmator} 
    utils.setup_checkpointing(hparams) #./estimate ...
    measurement_losses, l2_losses = utils.load_checkpoints(hparams) #{'dcgan':{}}, {'dcgan':{}} 

    x_hats_dict = {model_type : {} for model_type in hparams.model_types} #{'dcgan':{}}
    x_batch_dict = {}
    
    for key, x in xs_dict.items(): #{0: img}
        
        x_batch_dict[key] = x #{0: img}
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #[img]
        x_batch = np.concatenate(x_batch_list) #img, shape (1, 12288)

        # Construct measurements
        A = utils.get_A(hparams) #(n_input, n_measurement)
      
        if hparams.dataset == 'mnist' and hparams.measurement_type == 'gaussian_block': 
            mask_gaussian_block = np.load(hparams.mnist_block_mask) 
            x_batch = mask_gaussian_block * x_batch #(1,784)*(1,784)=(1,784)
            print('gaussian block matrix set')
        
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)

        if hparams.measurement_type == 'project':
            y_batch = x_batch + noise_batch
        else:
            y_batch = np.matmul(x_batch, A) + noise_batch
        print('Y shape check {}'.format(y_batch.shape))

        # Construct estimates using each estimator
        for model_type in hparams.model_types: #dcgan
            estimator = estimators[model_type] #dcgan
            
            if model_type != 'vae_langevin':
                x_hat_batch = estimator(A, y_batch, hparams)
            else:
                x_hat_batch, x_hat_batch_var, restart_idx = estimator(A, y_batch, hparams)

            for i, key in enumerate(x_batch_dict.keys()):
                x = xs_dict[key]
                y = y_batch[i]
                x_hat = x_hat_batch[i]

                # Save the estimate
                x_hats_dict[model_type][key] = x_hat

                # Compute and store measurement and l2 loss
                def pixel_recover(img):
                    if hparams.dataset == 'celebA':
                        img = (img+1.)/2.
                    img = img * 255.
                    return img
                measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y)
                l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x) 
                
                l2_pixl_loss_matrix = utils.get_pixl_l2_loss(x_hat, x) #(784,)

                if hparams.measurement_type == 'gaussian_block':
                    num_observe_info = hparams.num_observe
                else:
                    num_observe_info = 'NA'
                pickle_file_path = hparams.pickle_file_path
                
                if hparams.adaptive_round_count == -1:
                    round_info = 'NA' 
                else:
                    round_info = hparams.adaptive_round_count

                if hparams.mini_batch == 1:
                    mini_batch_info = 'True'
                else:
                    mini_batch_info = 'NA'

                if not os.path.exists(pickle_file_path):
                    d = {'dataset':[hparams.dataset], 'seed_no':[hparams.seed_no], 'measurement_type':[hparams.measurement_type], 'mini_batch':[mini_batch_info], 'measurement_num':[hparams.num_measurements], 
                        'EXTREME_pixel_observe':[num_observe_info], 'MSE_loss':[l2_losses[model_type][0]], 'noise':[noise_info], 'round':[round_info]}
                    df = pd.DataFrame(data=d)
                    df.to_pickle(pickle_file_path)
                else:
                    d = {'dataset':hparams.dataset, 'seed_no':hparams.seed_no, 'measurement_type':hparams.measurement_type, 'mini_batch':mini_batch_info, 'measurement_num':hparams.num_measurements, 
                        'EXTREME_pixel_observe':num_observe_info, 'MSE_loss':l2_losses[model_type][0], 'noise':noise_info, 'round':round_info}
                    df = pd.read_pickle(pickle_file_path)
                    df = df.append(d, ignore_index=True)
                    df.to_pickle(pickle_file_path)

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))

        x_batch_dict = {}


    #log output
    for model_type in hparams.model_types:
        print(x_hats_dict[model_type][0].shape) 
        img = x_hats_dict[model_type][0].reshape(hparams.image_shape) 
        if hparams.dataset == 'celebA':
            img = (img+1.)/2.
        img = img * 255.
        
        if hparams.measurement_type == 'gaussian_block':
            save_dir = './{}/extreme/top_{}/noise_{}/measurement_{}'.format(hparams.seed_no, hparams.num_observe, noise_info, hparams.num_measurements)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imsave( os.path.join(save_dir, 'output_EXTREME_observe_top_{}_measurement_{}_noise_{}.jpg'.format(
                    hparams.num_observe, hparams.num_measurements, noise_info)), img.astype(np.uint8))
        
        elif hparams.measurement_type == 'gaussian_block_general':
            save_dir = './{}/general/noise_{}/measurement_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imsave( os.path.join(save_dir, 'output_GENERAL_measurement_{}_noise_{}.jpg'.format(hparams.num_measurements, noise_info)), img.astype(np.uint8))
            np.save( os.path.join(save_dir, 'l2_pixl_loss_matrix_GENERAL_measurement_{}_noise_{}.npy'.format(hparams.num_measurements, noise_info)), l2_pixl_loss_matrix)
        
        elif hparams.measurement_type == 'gaussian':
            save_dir = './{}/full_gaussian/noise_{}/measurement_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            imsave( os.path.join(save_dir, 'output_FULL_GAUSSIAN_measurement_{}_noise_{}.jpg'.format(hparams.num_measurements, noise_info)), img.astype(np.uint8))
            np.save( os.path.join(save_dir, 'l2_pixl_loss_matrix_FULL_GAUSSIAN_measurement_{}_noise_{}.npy'.format(hparams.num_measurements, noise_info)), l2_pixl_loss_matrix)

        elif hparams.measurement_type == 'gaussian_block_adaptive': 
            save_dir = './{}/adaptive/noise_{}/measurement_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements)
            npy_dir = './{}/adaptive/noise_{}/measurement_{}/npy'.format(hparams.seed_no, noise_info, hparams.num_measurements)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)
            def normalize(a):
                return((a-np.min(a))/(np.max(a)-np.min(a)))
            x_hat_batch_var = x_hat_batch_var.reshape(hparams.image_shape) #(28,28,1)
            var = normalize(x_hat_batch_var)
            x_hat_batch_var = np.mean(x_hat_batch_var, axis=-1) #(28,28)
            var_m = normalize(x_hat_batch_var)
            A_var_m = matrix_builder(hparams, var_m)
            np.save( os.path.join(npy_dir, 'Measurement_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), A_var_m) 
            color_map( (var_m+0.1)/1.1, os.path.join(save_dir, 'Variance_map_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)) )
            np.save( os.path.join(npy_dir, 'Variance_map_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), (var_m+0.1)/1.1) 
            imsave( os.path.join(save_dir, 'Reconstruct_from_ROUND_{}_by_Restart_{}.jpg'.format(hparams.adaptive_round_count, restart_idx)), img.astype(np.uint8))

        elif hparams.measurement_type == 'gaussian_data_driven': 
            
            n_top = int(784*hparams.top_percent)
            
            #mini_batch
            if hparams.mini_batch == 1:
                if hparams.mini_batch_train == 1:
                    npy_dir = './data_driven_mini_batch/noise_{}/measurement_{}/npy/round{}'.format(noise_info, hparams.num_measurements, hparams.adaptive_round_count)
                    if not os.path.exists(npy_dir):
                        os.makedirs(npy_dir)

                    if hparams.last_mini_batch != 1: 
                        np.save( os.path.join(npy_dir, 'l2_pixl_loss_matrix_from_seed_{}.npy'.format(hparams.seed_no)), l2_pixl_loss_matrix)
                    else:
                        for i in range(hparams.mini_batch_start_seed, hparams.mini_batch_end_seed):
                            l2_path = os.path.join(npy_dir, 'l2_pixl_loss_matrix_from_seed_{}.npy'.format(i))
                            l2_pixl_loss_matrix += np.load(l2_path)
                        l2_pixl_loss_matrix = l2_pixl_loss_matrix/hparams.num_mini_batch #to check if clipped??? ###sep6
                        np.save( os.path.join(npy_dir, 'l2_pixl_loss_matrix_mean.npy'), l2_pixl_loss_matrix ) 
                else:
                    save_dir = './{}/data_driven_mini_batch/noise_{}/measurement_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    imsave( os.path.join(save_dir, 'Reconstruct_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)), img.astype(np.uint8))
                    np.save( os.path.join(save_dir, 'l2_pixl_loss_matrix_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), l2_pixl_loss_matrix)


            else:
                save_dir = './{}/data_driven/noise_{}/measurement_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements)
                npy_dir = './{}/data_driven/noise_{}/measurement_{}/npy'.format(hparams.seed_no, noise_info, hparams.num_measurements)
                load_npy_dir = './{}/data_driven/noise_{}/measurement_{}/npy'.format(hparams.load_seed_no, noise_info, hparams.num_measurements)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if not os.path.exists(npy_dir):
                    os.makedirs(npy_dir)

                top_idx = l2_pixl_loss_matrix.ravel().argsort()[:n_top]
                bot_idx = l2_pixl_loss_matrix.ravel().argsort()[-n_top:]
                
                lamb = hparams.lamb * (hparams.lamb_decay_rate ** hparams.adaptive_round_count)
                if hparams.adaptive_round_count == 0:
                    std_out, A = matrix_builder_data_driven(hparams, top_idx, bot_idx, lamb, hparams.adaptive_round_count, matrix=None)
                else:
                    std_in = np.load( os.path.join(load_npy_dir, 'energy_map_from_ROUND_{}.npy'.format(hparams.adaptive_round_count-1)) )
                    std_out, A = matrix_builder_data_driven(hparams, top_idx, bot_idx, lamb, hparams.adaptive_round_count, matrix=std_in)
                imsave( os.path.join(save_dir, 'Reconstruct_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)), img.astype(np.uint8))
                
                top_mask = np.zeros((28,28))
                bot_mask = np.zeros((28,28))
                top_pos = [(idx//28, idx%28) for idx in top_idx]
                bot_pos = [(idx//28, idx%28) for idx in bot_idx]
                for basis in top_pos:
                    top_mask[basis[0],basis[1]] = 1
                for basis in bot_pos:
                    bot_mask[basis[0],basis[1]] = 1
                top_mask[top_mask==0] = np.nan
                bot_mask[bot_mask==0] = np.nan
                
                np.save( os.path.join(npy_dir, 'least_error_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), top_mask) 
                np.save( os.path.join(npy_dir, 'most_error_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), bot_mask) 
                color_map(top_mask, os.path.join(save_dir, 'least_error_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))
                color_map(bot_mask, os.path.join(save_dir, 'most_error_from_ROUND_{}.jpg'.format(hparams.adaptive_round_count)))

                np.save( os.path.join(npy_dir, 'Measurement_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), A) 
                np.save( os.path.join(npy_dir, 'energy_map_from_ROUND_{}.npy'.format(hparams.adaptive_round_count)), std_out) 



if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./mnist_vae/models/mnist-vae/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='mnist', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='../images/*.jpg', help='Pattern to match to get images') ###july 2
    PARSER.add_argument('--num-input-images', type=int, default=1, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian_data_driven', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')
    PARSER.add_argument('--seed-no', type=int, default=256)
    PARSER.add_argument('--output-file-path', type=str, default='./result_mnist.txt')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--inpaint-size', type=int, default=1, help='size of block to inpaint')
    PARSER.add_argument('--superres-factor', type=int, default=2, help='how downsampled is the image')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--max-update-iter', type=int, default=1000, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=10, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')

    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.1, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')

    # k-sparse-wavelet specific hparams
    PARSER.add_argument('--sparsity', type=int, default=1, help='number of non zero entries allowed in k-sparse-wavelet')

    # Output
    PARSER.add_argument('--not-lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=1, help='checkpoint every x batches')
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

    #added by Young (all below)
    PARSER.add_argument('--variance-amount', type=float, default=4.0)
    PARSER.add_argument('--num-observe', type=int, default=49)
    PARSER.add_argument('--mnist_propotional_var_mask', type=str, default='./uncertainty/mnist_propotional_var_mask.npy')
    PARSER.add_argument('--pickle_file_path', type=str, default='./result_mnist.pkl')
    
    # adaptive
    PARSER.add_argument('--adaptive-round-count', type=int, default=-1)
    PARSER.add_argument('--sample-frequency', type=int, default=100)
    
    # annealed
    PARSER.add_argument('--annealed', dest='annealed', action='store_true')
    PARSER.add_argument('--no-annealed', dest='annealed', action='store_false')
    PARSER.set_defaults(annealed=True)
    
    # liklihood
    PARSER.add_argument('--mloss-weight', type=float, default=1.0, help='learning rate')
    PARSER.add_argument('--sigma-init', type=float, default=100, help='initial noise level for annealing langevin')
    PARSER.add_argument('--sigma-final', type=float, default=1.0, help='final noise level for annealing Langevin')
    
    # z
    PARSER.add_argument('--zprior-weight', type=float, default=0.1, help='weight on z prior')
    PARSER.add_argument('--zprior-sdev', type=float, default=1.0, help='standard deviation for target distributon of  z')
    
    # lr
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0., help='momentum value')

    # updated annealed langevin paras
    PARSER.add_argument('--start-std', type=float, default=4.0, help='initial variance of likelihood') #10
    PARSER.add_argument('--end-std', type=float, default=1.0, help='final variance of likelihood') #3
    PARSER.add_argument('--L', type=int, default=10, help='number of annealing steps') #10
    PARSER.add_argument('--T', type=int, default=200, help='number of iterations for each level of noise in Langevin annealing') #100
    PARSER.add_argument('--step', type=float, default=0.01, help="step size for langevin/gradient descent")

    # log
    PARSER.add_argument('--log-file-path', type=str, default='/home/sunyang/csgm/src/log_out_oct01.txt')

    # sampling
    PARSER.add_argument('--map-before-posterior', dest='map before posterior to optimize initialization', action='store_true')
    PARSER.set_defaults(map_before_posterior=True)
    PARSER.add_argument('--best-of-posterior', dest='choose the best posterior samples for subsequent map', action='store_true')
    PARSER.set_defaults(best_of_posterior=True)
    PARSER.add_argument('--generate-posterior-sample', dest='generate posterior samples to view', action='store_true')
    PARSER.set_defaults(generate_posterior_sample=True)

    # data driven
    PARSER.add_argument('--i', type=int, default=99, help='x-axis idx of block') 
    PARSER.add_argument('--j', type=int, default=99, help='x-axis idx of block') 
    PARSER.add_argument('--lamb', type=float, default=0.1, help='initial value of lambda') ###oct01!!!
    PARSER.add_argument('--lamb-decay-rate', type=float, default=0.95, help='lambda decay rate per round') ###oct01!!!
    PARSER.add_argument('--top-percent', type=float, default=0.33, help='choose top performing pixls') #10
    PARSER.add_argument('--block-size', type=int, default=4, help='choose top performing pixls') #10
    PARSER.add_argument('--load-seed-no', type=int, default=256)

    #mini batch data driven
    PARSER.add_argument('--mini-batch', type=int, default=0)
    PARSER.add_argument('--mini-batch-train', type=int, default=0)
    PARSER.add_argument('--last-mini-batch', type=int, default=0)
    PARSER.add_argument('--mini-batch-start-seed', type=int, default=1001)
    PARSER.add_argument('--mini-batch-end-seed', type=int, default=1020)
    PARSER.add_argument('--num-mini-batch', type=int, default=20)


    HPARAMS = PARSER.parse_args()

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        HPARAMS.mnist_block_mask = './uncertainty/mnist_var_top_{}_mask.npy'.format(HPARAMS.num_observe)
        HPARAMS.mnist_propotional_A = './uncertainty/mnist_propotional_A_mea_{}.npy'.format(HPARAMS.num_measurements)
        HPARAMS.mnist_block_data_driven_A = './uncertainty/mnist_block_{}_data_driven_A_i_{}_j_{}_mea_{}.npy'.format(
            HPARAMS.block_size, HPARAMS.i, HPARAMS.j, HPARAMS.num_measurements)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError

    main(HPARAMS)
