import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow.contrib.slim as slim
# import tf_slim as slim
# from models_64x64 import generator as gen
import matplotlib.pyplot as plt
# from config import argparser
# from data_process import noisy_meas

import tensorflow_probability as tfp
tfd = tfp.distributions
# from models_64x64 import generator as gen
# import tf_slim as slim
# from config import argparser
import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from itertools import combinations

from skimage.io import imread, imsave
from argparse import ArgumentParser
import utils
from skimage.io import imread, imsave

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from celebA_dcgan import model_def as celebA_dcgan_model_def

import faulthandler
import signal


def x_input(hparams, dim_like_, noise_var_):
    np.random.seed(hparams.seed_no)
    seed_no = hparams.seed_no
    xs_dict = model_input(hparams) #{0: img}
    x_true = xs_dict[0].reshape(hparams.image_shape)
    x_true = x_true.astype('float32')
    noisefile_path = 'noise3d_var{}_seed{}_dim{}.npy'.format(noise_var_, seed_no, dim_like_)
    if not os.path.exists(noisefile_path):
        print('**** noise file does not exist... creating new one ****')
        if dim_like_ != 12288:
            noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like_), dtype=np.float32), cov=np.eye(dim_like_, dim_like_)*noise_var_, size=1), (1,dim_like_))
        else:
            noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like_), dtype=np.float32), cov=np.eye(dim_like_, dim_like_)*noise_var_, size=1), (64,64,3)) #dtype=np.float32
        noise_mat3d = noise_mat3d.astype('float32')
        np.save(noisefile_path, noise_mat3d)
    noise_mat3d = np.load(noisefile_path)
    # x_noise = x_true + noise_mat3d 
    return x_true, noise_mat3d


def stats_main(hparams, mask, round_mcmc, n_sample, inpaint_size=8, used_indice_list=None, first_count=False, block_per_round=4, lambdas=0.9):
    
    #parameter 
    N = int(round_mcmc*2000)
    burn = int(0.5*N) #20000*0.5=10000
    n_eff = N-burn #10000
    batch_size = 1 
    z_dim = 100
    dim_prior = z_dim #100
    dim_like = 64*64*3
    noise_var = hparams.noise_std
    seed_no = hparams.seed_no


    
    #load posterior sample
    for i in range(round_mcmc):
        mcmc_samp = np.load('mask_{}_round_{}_mcmc_test_samples.npy'.format(int(np.sum(mask)), 2000*(i+1)))
        print('loaded mask_{}_round_{}_mcmc_test_samples.npy'.format(int(np.sum(mask)), 2000*(i+1)))
        if i == 0:
            mcmc_samps = np.squeeze(mcmc_samp) #2000,1,100 -> 2000,100
        else:
            mcmc_samps = np.concatenate((mcmc_samps, np.squeeze(mcmc_samp)), axis=0)
    print('total draws of posteriors: {}'.format(mcmc_samps.shape)) #20000,100


    #load posterior sample
    dis = int(n_eff/n_sample)
    eff_samps = mcmc_samps[burn:N:dis,:] 
    print('picked draws of posteriors: {}'.format(eff_samps.shape)) #2000,100


    #get input
    x_true, noise = x_input(hparams, dim_like, noise_var)
    x_noise = x_true + noise


    #mask
    # mask = np.ones((64, 64, 3)) #64 64 3
    # mask[start_row:end_row, start_col:end_col, :] = 0.
    dim_inpaint = int(np.sum(mask))


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')
    

    # Create the generator
    def dcgan_gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
        x_hat_full = x_hat_full[:batch_size]
        restore_vars = celebA_dcgan_model_def.gen_restore_vars()
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
        restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
        return x_hat_full, restore_dict, restore_path

    def gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=True)
        x_hat_full = x_hat_full[:batch_size]
        return x_hat_full[0]


    # X_hat, Initializer, Restore
    _, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)
    gen_out = gen(z_batch, hparams) #64 64 3
    diff_img = gen_out - tf.constant(x_noise) #64 64 3
    visible_img = tf.boolean_mask(diff_img, mask) #64 64 3
    visible_img = tf.reshape(visible_img, [dim_inpaint])


    #config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True


    #session
    with tf.Session(config=config) as sess:
        #Ini
        sess.run(tf.global_variables_initializer())

        #Load
        restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
        restorer_gen.restore(sess, restore_path_gen)
        
        #posterior generation 
        loss = np.zeros((len(eff_samps))) #e.g., 100
        x_mean = np.zeros((64,64,3))
        x2_mean = np.zeros((64,64,3))    
        
        for ii in range(n_sample):
            g_z, diff = sess.run([gen_out, visible_img], feed_dict={z_batch: np.expand_dims(eff_samps[ii,:], axis=0) }) #(200,64,64,3), (200,64,64,3)
            x_mean = x_mean + g_z #(64,64,3)
            x2_mean = x2_mean + g_z**2 #(64,64,3)
            #loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff[kk,:,:,:])**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2
            loss[ii] = 0.5*np.linalg.norm(diff)**2 + 0.5*noise_var*np.linalg.norm(eff_samps[ii, :])**2 #!!!!jun15
            # if ii % 20 == 0:
            #     g_z = (g_z+1.)/2.
            #     g_z = g_z * 255.
            #     imsave('mcmc_x_sample_{}.jpg'.format(ii), g_z.astype(np.uint8))
       
        
        #MAP
        map_ind = np.argmin(loss)
        # print('map index: {}'.format(map_ind))
        g_z_map = sess.run(gen_out, feed_dict={z_batch: np.expand_dims(eff_samps[map_ind,:], axis=0)}) #(64,64,3)
        rec_error = np.linalg.norm(g_z_map-x_true)/dim_like
        # print(' MAP recover loss: {}'.format(rec_error))


        #Bayesian posterior 
        x_mean = x_mean/n_sample
        x2_mean = x2_mean/n_sample    
        var = x2_mean - (x_mean)**2 #(64,63,3)
        var_m = np.mean(var, axis=-1) #(64,64)
        

        #saving
        def pixel_recover(img):
            img = (img+1.)/2.
            img = img * 255.
            return img

        def normalize(a):
            return((a-np.min(a))/(np.max(a)-np.min(a)))
        
        x_true = normalize(x_true) #(64,64,3)
        y_meas = x_true * mask
        y_meas[y_meas == 0.] = np.nan
        x_map = normalize(g_z_map) #(64,64,3)
        x_mean = normalize(x_mean) #(64,64,3)
        x_var = normalize(var)
        # imsave('mcmc_x_true.jpg', x_true.astype(np.uint8))
        # imsave('mcmc_y_meas.jpg', y_meas.astype(np.uint8))
        # imsave('mcmc_x_map.jpg', x_map.astype(np.uint8))
        # imsave('mcmc_x_mean.jpg', x_mean.astype(np.uint8))
        # imsave('mcmc_x_var.jpg', x_var.astype(np.uint8))
        
        fig, axs = plt.subplots(3,2, figsize=(20,20))
        im1 = axs[0][0].imshow(x_true)    
        fig.colorbar(im1, ax=axs[0][0])
        axs[0][0].set_title(r'$x_{{true}}$')
        im2 = axs[0][1].imshow(y_meas)    
        fig.colorbar(im2, ax=axs[0][1])
        axs[0][1].set_title(r'$y_{{meas}}$')
        
        im3 = axs[1][0].imshow(x_map)    
        fig.colorbar(im3, ax=axs[1][0])    
        axs[1][0].set_title(r'$x_{{map}}$')
        im4 = axs[1][1].imshow(x_map-x_true)    
        fig.colorbar(im4, ax=axs[1][1])
        axs[1][1].set_title(r'$x_{{map}} - x_{{true}}$')    
        
        im5 = axs[2][0].imshow(x_mean)    
        fig.colorbar(im5, ax=axs[2][0])
        axs[2][0].set_title(r'$x_{{mean}}$')
        im6 = axs[2][1].imshow(x_var)    
        fig.colorbar(im6, ax=axs[2][1])
        axs[2][1].set_title(r'$x_{{var}}$')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig('round_{}_adaptive_sampling.jpg'.format(int(dim_inpaint/(block_per_round*3*inpaint_size**2)))) ####
        plt.show()
        print('image saved')


        # #output
        # def var_block_finder(var_arr, win_size=8):
        #     arr_size = var_arr.shape[0]
        #     num_win = arr_size - win_size + 1
        #     best_mean = -99
        #     idxi, idxj = -1, -1
        #     for i in range(num_win):
        #         for j in range(num_win):
        #             mean = np.mean(var_arr[i:i+win_size, j:j+win_size])
        #             if mean > best_mean:
        #                 best_mean = mean
        #                 idxi, idxj = i, j
        #     return best_mean, idxi, idxj

        # max_var, idxi, idxj = var_block_finder(normalize(var_m), win_size=inpaint_size)

        # mask[idxi:idxi+inpaint_size, idxj:idxj+inpaint_size, :] = 1 #64,64,3
        # np.save('mask.npy', mask)
        # print('new mask idx {}, {}'.format(idxi, idxj))


        #new output
        # def mask_distance_norm(i,j,var_dis,win_size):
        #     min_dis = 100
        #     min_d, max_d = 1, (64/win_size-1)*np.sqrt(2)
        #     for tupple in var_dis:
        #         norm_dis = ( np.sqrt( (tupple[0]-i)**2 + (tupple[1]-j)**2 ) - min_d ) / ( max_d - min_d )
        #         if norm_dis < min_dis:
        #             min_dis = norm_dis 
        #     return min_dis


        # def var_block_distance_finder(var_arr, win_size, var_dis_, first_count_, lbd_=0.2):
        #     arr_size = var_arr.shape[0]
        #     num_win = int(arr_size/win_size)
        #     best_mean = -99
        #     best_dis = -99
        #     idxi, idxj = -1, -1
        #     for i in range(num_win):
        #         for j in range(num_win):
                    
        #             mean = np.mean(var_arr[i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size])
                    
        #             if first_count_:
        #                 dis = 0
        #                 if mean > best_mean:
        #                     best_mean = mean
        #                     best_dis = dis
        #                     idxi, idxj = i, j
        #                 var_dis_out = [(idxi, idxj)]
        #             else:
        #                 dis = mask_distance_norm(i,j,var_dis_, win_size)
        #                 if mean + lbd_ * dis > best_mean + lbd_ * best_dis: 
        #                     best_mean = mean
        #                     best_dis = dis
        #                     idxi, idxj = i, j
        #     if not first_count_:
        #         var_dis_.append((idxi, idxj))
        #         var_dis_out = var_dis_
                    
        #     return best_mean, best_dis, idxi, idxj, var_dis_out

        def mask_distance_norm(tuples, win_size):
            min_dis = 100
            min_d, max_d = 1, (64/win_size-1)*np.sqrt(2)
            for pair in combinations(tuples, 2):
                norm_dis = ( np.sqrt( (pair[0][0]-pair[1][0])**2 + (pair[0][1]-pair[1][1])**2 ) - min_d ) / ( max_d - min_d )
                if norm_dis < min_dis:
                    min_dis = norm_dis 
            return min_dis


        def var_block_distance_finder(var_arr, win_size, used_indice_, first_count_, num_block_per_round_=4, lbd_=0.9):
            #initials
            arr_size = var_arr.shape[0]
            num_win = int(arr_size/win_size)
            dis_idx_dic = {}
            
            #collect
            for i in range(num_win):
                for j in range(num_win):
                    mean = np.mean(var_arr[i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size])
                    dis_idx_dic[mean] = (i, j)
            
            #remove possible repeats
            if not first_count_:
                dis_idx_dic = {k: v for k, v in dis_idx_dic.items() if v not in used_indice_}

            #reverse sort
            sort_dis_idx_dic = {key:dis_idx_dic[key] for key in sorted(dis_idx_dic.keys(), reverse=True)[:int(num_block_per_round_*2)]}
            idx_candidates = [(key,sort_dis_idx_dic[key]) for key in sort_dis_idx_dic.keys()]
            
            #selection
            best_score = -99
            for quads in combinations(idx_candidates, num_block_per_round_):
                means = [quad[0] for quad in quads]   
                indice = [quad[1] for quad in quads] 
                mean_var = sum(means)/len(means)
                min_distance = mask_distance_norm(indice, win_size)
                score = 10 * mean_var + 10 * lbd_ * min_distance
                if score > best_score:
                    best_score = score
                    best_mean_var = mean_var
                    best_min_distance = min_distance
                    best_quads = [(quad[1][0],quad[1][1]) for quad in quads]
            
            if first_count_:
                used_indice_ = []
            used_indice_ += best_quads
            
            return best_score, best_mean_var, best_min_distance, best_quads, used_indice_

        best_score, best_mean_var, best_min_distance, best_quads, used_indice = var_block_distance_finder(normalize(var_m), 
            win_size=inpaint_size, used_indice_=used_indice_list, first_count_=first_count, num_block_per_round_=block_per_round, lbd_=lambdas)
        
        for idx_pair in best_quads:
            idxi, idxj = idx_pair
            mask[idxi*inpaint_size:(idxi+1)*inpaint_size, idxj*inpaint_size:(idxj+1)*inpaint_size, :] = 1 #64,64,3
        np.save('mask.npy', mask)


        #log
        print('mask info: {}'.format(dim_inpaint))
        print('MAP index: {}'.format(map_ind))
        print('MAP recover loss: {}'.format(rec_error))
        print('best_score {}'.format(best_score))
        print('best_mean_of_variance {}'.format(best_mean_var))
        print('best_minimum_distance {}'.format(best_min_distance))
        print('indice list {}'.format(best_quads))
        print('used_indice {}'.format(used_indice))
        
        return mask, used_indice


def mcmc_main(hparams, mask, round_mcmc=10, round_mcmc_start=0):
    
    #parameter
    dim_like = 64*64*3
    batch_size = 1
    N = 2000
    burn = int(0.5*N)
    noise_var = hparams.noise_std
    seed_no = hparams.seed_no


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')
    

    # Create the generator
    def dcgan_gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
        x_hat_full = x_hat_full[:batch_size]
        restore_vars = celebA_dcgan_model_def.gen_restore_vars()
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
        restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
        return x_hat_full, restore_dict, restore_path

    def gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=True)
        x_hat_full = x_hat_full[:batch_size]
        return x_hat_full[0]


    # X_hat output, Initializer, Restore
    x_hat_full, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)

    
    # get inputs
    # xs_dict = model_input(hparams) #{0: img}
    # x_true = xs_dict[0].reshape(hparams.image_shape)
    # noisefile_path = 'noise3d_var{}_seed{}.npy'.format(noise_var, seed_no)
    # if not os.path.exists(noisefile_path):
    #     print('**** noise file does not exist... creating new one ****')
    #     noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1), (64,64,3))
    #     np.save(noisefile_path, noise_mat3d)
    # noise_mat3d = np.load(noisefile_path)
    # x_noise = x_true + noise_mat3d ###!!!jun15
    x_true, noise = x_input(hparams, dim_like, noise_var)
    x_noise = x_true + noise


    
    #mask
    dim_inpaint = int(np.sum(mask))


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')


    #mcmc
    def joint_log_prob_ipt(z):              
        gen_out = gen(z, hparams) #64 64 3
        diff_img = gen_out - tf.constant(x_noise) #64 64 3
        visible_img = tf.boolean_mask(diff_img, mask) #64 64 3
        visible_img = tf.reshape(visible_img, [dim_inpaint])
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(100, dtype=np.float32), scale_diag=np.ones(100, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_inpaint, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_inpaint, dtype=np.float32)) ###!!!jun15
        return (prior.log_prob(z) + like.log_prob(visible_img))
                                            
    def unnormalized_posterior(z):
        return joint_log_prob_ipt(z)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior, 
        step_size=0.1,
        num_leapfrog_steps=3)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=int(burn * 0.8))

    samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
        num_results=N,
        num_burnin_steps=burn,
        current_state=z_batch,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size, pkr.inner_results.log_accept_ratio]) #check usuage?

    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))


    #config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    
    #session
    with tf.Session(config=config) as sess:
        #Ini
        sess.run(tf.global_variables_initializer())

        #Load
        restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
        restorer_gen.restore(sess, restore_path_gen)

        #mcmc
        for i in range(round_mcmc_start,round_mcmc): 
            if i == 0:
                z_ipt_inst = np.random.normal(size=[1, 100]).astype(np.float32)
                samples_ = sess.run(samples, feed_dict={z_batch: z_ipt_inst})
                np.save('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*(i+1)), samples_)
                print('mask {} round {}: acceptance ratio = {}'.format(dim_inpaint, i, sess.run(p_accept, feed_dict={z_batch: z_ipt_inst})))
                print('mask {} round {}: complete!'.format(dim_inpaint, i))
            else:
                inpaint_samps = np.load('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*i))
                print('successfully loaded: mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*i))
                z_ipt_inst = inpaint_samps[-1,:,:]
                samples_ = sess.run(samples, feed_dict={z_batch: z_ipt_inst})  
                np.save('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*(i+1)), samples_)
                print('mask {} round {}: acceptance ratio = {}'.format(dim_inpaint, i, sess.run(p_accept, feed_dict={z_batch: z_ipt_inst})))
                print('mask {} round {}: complete!'.format(dim_inpaint, i))




def compress_stats_main(hparams, mask, round_mcmc, n_sample, inpaint_size=8, used_indice_list=None, first_count=False, block_per_round=4, lambdas=0.9):
    
    #parameter 
    N = int(round_mcmc*2000)
    burn = int(0.5*N) #20000*0.5=10000
    n_eff = N-burn #10000
    batch_size = 1 
    z_dim = 100
    dim_prior = z_dim #100
    dim_like = 64*64*3
    num_measurements = 500
    noise_var = hparams.noise_std

    
    #load posterior sample
    for i in range(round_mcmc):
        mcmc_samp = np.load('mask_{}_round_{}_mcmc_test_samples.npy'.format(int(np.sum(mask)), 2000*(i+1)))
        print('loaded mask_{}_round_{}_mcmc_test_samples.npy'.format(int(np.sum(mask)), 2000*(i+1)))
        if i == 0:
            mcmc_samps = np.squeeze(mcmc_samp) #2000,1,100 -> 2000,100
        else:
            mcmc_samps = np.concatenate((mcmc_samps, np.squeeze(mcmc_samp)), axis=0)
    print('total draws of posteriors: {}'.format(mcmc_samps.shape)) #20000,100


    #load posterior sample
    dis = int(n_eff/n_sample)
    eff_samps = mcmc_samps[burn:N:dis,:] 
    print('picked draws of posteriors: {}'.format(eff_samps.shape)) #2000,100


    #get input
    # xs_dict = model_input(hparams) #{0: img}
    # x_true = xs_dict[0].reshape(1,-1) #(1,12288) 
    x_true, noise = x_input(hparams, num_measurements, noise_var)
    


    #matrix
    dim_compress = int(np.sum(mask)) 
    A = np.random.randn(dim_like, num_measurements)


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')
    

    # Create the generator
    def dcgan_gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
        x_hat_full = x_hat_full[:batch_size]
        restore_vars = celebA_dcgan_model_def.gen_restore_vars()
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
        restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
        return x_hat_full, restore_dict, restore_path

    def gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=True)
        x_hat_full = x_hat_full[:batch_size]
        return x_hat_full[0]


    # X_hat, Initializer, Restore
    _, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)
    gen_out = gen(z_batch, hparams) #64 64 3
    visible_out = gen_out * mask
    gen_out_compress = tf.matmul(tf.reshape(visible_out, [1,-1]), A) #(1,12288) @ (12288,n_mea) = (1,n_mea)
    x_true_compress = tf.matmul(x_true, A) #(1,12288) @ (12288,n_mea) = (1,n_mea)
    diff_img = gen_out_compress - x_true_compress + noise #(1,n_mea)


    #config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True


    #session
    with tf.Session(config=config) as sess:
        #Ini
        sess.run(tf.global_variables_initializer())

        #Load
        restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
        restorer_gen.restore(sess, restore_path_gen)
        
        #posterior generation 
        loss = np.zeros((len(eff_samps))) #e.g., 100
        x_mean = np.zeros((64,64,3))
        x2_mean = np.zeros((64,64,3))    
        
        for ii in range(n_sample):
            g_z, diff = sess.run([gen_out, diff_img], feed_dict={z_batch: np.expand_dims(eff_samps[ii,:], axis=0) }) #(200,64,64,3), (200,64,64,3)
            x_mean = x_mean + g_z #(64,64,3)
            x2_mean = x2_mean + g_z**2 #(64,64,3)
            #loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff[kk,:,:,:])**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2
            #loss[ii] = 0.5*np.linalg.norm(diff)**2
            loss[ii] = 0.5*np.linalg.norm(diff)**2 + 0.5*noise_var*np.linalg.norm(eff_samps[ii, :])**2 #!!!!jun15
            # if ii % 20 == 0:
            #     g_z = (g_z+1.)/2.
            #     g_z = g_z * 255.
            #     imsave('mcmc_x_sample_{}.jpg'.format(ii), g_z.astype(np.uint8))
       
        
        #MAP
        map_ind = np.argmin(loss)
        # print('map index: {}'.format(map_ind))
        g_z_map = sess.run(gen_out, feed_dict={z_batch: np.expand_dims(eff_samps[map_ind,:], axis=0)}) #(64,64,3)
        rec_error = np.linalg.norm(g_z_map.reshape(1,-1)-x_true)/dim_like
        # print(' MAP recover loss: {}'.format(rec_error))


        #Bayesian posterior 
        x_mean = x_mean/n_sample
        x2_mean = x2_mean/n_sample    
        var = x2_mean - (x_mean)**2 #(64,63,3)
        var_m = np.mean(var, axis=-1) #(64,64)
        

        #saving
        def pixel_recover(img):
            img = (img+1.)/2.
            img = img * 255.
            return img

        def normalize(a):
            return((a-np.min(a))/(np.max(a)-np.min(a)))
        
        x_true = x_true.reshape(hparams.image_shape)
        x_true = normalize(x_true) #(64,64,3)
        y_meas = x_true * mask
        y_meas[y_meas == 0.] = np.nan
        x_map = normalize(g_z_map) #(64,64,3)
        x_mean = normalize(x_mean) #(64,64,3)
        x_var = normalize(var)
        
        fig, axs = plt.subplots(3,2, figsize=(20,20))
        im1 = axs[0][0].imshow(x_true)    
        fig.colorbar(im1, ax=axs[0][0])
        axs[0][0].set_title(r'$x_{{true}}$')
        im2 = axs[0][1].imshow(y_meas)    
        fig.colorbar(im2, ax=axs[0][1])
        axs[0][1].set_title(r'$y_{{meas}}$')
        
        im3 = axs[1][0].imshow(x_map)    
        fig.colorbar(im3, ax=axs[1][0])    
        axs[1][0].set_title(r'$x_{{map}}$')
        im4 = axs[1][1].imshow(x_map-x_true)    
        fig.colorbar(im4, ax=axs[1][1])
        axs[1][1].set_title(r'$x_{{map}} - x_{{true}}$')    
        
        im5 = axs[2][0].imshow(x_mean)    
        fig.colorbar(im5, ax=axs[2][0])
        axs[2][0].set_title(r'$x_{{mean}}$')
        im6 = axs[2][1].imshow(x_var)    
        fig.colorbar(im6, ax=axs[2][1])
        axs[2][1].set_title(r'$x_{{var}}$')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.savefig('round_{}_adaptive_sampling.jpg'.format(int(dim_compress/(block_per_round*3*inpaint_size**2)))) 
        plt.show()
        print('image saved')


        #variance info 
        def mask_distance_norm(tuples, win_size):
            min_dis = 100
            min_d, max_d = 1, (64/win_size-1)*np.sqrt(2)
            for pair in combinations(tuples, 2):
                norm_dis = ( np.sqrt( (pair[0][0]-pair[1][0])**2 + (pair[0][1]-pair[1][1])**2 ) - min_d ) / ( max_d - min_d )
                if norm_dis < min_dis:
                    min_dis = norm_dis 
            return min_dis


        def var_block_distance_finder(var_arr, win_size, used_indice_, first_count_, num_block_per_round_=4, lbd_=0.9):
            #initials
            arr_size = var_arr.shape[0]
            num_win = int(arr_size/win_size)
            dis_idx_dic = {}
            
            #collect
            for i in range(num_win):
                for j in range(num_win):
                    mean = np.mean(var_arr[i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size])
                    dis_idx_dic[mean] = (i, j)
            
            #remove possible repeats
            if not first_count_:
                dis_idx_dic = {k: v for k, v in dis_idx_dic.items() if v not in used_indice_}

            #reverse sort
            sort_dis_idx_dic = {key:dis_idx_dic[key] for key in sorted(dis_idx_dic.keys(), reverse=True)[:int(num_block_per_round_*2)]}
            idx_candidates = [(key,sort_dis_idx_dic[key]) for key in sort_dis_idx_dic.keys()]
            
            #selection
            best_score = -99
            for quads in combinations(idx_candidates, num_block_per_round_):
                means = [quad[0] for quad in quads]   
                indice = [quad[1] for quad in quads] 
                mean_var = sum(means)/len(means)
                min_distance = mask_distance_norm(indice, win_size)
                score = 10 * mean_var + 10 * lbd_ * min_distance
                if score > best_score:
                    best_score = score
                    best_mean_var = mean_var
                    best_min_distance = min_distance
                    best_quads = [(quad[1][0],quad[1][1]) for quad in quads]
            
            if first_count_:
                used_indice_ = []
            used_indice_ += best_quads
            
            return best_score, best_mean_var, best_min_distance, best_quads, used_indice_

        best_score, best_mean_var, best_min_distance, best_quads, used_indice = var_block_distance_finder(normalize(var_m), 
            win_size=inpaint_size, used_indice_=used_indice_list, first_count_=first_count, num_block_per_round_=block_per_round, lbd_=lambdas)
        
        for idx_pair in best_quads:
            idxi, idxj = idx_pair
            mask[idxi*inpaint_size:(idxi+1)*inpaint_size, idxj*inpaint_size:(idxj+1)*inpaint_size, :] = 1 #64,64,3

        np.save('mask.npy', mask)


        #log
        print('mask info: {}'.format(dim_compress))
        print('MAP index: {}'.format(map_ind))
        print('MAP recover loss: {}'.format(rec_error))
        print('best_score {}'.format(best_score))
        print('best_mean_of_variance {}'.format(best_mean_var))
        print('best_minimum_distance {}'.format(best_min_distance))
        print('indice list {}'.format(best_quads))
        print('used_indice {}'.format(used_indice))
        
        return mask, used_indice


def compress_mcmc_main(hparams, mask, round_mcmc=10, round_mcmc_start=0):
    
    #parameter
    dim_like = 64*64*3
    batch_size = 1
    N = 2000
    burn = int(0.5*N)
    num_measurements = 500 #!!!!!!!!!only changeable jun16, need to align with number of ones (in one dimension) in mask 
    noise_var = hparams.noise_std


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')
    

    # Create the generator
    def dcgan_gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=False)
        x_hat_full = x_hat_full[:batch_size]
        restore_vars = celebA_dcgan_model_def.gen_restore_vars()
        restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
        restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)
        return x_hat_full, restore_dict, restore_path

    def gen(z, hparams):
        assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
        z_full = tf.zeros([64, 100]) + z
        model_hparams = celebA_dcgan_model_def.Hparams()
        x_hat_full = celebA_dcgan_model_def.generator(model_hparams, z_full, train=False, reuse=True)
        x_hat_full = x_hat_full[:batch_size]
        return x_hat_full[0]


    # X_hat output, Initializer, Restore
    x_hat_full, restore_dict_gen, restore_path_gen = dcgan_gen(z_batch, hparams)

    
    # get inputs
    # xs_dict = model_input(hparams) #{0: img}
    # x_true = xs_dict[0].reshape(1,-1)
    # print('original type:')
    # print(x_true.dtype)
    x_true, noise = x_input(hparams, num_measurements, noise_var) #!!!!!!!!!only changeable jun16
    
    #mask
    dim_inpaint = int(np.sum(mask))
    A = np.random.randn(dim_like, num_measurements) #!!!!!!!!!only changeable jun16


    # Set up palceholders
    z_batch = tf.Variable(tf.random_normal([1, 100]), name='z_batch')


    #mcmc
    def joint_log_prob_ipt(z):              
        gen_out = gen(z, hparams) #64 64 3
        # diff_img = gen_out - tf.constant(x_true) #64 64 3
        # visible_img = tf.boolean_mask(diff_img, mask) #64 64 3
        # visible_img = tf.reshape(visible_img, [dim_inpaint])
        visible_out = gen_out * mask
        gen_out_compress = tf.matmul(tf.reshape(visible_out, [1,-1]), A) #(1,12288) @ (12288,n_mea) = (1,n_mea)
        x_true_compress = tf.matmul(x_true, A) #(1,12288) @ (12288,n_mea) = (1,n_mea)
        diff_img = gen_out_compress - x_true_compress + noise #(1,n_mea)

        prior = tfd.MultivariateNormalDiag(loc=np.zeros(100, dtype=np.float32), scale_diag=np.ones(100, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(num_measurements, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(num_measurements, dtype=np.float32))
        return (prior.log_prob(z) + like.log_prob(diff_img))
                                            
    def unnormalized_posterior(z):
        return joint_log_prob_ipt(z)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior, 
        step_size=0.1,
        num_leapfrog_steps=3)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=int(burn * 0.8))

    samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
        num_results=N,
        num_burnin_steps=burn,
        current_state=z_batch,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size, pkr.inner_results.log_accept_ratio]) #check usuage?

    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))


    #config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    
    #session
    with tf.Session(config=config) as sess:
        #Ini
        sess.run(tf.global_variables_initializer())

        #Load
        restorer_gen = tf.train.Saver(var_list=restore_dict_gen)
        restorer_gen.restore(sess, restore_path_gen)

        #mcmc
        for i in range(round_mcmc_start,round_mcmc): 
            if i == 0:
                z_ipt_inst = np.random.normal(size=[1, 100]).astype(np.float32)
                samples_ = sess.run(samples, feed_dict={z_batch: z_ipt_inst})
                np.save('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*(i+1)), samples_)
                print('mask {} round {}: acceptance ratio = {}'.format(dim_inpaint, i, sess.run(p_accept, feed_dict={z_batch: z_ipt_inst})))
                print('mask {} round {}: complete!'.format(dim_inpaint, i))
            else:
                inpaint_samps = np.load('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*i))
                print('successfully loaded: mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*i))
                z_ipt_inst = inpaint_samps[-1,:,:]
                samples_ = sess.run(samples, feed_dict={z_batch: z_ipt_inst})  
                np.save('mask_{}_round_{}_mcmc_test_samples.npy'.format(dim_inpaint, N*(i+1)), samples_)
                print('mask {} round {}: acceptance ratio = {}'.format(dim_inpaint, i, sess.run(p_accept, feed_dict={z_batch: z_ipt_inst})))
                print('mask {} round {}: complete!'.format(dim_inpaint, i))





if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='../models/celebA/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='../../GANPriors-master/celeba/data/test/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=1, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='inpaint', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')
    PARSER.add_argument('--seed-no', type=int, default=1008)

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--inpaint-size', type=int, default=4, help='size of block to inpaint')
    PARSER.add_argument('--superres-factor', type=int, default=2, help='how downsampled is the image')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default='dcgan', help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.0, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate') #######################
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
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
        
    HPARAMS = PARSER.parse_args()

    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        # from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError


    
    faulthandler.register(signal.SIGUSR1)
    #adaptive_mcmc
    for i in range(12):
        if i == 0:
            mcmc_main(HPARAMS, np.zeros((64, 64, 3)), round_mcmc=1)
            new_mask, new_used_indice_list = stats_main(HPARAMS, mask=np.zeros((64, 64, 3)), round_mcmc=1, n_sample=100, inpaint_size=8, used_indice_list=None, first_count=True, block_per_round=4, lambdas=1.5)
        else:
            mcmc_main(HPARAMS, new_mask, round_mcmc=1)
            new_mask, new_var_list = stats_main(HPARAMS, mask=new_mask, round_mcmc=1, n_sample=100, inpaint_size=8, used_indice_list=new_used_indice_list, first_count=False, block_per_round=4, lambdas=1.5)

    
    # # adaptive compressing 
    # for i in range(12):
    #     if i == 0:
    #         compress_mcmc_main(HPARAMS, np.zeros((64, 64, 3)), round_mcmc=1)
    #         new_mask, new_used_indice_list = compress_stats_main(HPARAMS, mask=np.zeros((64, 64, 3)), round_mcmc=1, n_sample=100, inpaint_size=8, used_indice_list=None, first_count=True, block_per_round=4, lambdas=1.5)
    #     else:
    #         compress_mcmc_main(HPARAMS, new_mask, round_mcmc=1)
    #         new_mask, new_var_list = compress_stats_main(HPARAMS, mask=new_mask, round_mcmc=1, n_sample=100, inpaint_size=8, used_indice_list=new_used_indice_list, first_count=False, block_per_round=4, lambdas=1.5)
    
    
    # new_mask = stats_main(HPARAMS, round_mcmc=10, n_sample=2000, inpaint_size=8)
    # new_mask = np.load('mask.npy')
    # mcmc_main(HPARAMS, new_mask, round_mcmc=2)

    # new_mask = np.zeros((64, 64, 3))
    # new_mask[0:16, 0:16, :] = 1
    # new_mask[0:16, 48:64, :] = 1
    # new_mask[48:64, 0:16, :] = 1
    # new_mask[48:64, 48:64, :] = 1
    # new_mask[0:16, 16:32, :] = 1
    
    # new_var_list = [(0,0)]
    
    # for i in range(4):
    #     if i == 0:
    #         # mcmc_main(HPARAMS, np.zeros((64, 64, 3)), round_mcmc=1)
    #         new_mask, new_var_list = stats_main(HPARAMS, mask=new_mask, round_mcmc=1, n_sample=100, inpaint_size=16, var_dis_list=new_var_list, first_count=False, lambdas=3.0)
    #     else:
    #         mcmc_main(HPARAMS, new_mask, round_mcmc=1)
    #         new_mask, new_var_list = stats_main(HPARAMS, mask=new_mask, round_mcmc=1, n_sample=100, inpaint_size=16, var_dis_list=new_var_list, first_count=False, lambdas=3.0)
