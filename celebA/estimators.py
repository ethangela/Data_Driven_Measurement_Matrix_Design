"""Estimators for compressed sensing"""
import copy
import heapq
import torch
import numpy as np
import scipy.fftpack as fftpack
import sys
import os
import utils as utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from stylegan2.model import Generator
# from ncsnv2.models import get_sigmas, ema
# from ncsnv2.models.ncsnv2 import NCSNv2, NCSNv2Deepest
from glow import model as glow_model

try:
    import tensorflow as tf
    # import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()
except:
    print('did not import tensorflow')

import torch.nn.functional as F

# from PULSE import PULSE
import yaml
import argparse
import time
import math
from PIL import Image
# from include import fit, decoder

def get_measurements_torch(x_hat_batch, A, measurement_type, hparams):
    batch_size = hparams.batch_size
    if measurement_type == 'project':
        y_hat_batch = x_hat_batch
    elif measurement_type == 'gaussian':
        y_hat_batch = torch.mm(x_hat_batch.view(batch_size,-1), A)
    elif measurement_type == 'circulant':
        sign_pattern = torch.Tensor(hparams.sign_pattern).to(hparams.device)
        y_hat_batch = utils.partial_circulant_torch(x_hat_batch, A, hparams.train_indices,sign_pattern)
    elif measurement_type == 'superres':
        x_hat_reshape_batch = x_hat_batch.view((batch_size,) + hparams.image_shape)
        y_hat_batch = F.avg_pool2d(x_hat_reshape_batch, hparams.downsample)
    return y_hat_batch.view(batch_size, -1)


def glow_annealed_map_estimator(hparams):

    annealed = hparams.annealed
    # set up model and session
    dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev)

    x_hat_batch_nhwc = dec_x + 0.5


    # Set up palceholders
    if hparams.measurement_type == 'circulant':
        A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
    else:
        A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A') #jul28 
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # convert from NHWC to NCHW
    # since I used pytorch for reading data, the measurements
    # are from a ground truth of data format NCHW
    # Meanwhile GLOW's output has format NHWC
    x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])
    # RESECALE aug25
    x_hat_batch_nchw = x_hat_batch_nchw[:,:,64:192,64:192] #[b,3,128,128]

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
    elif hparams.measurement_type == 'circulant':
        sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
        y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
    elif hparams.measurement_type == 'superres':
        y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))
    else:
        y_hat_batch = tf.matmul(tf.reshape(x_hat_batch_nchw, (hparams.batch_size,-1)), A, name='y2_batch') # (b,n) @ (n,m) = (b,mea) #jul28  

    # define all losses
    z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
    z_stack = tf.concat(z_list, axis=1)
    z_loss_batch = tf.reduce_sum(z_stack**2, 1)
    z_loss = tf.reduce_sum(z_loss_batch)
    y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_sum(y_loss_batch)

    # mloss_weight should be m/2sigma^2 for proper langevin
    # zprior_weight should be 1/(2*0.49) for proper langevin
    sigma = tf.placeholder(tf.float32, shape=[])
    

    mloss_weight = 0.5 * hparams.num_measurements / (sigma ** 2)
    zprior_weight = 1/(2 * hparams.zprior_sdev**2)

    
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')

        sess = utils.tensorflow_session()

        # initialize variables
        uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])
    # sess.run(init_op)

    def estimator(A_val, y_val, hparams):
        """added info, 26 Jul"""
        xs_dict = utils.model_input(hparams) #{0: img}
        original_img = xs_dict[0].reshape((3, 256, 256))[:,64:192,64:192]
        original_img = original_img.reshape(-1)
        noise_info = '_'.join(str(hparams.noise_std).split('.'))

        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        # best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)

        feed_dict = hparams.feed_dict.copy()
        feed_dict.update({A: A_val, y_batch: y_val})

        for i in range(hparams.num_random_restarts):
            sess.run(init_op)
            # sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):

                factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1))

                if annealed:
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2)
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T)
                    lr_value = hparams.learning_rate * lr_lambda(j)
                else:
                    sigma_value = hparams.sigma_final
                    lr_value = hparams.learning_rate

                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})
                _, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, total_loss, z_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {}/{}  lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, hparams.max_update_iter, lr_value, total_loss_value, y_loss_value, z_loss_value))


            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)
            x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
            best_keeper.report(x_hat_batch_value, total_loss_batch_value)
            # best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
        
        #log
        recovery_loss = utils.get_l2_loss(original_img, best_keeper.get_best().reshape(-1))
        center_rloss, corner_rloss = utils.get_l2_loss_region(original_img, best_keeper.get_best().reshape(-1))
        tmp = sys.stdout   
        sys.stdout = open(hparams.log_file_path, 'a') 
        print('Measurement type {}, num {}, Noise {}, Best_Recovery_Loss {} Center_loss {}, Corner_loss {}'.format( hparams.measurement_type, 
            hparams.num_measurements, noise_info, recovery_loss, center_rloss, corner_rloss ))
        print('\t')
        sys.stdout.close()  
        sys.stdout = tmp  
        
        # return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best
        return best_keeper.get_best(), best_keeper.losses_val_best


    return estimator


def glow_annealed_langevin_estimator(hparams):

    annealed = hparams.annealed
    # set up model and session
    dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev)

    x_hat_batch_nhwc = dec_x + 0.5


    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(1,hparams.n_input), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')


    # convert from NHWC to NCHW
    # since I used pytorch for reading data, the measurements
    # are from a ground truth of data format NCHW
    # Meanwhile GLOW's output has format NHWC
    x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2])

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch_nchw, name='y2_batch')
    elif hparams.measurement_type == 'circulant':
        sign_pattern_tf = tf.constant(hparams.sign_pattern, name='sign_pattern')
        y_hat_batch = utils.partial_circulant_tf(x_hat_batch_nchw, A, hparams.train_indices, sign_pattern_tf)
    elif hparams.measurement_type == 'superres':
        y_hat_batch = tf.reshape(utils.blur(x_hat_batch_nchw, hparams.downsample),(hparams.batch_size, -1))
    else: #gaussian
        y_hat_batch = tf.matmul(tf.reshape(x_hat_batch_nchw, (hparams.batch_size,-1)), A, name='y2_batch') # (b,n) @ (n,m) = (b,mea) #jul28    

    # create noise placeholders for langevin
    noise_vars = [tf.placeholder(tf.float32, shape=dec_eps[i].get_shape()) for i in range(len(dec_eps))]

    # define all losses
    z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
    z_stack = tf.concat(z_list, axis=1)
    z_loss_batch = tf.reduce_sum(z_stack**2, 1)
    z_loss = tf.reduce_sum(z_loss_batch)
    y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_sum(y_loss_batch)

    # mloss_weight should be m/2sigma^2 for proper langevin
    # zprior_weight should be 0.5 for proper langevin
    sigma = tf.placeholder(tf.float32, shape=[])
    mloss_weight = 0.5 * hparams.num_measurements / (sigma ** 2)
    if hparams.zprior_weight is None:
        zprior_weight = 0.5
    else:
        zprior_weight = hparams.zprior_weight
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')
        noise_ops = [dec_eps[i].assign_add(noise_vars[i]) for i in range(len(dec_eps))]

        sess = utils.tensorflow_session()

        # initialize variables
        uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        feed_dict = hparams.feed_dict.copy()

        if hparams.measurement_type == 'circulant':
            feed_dict.update({A: A_val, y_batch: y_val})
        else:
            feed_dict.update({y_batch: y_val})

        for i in range(hparams.num_random_restarts):
            sess.run(init_op)
            for j in range(hparams.max_update_iter):
                if hparams.gif and (( j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch_nchw, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)


                factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1))

                if annealed:
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2)
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T)
                else:
                    lr_lambda = lambda t: 1
                    sigma_value = hparams.sigma_final
                lr_value = hparams.learning_rate * lr_lambda(j)

                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})
                _, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, total_loss, z_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))
                if math.isnan(total_loss_value):
                    raise Exception("infinity loss emerged!")

                # gradient_noise_weight should be sqrt(2*lr) for proper langevin
                gradient_noise_weight = np.sqrt(2*lr_value/(1-hparams.momentum))
                for noise_var in noise_vars:
                    noise_shape = noise_var.get_shape().as_list()
                    feed_dict.update({noise_var: gradient_noise_weight *np.random.randn(hparams.batch_size, noise_shape[1], noise_shape[2], noise_shape[3])})
                results = run(sess,noise_ops,feed_dict)


            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)

            x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
            best_keeper.report(x_hat_batch_value, total_loss_batch_value)
            best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)
        return best_keeper.get_best(), best_keeper_z.get_best(), best_keeper.losses_val_best

    return estimator



def deep_decoder_estimator(hparams):
    num_channels = [700]*7
    output_depth = 3 # hparams.image_size

    def estimator(A_val, y_val, hparams):

        y_batch = torch.Tensor(y_val).to(hparams.device)
        if A_val is not None:
            A = torch.Tensor(A_val).to(hparams.device)
        else:
            A = None

        def apply_f(x):
            return get_measurements_torch(x.view(hparams.batch_size,-1),A,hparams.measurement_type,hparams)

        net = decoder.decodernw(output_depth, num_channels_up=num_channels, upsample_first=True).cuda()

        rn = 0.005
        rnd = 500
        numit = 4000

        print(hparams.max_update_iter)
        mse_n, mse_t, ni, net = fit(
                       num_channels=num_channels,
                        reg_noise_std=rn,
                        reg_noise_decayevery = rnd,
                        num_iter=hparams.max_update_iter,
                        LR=hparams.learning_rate,
                        OPTIMIZER=hparams.optimizer_type,
                        img_noisy_var=y_batch,
                        net=net,
                        img_clean_var=torch.zeros_like(y_batch),
                        find_best=True,
                        apply_f=apply_f,
                        )
        return net(ni.cuda()).view(hparams.batch_size,-1).detach().cpu().numpy(), np.zeros(hparams.batch_size), np.zeros(hparams.batch_size)

    return estimator



def glow_annealed_langevin_estimator_adaptive(hparams):

    annealed = hparams.annealed
    
    # set up model and session
    dec_x, dec_eps, hparams.feed_dict, run = glow_model.get_model(hparams.checkpoint_path, hparams.batch_size, hparams.zprior_sdev)
    x_hat_batch_nhwc = dec_x + 0.5


    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A') #jul28 
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')


    # convert from NHWC to NCHW
    # since I used pytorch for reading data, the measurements
    # are from a ground truth of data format NCHW
    # Meanwhile GLOW's output has format NHWC
    x_hat_batch_nchw = tf.transpose(x_hat_batch_nhwc, perm = [0,3,1,2]) #[b,3,256,256]
    # RESECALE aug25
    x_hat_batch_nchw = x_hat_batch_nchw[:,:,64:192,64:192] #[b,3,128,128]

    # measure the estimate
    y_hat_batch = tf.matmul(tf.reshape(x_hat_batch_nchw, (hparams.batch_size,-1)), A, name='y2_batch') # (b,n) @ (n,m) = (b,mea) #jul28       

    # create noise placeholders for langevin
    noise_vars = [tf.placeholder(tf.float32, shape=dec_eps[i].get_shape()) for i in range(len(dec_eps))]

    # define all losses
    z_list = [tf.reshape(dec_eps[i],(hparams.batch_size,-1)) for i in range(6)]
    z_stack = tf.concat(z_list, axis=1)
    z_loss_batch = tf.reduce_sum(z_stack**2, 1)
    z_loss = tf.reduce_sum(z_loss_batch)
    y_loss_batch = tf.reduce_sum((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_sum(y_loss_batch)

    # mloss_weight should be m/2sigma^2 for proper langevin
    # zprior_weight should be 0.5 for proper langevin
    sigma = tf.placeholder(tf.float32, shape=[])
    mloss_weight = 0.5 * hparams.num_measurements / (sigma ** 2)
    if hparams.zprior_weight is None:
        zprior_weight = 0.5
    else:
        zprior_weight = hparams.zprior_weight
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * z_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = utils.get_optimizer(dec_eps,learning_rate, hparams)
        update_op = opt.minimize(total_loss, var_list=dec_eps, global_step=global_step, name='update_op')
        noise_ops = [dec_eps[i].assign_add(noise_vars[i]) for i in range(len(dec_eps))]

        sess = utils.tensorflow_session()

        # initialize variables
        uninitialized_vars = set(sess.run(tf.report_uninitialized_variables()))
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.op.name.encode('UTF-8') in uninitialized_vars])

    def estimator(A_val, y_val, hparams):
        """added info, 26 Jul"""
        xs_dict = utils.model_input(hparams) #{0: img}
        original_img = xs_dict[0].reshape((3, 256, 256))[:,64:192,64:192]
        original_img = original_img.reshape(-1)
        noise_info = '_'.join(str(hparams.noise_std).split('.'))

        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        # best_keeper_z = utils.BestKeeper(hparams.batch_size, hparams.n_input)
        
        feed_dict = hparams.feed_dict.copy()
        feed_dict.update({A: A_val, y_batch: y_val})

        # RESECALE aug25
        x_mean = np.zeros((128,128,3))
        x2_mean = np.zeros((128,128,3))
        n_sample = 0
        recovery_loss_sum = 0
        
        for i in range(hparams.num_random_restarts):
            sess.run(init_op)
            for j in range(hparams.max_update_iter):

                factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1))

                if annealed:
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2)
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T)
                else:
                    lr_lambda = lambda t: 1
                    sigma_value = hparams.sigma_final
                lr_value = hparams.learning_rate * lr_lambda(j)

                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})
                _, total_loss_value, z_loss_value, y_loss_value = run(sess, [update_op, total_loss, z_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {}/{} lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, hparams.max_update_iter, lr_value, total_loss_value, y_loss_value, z_loss_value))

                # gradient_noise_weight should be sqrt(2*lr) for proper langevin
                gradient_noise_weight = np.sqrt(2*lr_value/(1-hparams.momentum))
                for noise_var in noise_vars:
                    noise_shape = noise_var.get_shape().as_list()
                    feed_dict.update({noise_var: gradient_noise_weight *np.random.randn(hparams.batch_size, noise_shape[1], noise_shape[2], noise_shape[3])})
                results = run(sess, noise_ops,feed_dict)

                # posterior sampling
                if j >= int(0.5*hparams.max_update_iter) and j % hparams.sample_frequency == 0: 
                    x_hat_batch_sample = run(sess, x_hat_batch_nchw, feed_dict=feed_dict)
                    assert x_hat_batch_sample.shape[0] == 1
                    x_hat_batch_sample_hwc = np.transpose(x_hat_batch_sample[0,:,:,:], (1, 2, 0))
                    x_mean += x_hat_batch_sample_hwc
                    x2_mean += x_hat_batch_sample_hwc**2
                    n_sample += 1
                    recovery_loss = utils.get_l2_loss(original_img.reshape(hparams.image_shape), np.transpose(x_hat_batch_sample_hwc, (2,0,1))) 
                    # logging_format = 'SAMPLE {} restart {} L {} T {} lr {} sigma {} total_loss {} y_loss {} z_loss {} recovery_loss {}'
                    # print(logging_format.format(n_sample, restart, l, t, lr_value, sigma_value, total_loss_value, y_loss_value, z_loss_value, recovery_loss))
                    recovery_loss_sum += recovery_loss

                    # save image
                    def save_image(image, path):
                        """Save an image as a png file"""
                        x_png = np.uint8(np.clip(image*256,0,255))
                        x_png = x_png.transpose(1,2,0)
                        if x_png.shape[-1] == 1:
                            x_png = x_png[:,:,0]
                        x_png = Image.fromarray(x_png).save(path)
                    
                    save_dir = '../../src/{}/adaptive/noise_{}/measurement_{}/samples'.format(hparams.img_no, noise_info, hparams.num_measurements)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_image(x_hat_batch_sample[0,:,:,:], os.path.join(save_dir,'sample_{}.png'.format(j)))

            #old output
            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = run(sess, [x_hat_batch_nchw, z_stack, total_loss_batch], feed_dict=feed_dict)
            x_hat_batch_value = x_hat_batch_value.reshape(hparams.batch_size, -1)
            best_keeper.report(x_hat_batch_value, total_loss_batch_value)
            # best_keeper_z.report(z_hat_batch_value, total_loss_batch_value)

        # samples mean/var
        recovery_loss_sum = recovery_loss_sum/n_sample
        x_mean = x_mean/n_sample
        x2_mean = x2_mean/n_sample    
        var = x2_mean - (x_mean)**2 #(256,256,3)
    
        #log
        center_rloss, corner_rloss = utils.get_l2_loss_region(original_img, best_keeper.get_best().reshape(-1))
        tmp = sys.stdout   
        sys.stdout = open(hparams.log_file_path, 'a') 
        print('Measurement type {}, num {}, Noise {}, Round {}'.format( hparams.measurement_type, hparams.num_measurements, 
            noise_info, hparams.adaptive_round_count ))
        print('SAMPLED {} INSTANCES FROM POSTERIOR PROBABILITY with Avg_Recovery_Loss {}, MAP_Recovery_Loss {}, MAP_center_loss {}, MAP_corner_loss {}'.format(n_sample, 
            recovery_loss_sum, utils.get_l2_loss(original_img, best_keeper.get_best().reshape(-1)), center_rloss, corner_rloss))
        print('\t')
        sys.stdout.close()  
        sys.stdout = tmp   

        # return best_keeper.get_best(), var, best_keeper_z.get_best(), best_keeper.losses_val_best
        return best_keeper.get_best(), var, best_keeper.losses_val_best

    return estimator
