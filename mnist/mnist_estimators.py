"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mnist_model_def
from mnist_utils import save_image
from mnist_input import model_input
import utils
import sys
from skimage.io import imread, imsave
import os


def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            x_hat = utils.solve_lasso(A_val, y_val, hparams)
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def omp_estimator(hparams):
    """OMP estimator"""
    omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs=hparams.omp_k)
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            omp_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
            x_hat = omp_est.coef_
            x_hat = np.reshape(x_hat, [-1])
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def vae_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print (logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val))

                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator


def learned_estimator(hparams):

    sess = tf.Session()
    y_batch, x_hat_batch, restore_dict = mnist_model_def.end_to_end(hparams)
    restore_path = utils.get_A_restore_path(hparams)

    # Intialize and restore model parameters
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):  # pylint: disable = W0613
        """Function that returns the estimated image"""
        x_hat_batch_val = sess.run(x_hat_batch, feed_dict={y_batch: y_batch_val})
        return x_hat_batch_val

    return estimator


# added by Young in 20 Jul 2021
def vae_map_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

    # define all losses
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1)
    zp_loss = tf.reduce_sum(zp_loss_batch)
    y_loss_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_mean(y_loss_batch)

    # define total loss
    sigma = tf.placeholder(tf.float32, shape=[]) 
    annealed = hparams.annealed 

    mloss_weight = 0.5 * hparams.num_measurements / sigma**2
    zprior_weight = 1/(2 * hparams.zprior_sdev**2) 

    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * zp_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    learning_rate = tf.placeholder(tf.float32, shape=[]) 
    global_step = tf.Variable(0, trainable=False, name='global_step')
    opt = utils.get_optimizer(learning_rate, hparams) #tf.train.AdamOptimizer(learning_rate)
    update_op = opt.minimize(total_loss, var_list=[z_batch], global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, [z_batch], global_step)

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    # opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """added original image, 26 Jul"""
        xs_dict = model_input(hparams) #{0: img}
        original_img = xs_dict[0]
        noise_info = '_'.join(str(hparams.noise_std).split('.'))

        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op) #sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                
                if annealed:
                    factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1)) #factor = (final/ini) ** (1/(L-1))
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2) #lr(t) = ( ini/final * factor**(t/T) )**2
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T) #sigma = ini * factor**(j/T)
                    lr_value = hparams.learning_rate * lr_lambda(j) #lr = lr_ini * lr(j)
                else:
                    sigma_value = hparams.sigma_final
                    lr_value = hparams.learning_rate
                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})

                sess.run(update_op, feed_dict)
                   
            # MAP 
            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = sess.run([x_hat_batch, z_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_value, z_hat_batch_value, total_loss_batch_value)
            


        #log
        tmp = sys.stdout   
        sys.stdout = open(hparams.log_file_path, 'a') 
        print('Measurement {} Noise {} Mode {}'.format( hparams.num_measurements, noise_info, hparams.measurement_type ))
        print('Recovery_loss {}'.format( utils.get_l2_loss(original_img, best_keeper.get_best()[0].reshape(-1)) ))
        print('\t')
        sys.stdout.close()  
        sys.stdout = tmp  

        
        return best_keeper.get_best()[0]

    return estimator

def vae_langevin_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')
        
    # create noise placeholders for langevin
    noise_vars = tf.placeholder(tf.float32, shape=(hparams.batch_size, 20)) 

    # define all losses
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1)
    zp_loss = tf.reduce_sum(zp_loss_batch)
    y_loss_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1)
    y_loss = tf.reduce_mean(y_loss_batch)

    # define total loss
    sigma = tf.placeholder(tf.float32, shape=[]) 
    annealed = hparams.annealed 
    
    mloss_weight = 0.5 * hparams.num_measurements / (sigma**2)
    zprior_weight = 1/(2 * hparams.zprior_sdev**2) 
    
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * zp_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[]) #learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams) #tf.train.AdamOptimizer(learning_rate)
    update_op = opt.minimize(total_loss, var_list=[z_batch], global_step=global_step, name='update_op')
    init_op = tf.global_variables_initializer()
    opt_reinit_op = utils.get_opt_reinit_op(opt, [z_batch], global_step)
    noise_ops = z_batch.assign_add(noise_vars) 

    # Intialize and restore model parameters
    # opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        
        x_mean = np.zeros((1,784))
        x2_mean = np.zeros((1,784))
        n_sample = 0
        
        
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op) #sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):

                if annealed:
                    factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1)) #factor = (final/ini) ** (1/(L-1))
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2) #lr(t) = ( ini/final * factor**(t/T) )**2
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T) #sigma = ini * factor**(j/T)
                    lr_value = hparams.learning_rate * lr_lambda(j) #lr = lr_ini * lr(j)
                else:
                    sigma_value = hparams.sigma_final
                    lr_value = hparams.learning_rate
                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})

                _, total_loss_value, z_loss_value, y_loss_value = sess.run([update_op, total_loss, zp_loss, y_loss], feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} y_loss {} z_loss {}'
                print(logging_format.format(i, j, lr_value, total_loss_value, y_loss_value, z_loss_value))

                # added noise
                gradient_noise_weight = np.sqrt(2*lr_value/(1-hparams.momentum)) 
                feed_dict.update({noise_vars: gradient_noise_weight * np.random.randn(hparams.batch_size, 20)})   
                _ = sess.run(noise_ops, feed_dict) #checkjul21
                
                # posterior sampling
                if j > int(0.75*hparams.max_update_iter) and j % hparams.sample_frequency == 0: 
                    x_hat_batch_value = sess.run(x_hat_batch, feed_dict=feed_dict) #(batch_size,n_input) 
                    assert x_hat_batch_value.shape[0] == 1
                    x_mean += x_hat_batch_value
                    x2_mean += x_hat_batch_value**2
                    n_sample += 1
        
        # samples mean/var
        x_mean = x_mean/n_sample
        x2_mean = x2_mean/n_sample    
        var = x2_mean - (x_mean)**2 

        return x_mean, var #(batch_size=1,n_input), #(batch_size=1,n_input)

    return estimator



def vae_annealed_langevin_estimator(hparams):

    # Get a session
    sess = tf.Session()

    # Set up palceholders
    A = tf.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')
    z_batch_best = tf.placeholder(tf.float32, shape=(hparams.batch_size, 20), name='z_batch_best')

    # Create the generator
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch') #[1,n] * [n,m] = [1,m]
        
    # create noise placeholders for langevin
    noise_vars = tf.placeholder(tf.float32, shape=(hparams.batch_size, 20)) 

    # define all losses
    zp_loss_batch = tf.reduce_sum(z_batch**2, 1) #scalar
    zp_loss = tf.reduce_sum(zp_loss_batch)
    y_loss_batch = tf.reduce_mean((y_batch - y_hat_batch)**2, 1) #scalar
    y_loss = tf.reduce_mean(y_loss_batch)

    # define total loss
    sigma = tf.placeholder(tf.float32, shape=[]) 
    annealed = hparams.annealed 
    
    mloss_weight = hparams.num_measurements / (2 * sigma**2) 
    zprior_weight = 1/(2 * hparams.zprior_sdev**2) 
    
    total_loss_batch = mloss_weight * y_loss_batch + zprior_weight * zp_loss_batch
    total_loss = tf.reduce_sum(total_loss_batch)

    # Set up gradient descent
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, shape=[]) #learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams) #tf.train.AdamOptimizer(learning_rate)
    update_op = opt.minimize(total_loss, var_list=[z_batch], global_step=global_step, name='update_op')
    init_op = tf.global_variables_initializer()
    opt_reinit_op = utils.get_opt_reinit_op(opt, [z_batch], global_step)
    noise_ops = z_batch.assign_add(noise_vars) 
    z_assign_ops = z_batch.assign(z_batch_best)

    # Intialize and restore model parameters
    # opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)
    sess.run(init_op)
    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """added info, 26 Jul"""
        xs_dict = model_input(hparams) #{0: img}
        original_img = xs_dict[0]
        noise_info = '_'.join(str(hparams.noise_std).split('.'))
        
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        
        x_mean = np.zeros((1,784))
        x2_mean = np.zeros((1,784))
        n_sample = 0
        recovery_loss_sum = 0
        
        
        for restart in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op) #sess.run(opt_reinit_op)

            # MAP first
            if hparams.map_before_posterior:
                for j in range(hparams.max_update_iter):
                    
                    if annealed:
                        factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1)) #factor = (final/ini) ** (1/(L-1))
                        lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2) #lr(t) = ( ini/final * factor**(t/T) )**2
                        sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T) #sigma = ini * factor**(j/T)
                        lr_value = hparams.learning_rate * lr_lambda(j) #lr = lr_ini * lr(j)
                    else:
                        sigma_value = hparams.sigma_final
                        lr_value = hparams.learning_rate
                    feed_dict.update({learning_rate: lr_value, sigma: sigma_value})

                    _ = sess.run(update_op, feed_dict)
            
            # posterior sampling
            posterior_best_keeper = utils.BestKeeper(hparams)
            std = np.geomspace(hparams.start_std, hparams.end_std, hparams.L) #jul26para
            for l in range(hparams.L):
                alpha_l = hparams.step * (std[l]**2/std[-1]**2) 

                for t in range(hparams.T):
                    lr_value = alpha_l
                    sigma_value = std[l]
                    feed_dict.update({learning_rate: lr_value, sigma: sigma_value})

                    _, total_loss_value, z_loss_value, y_loss_value = sess.run([update_op, total_loss, zp_loss, y_loss], feed_dict)
                    # logging_format = 'restart {} L {} T {} lr {} sigma {} total_loss {} y_loss {} z_loss {}'
                    # print(logging_format.format(restart, l, t, lr_value, sigma_value, total_loss_value, y_loss_value, z_loss_value))

                    # added noise
                    gradient_noise_weight = np.sqrt(2*lr_value) 
                    feed_dict.update({noise_vars: gradient_noise_weight * np.random.randn(hparams.batch_size, 20)})   
                    _ = sess.run(noise_ops, feed_dict) #checkjul21
                
                    # sampling
                    sample_count = (l+1)*hparams.T + (t+1)
                    if sample_count > int(0.5*(hparams.L*hparams.T)) and sample_count % hparams.sample_frequency == 0: 
                        x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = sess.run([x_hat_batch, z_batch, total_loss_batch], feed_dict=feed_dict) #(batch_size,n_input)

                        # generate variance
                        assert x_hat_batch_value.shape[0] == 1
                        x_mean += x_hat_batch_value
                        x2_mean += x_hat_batch_value**2
                        n_sample += 1
                        recovery_loss = utils.get_l2_loss(original_img, x_hat_batch_value.reshape(-1))
                        recovery_loss_sum += recovery_loss

                        # sampled images 
                        if hparams.generate_posterior_sample:
                            noise_info = '_'.join(str(hparams.noise_std).split('.'))
                            img = x_hat_batch_value.reshape(hparams.image_shape) 
                            img = img * 255.
                            save_dir = '../src/{}/adaptive/noise_{}/measurement_{}/Round_{}/Restart_{}'.format(hparams.seed_no, noise_info, hparams.num_measurements, hparams.adaptive_round_count, restart)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            imsave( os.path.join(save_dir, 'Sample_{}.jpg'.format(sample_count)), img.astype(np.uint8))

                        # select best among all posterior samples
                        posterior_best_keeper.report(x_hat_batch_value, z_hat_batch_value, total_loss_batch_value, sample_count)
            
                   
            # MAP
            feed_dict.update({z_batch_best: posterior_best_keeper.get_best()[1]})
            _ = sess.run(z_assign_ops, feed_dict)

            # tmp log
            tmp = sys.stdout   
            sys.stdout = open(hparams.log_file_path, 'a') 
            print('Seed {}, Measurement {}, Noise {}, Round {}, Restart {}, Best Sample {}'.format(
                hparams.seed_no, hparams.num_measurements, noise_info, hparams.adaptive_round_count, restart, posterior_best_keeper.get_best()[2]))
            print('\t')
            sys.stdout.close()  
            sys.stdout = tmp   

            # gradient decent
            for j in range(hparams.max_update_iter):
                
                if annealed:
                    factor = np.power(hparams.sigma_final / hparams.sigma_init, 1/(hparams.L-1)) #factor = (final/ini) ** (1/(L-1))
                    lr_lambda = lambda t: (hparams.sigma_init * np.power(factor, t//hparams.T))**2 / (hparams.sigma_final **2) #lr(t) = ( ini/final * factor**(t/T) )**2
                    sigma_value = hparams.sigma_init * np.power(factor, j//hparams.T) #sigma = ini * factor**(j/T)
                    lr_value = hparams.learning_rate * lr_lambda(j) #lr = lr_ini * lr(j)
                else:
                    sigma_value = hparams.sigma_final
                    lr_value = hparams.learning_rate
                feed_dict.update({learning_rate: lr_value, sigma: sigma_value})

                sess.run(update_op, feed_dict)

            x_hat_batch_value, z_hat_batch_value, total_loss_batch_value = sess.run([x_hat_batch, z_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_value, z_hat_batch_value, total_loss_batch_value, restart)
                   

        # samples mean/var
        recovery_loss_sum = recovery_loss_sum/n_sample
        x_mean = x_mean/n_sample
        x2_mean = x2_mean/n_sample    
        var = x2_mean - (x_mean)**2 
        
        # return x_mean, var #(batch_size=1,n_input), #(batch_size=1,n_input)
        return best_keeper.get_best()[0], var, best_keeper.get_best()[2]


    return estimator
