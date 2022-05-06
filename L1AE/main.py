from __future__ import division
import os
import numpy as np
from time import time
from scipy import sparse
from sklearn.preprocessing import normalize
import math
import pandas as pd 
import matplotlib.pyplot as plt
from gurobipy import *
from utils import l1_min_avg_err
from model import L1AE

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

SEED = 43
np.random.seed(SEED) 


# tf parameters defination
flags = tf.app.flags
flags.DEFINE_integer("input_dim", 1000, "Input dimension [1000]")
flags.DEFINE_integer("powerlaw_exp", 1, "Exponent in the power law [1]")
flags.DEFINE_integer("powerlaw_bias", 1, "Bias in the power law [1]")
flags.DEFINE_integer("avg_sparsity", 10, "Average nonzeros [10]")
flags.DEFINE_integer("num_samples", 10000, "Number of total samples [10000]")
flags.DEFINE_integer("emb_dim", 10, "Number of measurements [10]")
flags.DEFINE_integer("decoder_num_steps", 10,
                     "Depth of the decoder network [10]")
flags.DEFINE_integer("batch_size", 128, "Batch size [128]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", int(2e4),
                     "Maximum number of training epochs [1e3]")
flags.DEFINE_integer("display_interval", 100,
                     "Print the training info every [100] epochs")
flags.DEFINE_integer("validation_interval", 10,
                     "Compute validation loss every [10] epochs")
flags.DEFINE_integer("max_steps_not_improve", 5,
                     "stop training when the validation loss \
                      does not improve for [5] validation_intervals")
flags.DEFINE_string("checkpoint_dir", "ckpts/synthetic_powerlaw/",
                    "Directory name to save the checkpoints \
                    [ckpts/synthetic_powerlaw/]")
flags.DEFINE_integer("num_random_dataset", 10,
                     "Number of random datasets [10]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments [1]")
FLAGS = flags.FLAGS


# model parameters
# input_dim = FLAGS.input_dim
# powerlaw_exp = FLAGS.powerlaw_exp
# powerlaw_bias = FLAGS.powerlaw_bias
# avg_sparsity = FLAGS.avg_sparsity
# num_samples = FLAGS.num_samples
# emb_dim = FLAGS.emb_dim
decoder_num_steps = FLAGS.decoder_num_steps

# training parameters
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_training_epochs = FLAGS.max_training_epochs
display_interval = FLAGS.display_interval
validation_interval = FLAGS.validation_interval
max_steps_not_improve = FLAGS.max_steps_not_improve

# checkpoint directory
checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# number of experiments
num_random_dataset = FLAGS.num_random_dataset
num_experiment = FLAGS.num_experiment


def synthetic_power_law_data(input_dim, powerlaw_exp, powerlaw_bias,
                             avg_sparsity, num_samples,
                             train_ratio=0.6, valid_ratio=0.2):
    """
    Generate synthetic sparse dataset with average sparsity = avg_sparsity.
    For vector x, its i-th entry is nonzero with a probability proportional to
    1/(i+powerlaw_bias)^(powerlaw_exp), where i is from 0 to input_dim-1.
    """
    probs = 1.0/np.power(np.arange(input_dim)+powerlaw_bias, powerlaw_exp)
    probs = probs/np.sum(probs)*avg_sparsity
    probs = np.minimum(probs, np.ones_like(probs))
    X = np.zeros((num_samples, input_dim))
    for i in range(num_samples):
        X[i, :] = np.random.binomial(1, probs)
    X = np.array(X)
    shuffle = np.random.permutation(X.shape[0])
    X = X[shuffle, :]
    # convert it to csr_matrix format
    X = sparse.csr_matrix(X)
    # set the nonzeros to be unifrom in [0,1]
    X.data = np.random.uniform(0.0, 1.0, len(X.data))
    # make each sample unit norm
    normalize(X, norm='l2', axis=1, copy=False, return_norm=False) ###TODO: to check if this works
    # split into train/valid/test
    train_size = int(train_ratio*num_samples)
    valid_size = int(valid_ratio*num_samples)
    X_train = X[:train_size, :]
    X_valid = X[train_size:(train_size+valid_size), :]
    X_test = X[(train_size+valid_size):, :]
    return X_train, X_valid, X_test



def evaluate_encoder_matrix(A, Y, X):
    """
    Run l1-min using the sensing matrix A.
    Args:
        A: 2-D array, shape=(emb_dim, input_dim)
        Y: 2-D array, shape=(num_sample, emb_dim)
        X: 2-D csr_matrix, shape=(num_sample, input_dim)
    """
    # l1ae_l1_err, l1ae_l1_exact, _ = l1_min_avg_err(A, Y,
    #                                                X, use_pos=False)
    avg_err, exact_ratio, solved_ratio = l1_min_avg_err(
                                                       A, Y,
                                                       X, use_pos=True)
    
    # res = {}
    # # res['l1ae_l1_err'] = l1ae_l1_err
    # # res['l1ae_l1_exact'] = l1ae_l1_exact
    # res['l1ae_l1_err_pos'] = l1ae_l1_err_pos
    # res['l1ae_l1_exact_pos'] = l1ae_l1_exact_pos
    return avg_err, exact_ratio, solved_ratio



def propotional_matrix(input_dim, emb_dim, powerlaw_exp, powerlaw_bias, avg_sparsity):
    """
    Generate measurement matrix with shape=(input_dim, emb_dim)
    """
    probs = 1.0/np.power(np.arange(input_dim)+powerlaw_bias, powerlaw_exp)
    probs = probs/np.sum(probs)*avg_sparsity
    probs = np.minimum(probs, np.ones_like(probs))
    
    X = np.zeros((input_dim, emb_dim))
    enery_unit = (1/emb_dim) * input_dim / np.sum(probs)
    for i in range(input_dim):
        var_i = probs[i] * enery_unit
        X[i, :] = np.sqrt(var_i) * np.random.randn(emb_dim) 
    return X


def l1_min_hat(A, Y, true_X, use_pos=False, eps=1e-10):
    """
    Run l1_min for each sample, and compute the RMSE.
    true_X is a 2D csr_matrix with shape=(num_sample, input_dim).
    """
    
    def l1_min_pos(A, y, true_x):
        """
        Solve min_x sum_ix_i s.t. Ax=y, x_i>= 0 and compute err = ||x-true_x||_2
        """
        emb_dim, input_dim = A.shape
        model = Model()
        model.params.outputflag = 0  # disable solver output
        x = []
        for i in range(input_dim):
            # The lower bound lb=0.0 indicates that x>=0
            x.append(model.addVar(lb=0.0, ub=GRB.INFINITY, obj=1))
        model.update()
        # add equality constraints
        for i in range(emb_dim):
            coeff = A[i, :]
            expr = LinExpr(coeff, x)
            model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=y[i])
        # optimize the model and obtain the results
        model.optimize()
        res = []
        for v in model.getVars():
            res.append(v.x)
        return res[:input_dim], np.linalg.norm(res[:input_dim]-true_x)

    num_sample = Y.shape[0]
    num_exact = 0  # number of samples that are exactly recovered
    num_solved = num_sample  # number of samples that successfully runs l1_min
    err = 0
    bad_sample = []
    x_sample = []
    x_hat_sample = []
    for i in range(num_sample):
        y = Y[i, :].reshape(-1,)
        x = true_X[i, :].toarray().reshape(-1,)
        try:
            x_hat, temp_err = l1_min_pos(A, y, x)
            x_hat_sample.append(x_hat)
            x_sample.append(x)
            if temp_err < eps:
                num_exact += 1
            err += temp_err**2  # squared error
        except Exception:
            num_solved -= 1
    
    return np.array(x_hat_sample), np.array(x_sample), err, num_solved




def get_pixl_l2_loss(image1, image2):
    assert image1.shape == image2.shape
    error = np.mean((image1 - image2)**2, axis=0) #(input_dim,)
    return error


def matrix_builder_data_driven(input_dim, emb_dim, top_idx, bot_idx, lambd, round, matrix=None):
    if round == 0:
        std = np.ones((input_dim,)) #(n,)
    else:
        std = matrix

    for i in range(input_dim):
        if i in top_idx:
            std[i] = std[i] * math.exp(-lambd)
        elif i in bot_idx:
            std[i] = std[i] * math.exp(lambd)
    std = std / np.sum(std) * (input_dim/emb_dim)
    
    A = np.zeros((input_dim, emb_dim))
    for i in range(input_dim):
        A[i,:] = np.sqrt(std[i]) * np.random.randn(emb_dim)      
    
    return std, A



class data_driven:
    def __init__(self, input_dim, emb_dim, round, top_percent, lamb, lamb_decay_rate, noise, random_data):
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.round = round
        self.top_percent = top_percent
        self.lamb = lamb
        self.lamb_decay_rate = lamb_decay_rate
        self.npy_dir = f'./data_driven/{random_data}/{noise}/mea_{self.emb_dim}'
        if not os.path.exists(self.npy_dir):
            os.makedirs(self.npy_dir)


    def train(self, X_train):
        train_length = X_train.shape[0] #num_sample
        batch_length = int(train_length/self.round)

        for adaptive_round_count in range(self.round):
            print(f'round {adaptive_round_count} started ...')
            if adaptive_round_count == 0:
                G = np.random.randn(self.input_dim, self.emb_dim) / np.sqrt(self.emb_dim)
            else:
                G = np.load( os.path.join(self.npy_dir, 'Measurement_from_ROUND_{}.npy'.format(adaptive_round_count-1)) )
            
            X_batch = X_train[batch_length*adaptive_round_count:batch_length*(adaptive_round_count+1),:]
            Y_batch = X_batch.dot(G)  
            
            X_hat_batch, X_batch, avg_err, num_solved = l1_min_hat(np.transpose(G), Y_batch, X_batch)
            print(f'traing error: {avg_err}, #not working sample: {batch_length-num_solved}')
            
            l2_pixl_loss_matrix = get_pixl_l2_loss(X_hat_batch, X_batch)
            np.save( os.path.join(self.npy_dir, 'error_from_ROUND_{}.npy'.format(adaptive_round_count)), l2_pixl_loss_matrix ) 
            n_top = int(self.input_dim*self.top_percent)
            top_idx = l2_pixl_loss_matrix.ravel().argsort()[:n_top]
            print('top idx check', top_idx)
            bot_idx = l2_pixl_loss_matrix.ravel().argsort()[-n_top:]
            print('bot idx check', bot_idx)

            lamb = self.lamb * (self.lamb_decay_rate ** adaptive_round_count)
            if adaptive_round_count == 0:
                std_out, A = matrix_builder_data_driven(self.input_dim, self.emb_dim, top_idx, bot_idx, lamb, adaptive_round_count, matrix=None)
            else:
                std_in = np.load( os.path.join(self.npy_dir, 'energy_map_from_ROUND_{}.npy'.format(adaptive_round_count-1)) )
                std_out, A = matrix_builder_data_driven(self.input_dim, self.emb_dim, top_idx, bot_idx, lamb, adaptive_round_count, matrix=std_in)
            np.save( os.path.join(self.npy_dir, 'Measurement_from_ROUND_{}.npy'.format(adaptive_round_count)), A) 
            np.save( os.path.join(self.npy_dir, 'energy_map_from_ROUND_{}.npy'.format(adaptive_round_count)), std_out) 
            print(f'round {adaptive_round_count} completed')
            


    def weight_generator(self):
        return np.load( os.path.join(self.npy_dir, 'Measurement_from_ROUND_{}.npy'.format(self.round-1)) )



if __name__ == '__main__':
    pickle_file_path = './result_l1ae.pkl'

    
    
    for data_id in range(num_random_dataset):
        print("---Dataset: {}---".format(data_id))
        
        t0 = time()
        X_train, X_valid, X_test = synthetic_power_law_data(input_dim=1000, powerlaw_exp=1, powerlaw_bias=1, avg_sparsity=10, num_samples=10000)
        t1 = time()
        print(f"data building takes {t1-t0} secs")

        for noise_level in [0.0]:#, 0.1, 0.2, 0.5, 1.0]:
            noise_info = '_'.join(str(noise_level).split('.'))
            for mea_dim in [20,40,60,80,100]:
                
                # L1AE
                t0 = time()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                sparse_AE = L1AE(sess, 1000, mea_dim, decoder_num_steps)

                print("Start training......")
                sparse_AE.train(X_train, X_valid, batch_size, learning_rate, max_training_epochs, display_interval, validation_interval, max_steps_not_improve)
                print("Start evaluating......")
                test_sq_loss = sparse_AE.inference(X_test, batch_size)
                sqrt_test_sq_loss = np.sqrt(test_sq_loss)
                print(f'RMSE on test data via decoder: {sqrt_test_sq_loss}')
                
                try:
                    G = sparse_AE.sess.run(sparse_AE.encoder_weight)
                    npy_dir = f'./trained/{data_id}/{noise_info}/mea_{mea_dim}'
                    if not os.path.exists(npy_dir):
                        os.makedirs(npy_dir)
                    np.save( os.path.join(npy_dir, 'measurement_matrix.npy'), G )
                    Y = X_test.dot(G) + noise_level * np.random.randn(X_test.shape[0], mea_dim) / np.sqrt(mea_dim)
                    avg_err, exact_ratio, solved_ratio = l1_min_avg_err(np.transpose(G), Y, X_test, use_pos=True)
                    print('average error', avg_err)
                    print('exact ratio', exact_ratio)
                    print('solved ratio', solved_ratio)
                    t1 = time()
                    print(f"L1-min takes {t1-t0} secs")
                except Exception:
                    avg_err = 'NaN'

                if not os.path.exists(pickle_file_path):
                    d = {'measurement_type':['trained'], 'measurement_num':[mea_dim], 'MSE_loss':[avg_err], 'evaluate_loss':[sqrt_test_sq_loss], 'noise':[noise_info], 'dataset':[data_id]}
                    df = pd.DataFrame(data=d)
                    df.to_pickle(pickle_file_path)
                else:
                    d = {'measurement_type':'trained', 'measurement_num':mea_dim, 'MSE_loss':avg_err, 'evaluate_loss':sqrt_test_sq_loss, 'noise':noise_info, 'dataset':data_id}
                    df = pd.read_pickle(pickle_file_path)
                    df = df.append(d, ignore_index=True)
                    df.to_pickle(pickle_file_path)

                
                
                # gaussian
                t0 = time()
                G = np.random.randn(1000, mea_dim) / np.sqrt(mea_dim)
                Y = X_test.dot(G) + noise_level * np.random.randn(X_test.shape[0], mea_dim) / np.sqrt(mea_dim)
                avg_err, exact_ratio, solved_ratio = l1_min_avg_err(np.transpose(G), Y, X_test, use_pos=True)
                print('average error', avg_err)
                print('exact ratio', exact_ratio)
                print('solved ratio', solved_ratio)
                t1 = time()
                print(f"L1-min takes {t1-t0} secs")

                if not os.path.exists(pickle_file_path):
                    d = {'measurement_type':['gaussian'], 'measurement_num':[mea_dim], 'MSE_loss':[avg_err], 'noise':[noise_info], 'dataset':[data_id]}
                    df = pd.DataFrame(data=d)
                    df.to_pickle(pickle_file_path)
                else:
                    d = {'measurement_type':'gaussian', 'measurement_num':mea_dim, 'MSE_loss':avg_err, 'noise':noise_info, 'dataset':data_id}
                    df = pd.read_pickle(pickle_file_path)
                    df = df.append(d, ignore_index=True)
                    df.to_pickle(pickle_file_path)



                #propotional
                t0 = time()
                G = propotional_matrix(input_dim=1000, emb_dim=mea_dim, powerlaw_exp=1, powerlaw_bias=1, avg_sparsity=10)
                Y = X_test.dot(G) + noise_level * np.random.randn(X_test.shape[0], mea_dim) / np.sqrt(mea_dim)
                avg_err, exact_ratio, solved_ratio = l1_min_avg_err(np.transpose(G), Y, X_test, use_pos=True)
                print('average error', avg_err)
                print('exact ratio', exact_ratio)
                print('solved ratio', solved_ratio)
                t1 = time()
                print(f"L1-min of propotional G takes {t1-t0} secs")

                if not os.path.exists(pickle_file_path):
                    d = {'measurement_type':['propotional'], 'measurement_num':[mea_dim], 'MSE_loss':[avg_err], 'noise':[noise_info], 'dataset':[data_id]}
                    df = pd.DataFrame(data=d)
                    df.to_pickle(pickle_file_path)
                else:
                    d = {'measurement_type':'propotional', 'measurement_num':mea_dim, 'MSE_loss':avg_err, 'noise':noise_info, 'dataset':data_id}
                    df = pd.read_pickle(pickle_file_path)
                    df = df.append(d, ignore_index=True)
                    df.to_pickle(pickle_file_path)



                #data driven
                t0 = time()
                driven = data_driven(input_dim=1000, emb_dim=mea_dim, round=20, top_percent=0.33, lamb=0.1, lamb_decay_rate=0.95, noise=noise_info, random_data=data_id) 
                driven.train(X_train)
                t1 = time()
                print(f"data driven training takes {t1-t0} secs")
                G = driven.weight_generator()
                Y = X_test.dot(G) + noise_level * np.random.randn(X_test.shape[0], mea_dim) / np.sqrt(mea_dim)
                avg_err, exact_ratio, solved_ratio = l1_min_avg_err(np.transpose(G), Y, X_test, use_pos=True)
                print('average error', avg_err)
                print('exact ratio', exact_ratio)
                print('solved ratio', solved_ratio)
                t2 = time()
                print(f"L1-min of data driven G takes {t2-t1} secs")

                if not os.path.exists(pickle_file_path):
                    d = {'measurement_type':['data_driven'], 'measurement_num':[mea_dim], 'MSE_loss':[avg_err], 'noise':[noise_info], 'dataset':[data_id]}
                    df = pd.DataFrame(data=d)
                    df.to_pickle(pickle_file_path)
                else:
                    d = {'measurement_type':'data_driven', 'measurement_num':mea_dim, 'MSE_loss':avg_err, 'noise':noise_info, 'dataset':data_id}
                    df = pd.read_pickle(pickle_file_path)
                    df = df.append(d, ignore_index=True)
                    df.to_pickle(pickle_file_path)



        







    
    
