"""Inputs for MNIST dataset"""

import math
import numpy as np
import mnist_model_def
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow_datasets
import random


NUM_TEST_IMAGES = 10000


def get_random_test_subset(mnist, sample_size):
    """Get a small random subset of test images"""
    idxs = np.random.choice(NUM_TEST_IMAGES, sample_size)
    #images = [mnist.test.images[idx] for idx in idxs]
    images = [mnist[1][0][idx] for idx in idxs]
    images = {i: image for (i, image) in enumerate(images)}
    return images


def sample_generator_images(hparams):
    """Sample random images from the generator"""

    # Create the generator
    _, x_hat, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # Get a session
    sess = tf.Session()

    # Intialize and restore model parameters
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restorer = tf.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    images = {}
    counter = 0
    rounds = int(math.ceil(hparams.num_input_images/hparams.batch_size))
    for _ in range(rounds):
        images_mat = sess.run(x_hat)
        for (_, image) in enumerate(images_mat):
            if counter < hparams.num_input_images:
                images[counter] = image
                counter += 1

    # Reset TensorFlow graph
    sess.close()
    tf.reset_default_graph()

    return images


def model_input(hparams):
    """Create input tensors"""
    random.seed(hparams.seed_no)
    #mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    #mnist = tensorflow_datasets.load('./data/mnist')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = np.expand_dims(x_test, axis=-1)
    if hparams.num_input_images != 1:
        x_enumerate = x_test[:hparams.num_input_images]
    else:
        idx = random.randint(0,len(x_test))
        x_enumerate = x_test[idx:idx+1]
    x_enumerate = x_enumerate / 255.

    if hparams.input_type == 'full-input':
        images = {i: image.reshape(-1) for (i, image) in enumerate(x_enumerate)} #28,28,1
    elif hparams.input_type == 'random-test':
        images = get_random_test_subset(mnist, hparams.num_input_images)
    elif hparams.input_type == 'gen-span':
        images = sample_generator_images(hparams)
    else:
        raise NotImplementedError

    return images



