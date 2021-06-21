"""Compressed sensing main script"""
# pylint: disable=C0301,C0103,C0111

from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
from skimage.io import imread, imsave
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


def main(hparams):

    ##-- compressed sensing --##
    # Set up some stuff accoring to hparams
    hparams.n_input = np.prod(hparams.image_shape)
    utils.set_num_measurements(hparams)
    utils.print_hparams(hparams)

    # get inputs
    xs_dict = model_input(hparams) #{0: img}
    img = xs_dict[0].reshape(hparams.image_shape)
    img = (img+1.)/2.
    img = img * 255.
    imsave('in_put.jpg', img.astype(np.uint8))
    if hparams.measurement_type == 'inpaint':
        mask = np.load('/home/sunyang/csgm/src/mask.npy')
        mask_img = img * mask
        imsave('in_put_mask.jpg', mask_img.astype(np.uint8))
    print('input image(s) saved')
        

    estimators = utils.get_estimators(hparams) #{dcgan_estmator} #############################
    utils.setup_checkpointing(hparams) #./estimate ...
    measurement_losses, l2_losses = utils.load_checkpoints(hparams) #{'dcgan':{}}, {'dcgan':{}} #############################

    x_hats_dict = {model_type : {} for model_type in hparams.model_types} #{'dcgan':{}}
    x_batch_dict = {}
    for key, x in xs_dict.items(): #{0: img}
        if not hparams.not_lazy:
            # If lazy, first check if the image has already been
            # saved before by *all* estimators. If yes, then skip this image.
            save_paths = utils.get_save_paths(hparams, key)
            is_saved = all([os.path.isfile(save_path) for save_path in save_paths.values()])
            if is_saved:
                continue

        x_batch_dict[key] = x #{0: img}
        if len(x_batch_dict) < hparams.batch_size:
            continue

        # Reshape input
        x_batch_list = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #[img]
        x_batch = np.concatenate(x_batch_list) #img, shape (1, 12288)

        # Construct noise and measurements
        A = utils.get_A(hparams) #(n_input, n_measurement)
        print('A shape check {}'.format(A.shape))
        noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_measurements)
        if hparams.measurement_type == 'project':
            y_batch = x_batch + noise_batch
        else:
            y_batch = np.matmul(x_batch, A) + noise_batch
        print('Y shape check {}'.format(y_batch.shape))

        # Construct estimates using each estimator
        for model_type in hparams.model_types: #dcgan
            estimator = estimators[model_type] #dcgan
            x_hat_batch = estimator(A, y_batch, hparams)

            for i, key in enumerate(x_batch_dict.keys()):
                x = xs_dict[key]
                y = y_batch[i]
                x_hat = x_hat_batch[i]

                # Save the estimate
                x_hats_dict[model_type][key] = x_hat

                # Compute and store measurement and l2 loss
                measurement_losses[model_type][key] = utils.get_measurement_loss(x_hat, A, y)
                l2_losses[model_type][key] = utils.get_l2_loss(x_hat, x)
                print(measurement_losses['dcgan'][0])
                print(l2_losses['dcgan'][0])

        print('Processed upto image {0} / {1}'.format(key+1, len(xs_dict)))

        x_batch_dict = {}

    print(x_hats_dict['dcgan'][0].shape)
    img = x_hats_dict['dcgan'][0].reshape(hparams.image_shape)
    img = (img+1.)/2.
    img = img * 255.
    imsave('out_put.jpg', img.astype(np.uint8))
    

    ##-- mcmc --##


















    # # Final checkpoint
    # if hparams.save_images:
    #     print('saving images')
    #     utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
    #     print ('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))

    # if hparams.print_stats:
    #     for model_type in hparams.model_types:
    #         print (model_type)
    #         #mean_m_loss = np.mean(measurement_losses[model_type].values())
    #         #mean_l2_loss = np.mean(l2_losses[model_type].values())
    #         mean_m_loss = measurement_losses[model_type].values()
    #         mean_l2_loss = l2_losses[model_type].values()
    #         print ('mean measurement loss = {0}'.format(mean_m_loss))
    #         print ('mean l2 loss = {0}'.format(mean_l2_loss))

    # if hparams.image_matrix > 0:
    #     utils.image_matrix(xs_dict, x_hats_dict, view_image, hparams)

    # # Warn the user that some things were not processsed
    # if len(x_batch_dict) > 0:
    #     print ('\nDid NOT process last {} images because they did not fill up the last batch.'.format(len(x_batch_dict)))
    #     print ('Consider rerunning lazily with a smaller batch size.')


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='random_test', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='../../GANPriors-master/celeba/data/test/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=10, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=64, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')

    # Measurement type specific hparams
    PARSER.add_argument('--num-measurements', type=int, default=200, help='number of gaussian measurements')
    PARSER.add_argument('--inpaint-size', type=int, default=1, help='size of block to inpaint')
    PARSER.add_argument('--superres-factor', type=int, default=2, help='how downsampled is the image')

    # Model
    PARSER.add_argument('--model-types', type=str, nargs='+', default=None, help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=0.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.0, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='momentum', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
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
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError

    main(HPARAMS)
