# Get input

# from tensorflow.examples.tutorials.mnist import input_data

# def mnist_data_iteratior():
#     mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#     def iterator(hparams, num_batches):
#         for _ in range(num_batches):
#             yield mnist.train.next_batch(hparams.batch_size)
#     return iterator


#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class DataSet(object):

    def __init__(self,
                reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        #inject data
        mnist = tf.keras.datasets.mnist
        (images, l_train), (x_test, l_test) = mnist.load_data()
        images = np.expand_dims(images, axis=-1)
        labels = np.zeros((l_train.shape[0], l_train.max()+1), dtype=np.float32)
        labels[np.arange(l_train.shape[0]), l_train] = 1


        #number of training samples
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0] #60000

        # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def get_mnist():
    return DataSet()


def mnist_data_iteratior():
    mnist = get_mnist()
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.next_batch(hparams.batch_size)
    return iterator
    
