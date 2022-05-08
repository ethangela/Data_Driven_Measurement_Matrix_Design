def adaptive(n_sample_=50000, n_top_=784, num_mea_=10):
    #variance from training samples 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255. 
    x_train = np.expand_dims(x_train, axis=-1) #60000, 28, 28, 1
    x_test = np.expand_dims(x_test, axis=-1) #10000, 28, 28, 1

    #variance
    #np.random.seed(1008)
    perm = np.arange(x_train.shape[0])
    np.random.shuffle(perm)
    x_train = x_train[perm]
    x_mean, x2_mean = 0, 0
    # x_mean_lis, x2_mean_lis = [], []
    for i in range(n_sample_):
        sample = np.array(x_train[i])
        x_mean = x_mean + sample #(28,28,1)
        x2_mean = x2_mean + np.square(sample) #(28,28,1)
        
    x_mean = x_mean/n_sample_
    x2_mean = x2_mean/n_sample_
    var = x2_mean - np.square(x_mean) #(28,28,1)
    var_m = np.mean(var, axis=-1) #(28,28)
    def normalize(a):        
        return((a-np.min(a))/(np.max(a)-np.min(a)))

      
    def color_map(img, path):
        plt.figure(figsize=(8,5))
        plt.imshow(img)
        plt.colorbar()
        plt.savefig(path)

        
    def mask_generation(n_top, num_mea, noise=None):
        if n_top != 784:
            #find tops
            idx = var_m.ravel().argsort()[-n_top:][::-1]
            d1_idx = var_m.ravel().argsort()[-n_top:][::-1]
            d2idx = [(idx//28, idx%28) for idx in d1_idx]

            #gausian_block
            mask = np.zeros((28,28,1))
            for (i,j) in d2idx:
                mask[i,j,:] = 1

            # output
            color_map(mask, './uncertainty/mnist_var_top_{}_mask_color.jpg'.format(n_top))
            mask = mask.reshape((1,-1))
            np.save('./uncertainty/mnist_var_top_{}_mask.npy'.format(n_top), mask)
            
        else:
          
            #adjusted A
            mask = np.zeros((28,28))
            var_m_norm = normalize(var_m)
            var_m_norm = (var_m_norm+0.1)/(1+0.1)
            unit = 784. / np.sum(var_m_norm)
            for i in range(var_m.shape[0]):
                for j in range(var_m.shape[1]):
                    mask[i,j] = np.sqrt(var_m_norm[i,j] * unit)
            mask_reshape = np.reshape(mask, (-1))
            A = np.zeros((784,num_mea))
            for i in range(784):
                A[i,:] = mask_reshape[i] * np.random.randn(num_mea)
            
            # output
            color_map(var_m_norm, './uncertainty/mnist_propotional_var_color.jpg') 
            color_map(mask, './uncertainty/mnist_propotional_std.jpg')
            np.save('./uncertainty/mnist_propotional_std.npy', mask)
            np.save('./uncertainty/mnist_propotional_A_mea_{}.npy'.format(num_mea), A)
            print('mask building completed')
        
        
    mask_generation(n_top_, num_mea_)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pickle', type=str, default='nan')
    hparams = parser.parse_args()

    if hparams.pickle == 'nan':
        for num in [784]:
            for mea in [10,30,60,150]:
                adaptive(50000, num, mea)

