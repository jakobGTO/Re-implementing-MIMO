from mimo import *
import numpy as np

def fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps):
    res_list = []
    for M in range(1,10):
        input_shape=(M,32,32,3)

        resnet = ResNet20_10()
        resnet_architecture = resnet.build(input_shape, K, M)

        model = MIMO(resnet_architecture, num_batch_reps, M)
        model.compile(optimizer=optimizer)
        model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs, use_multiprocessing=True, workers=4096, max_queue_size=512, validation_freq=5, batch_size=None, shuffle=False)
        results = model.evaluate(test_dataset, batch_size=None)
        
        res_list.append(results)
    return np.array(res_list)
    

if __name__ == '__main__':
    print(tf.__version__)
    tf.config.run_functions_eagerly(True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Data handling
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    '''    x_train = x_train[:50,:,:,:]
    y_train = y_train[:50,:]
    x_test = x_test[:50,:,:,:]
    y_test = y_test[:50,:]
    '''
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64

    # During training send in randomly sampled images
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # During testing we send the same image to each node
    test_dataset = test_dataset.batch(batch_size)
    
    # Set optimizer, adapted from https://keras.io/api/optimizers/ and with
    # hyperparameters described in Annex B of the original paper.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=100000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    # Exp 1: Cifar 10
    K = 10
    num_batch_reps = 0
    num_epochs = 20

    fig5_res = fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps)
    np.save('experiment_results/fig5_experiment_results_cifar10', fig5_res)

    # Exp 2: Cifar 100
    K = 100
    num_batch_reps = 0
    num_epochs = 20

    #fig5_res = fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps)
    #np.save('experiment_results/fig5_experiment_results_cifar100', fig5_res)

