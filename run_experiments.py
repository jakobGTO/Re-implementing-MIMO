from mimo import *
import numpy as np
import json

def fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps):
    res_list = []
    for M in range(1,6):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=100000, decay_rate=0.1)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True)

        input_shape=(M,32,32,3)

        resnet = ResNet20_10()
        resnet_architecture = resnet.build(input_shape, K, M)

        model = MIMO(resnet_architecture, num_batch_reps, M)
        model.compile(optimizer=optimizer)
        model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs, use_multiprocessing=True, workers=4096, max_queue_size=512, validation_freq=5, batch_size=None, shuffle=False)
        results = model.evaluate(test_dataset, batch_size=None)

        res_list.append(results)
    return np.array(res_list)
    
def fig6_exp(num_epochs):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    K = 10

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dict = {}
    res_list = []
    for num_batch_reps in range(1,4):
        for bs, lr, M in zip([28,30,32],[0.1,0.01,0.005],[4,3,2]):
            print()
            print('CURRENT SETTINGS:  ', 'bs:', bs, 'lr: ', lr, 'M:', M, 'num_batch_reps',num_batch_reps)
            print()
            # During training send in randomly sampled images
            trainx = train_dataset.shuffle(buffer_size=1024).batch(bs)
            # During testing we send the same image to each node
            testx = test_dataset.batch(bs)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=100000, decay_rate=0.1)
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_schedule, momentum=0.9, nesterov=True)

            input_shape=(M,32,32,3)

            resnet = ResNet20_10()
            resnet_architecture = resnet.build(input_shape, K, M)

            model = MIMO(resnet_architecture, num_batch_reps, M)
            model.compile(optimizer=optimizer)
            model.fit(trainx, validation_data=testx, epochs=num_epochs, use_multiprocessing=True, workers=4096, max_queue_size=512, validation_freq=5, batch_size=None, shuffle=False)
            results = model.evaluate(testx, batch_size=None)

            dict['bs'], dict['lr'], dict['M'], dict['num_batch_reps'], dict['results'] = bs, lr, M, num_batch_reps, results
            res_list.append(dict)
            dict = {}

    return res_list        

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64

    # During training send in randomly sampled images
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # During testing we send the same image to each node
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset

def load_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64

    # During training send in randomly sampled images
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # During testing we send the same image to each node
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset

if __name__ == '__main__':
    print(tf.__version__)
    tf.config.run_functions_eagerly(True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Exp 1: Cifar 10
    train_dataset, test_dataset = load_cifar10()

    K = 10
    num_batch_reps = 0
    num_epochs = 30

    fig5_res = fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps)
    np.save('experiment_results/fig5_experiment_results_cifar10', fig5_res)
    
    # Exp 2: Cifar 100

    train_dataset, test_dataset = load_cifar100()

    K = 100
    num_epochs = 20
    num_batch_reps = 0

    fig5_res = fig5_exp(train_dataset, test_dataset, K, batch_size, num_epochs, num_batch_reps)
    np.save('experiment_results/fig5_experiment_results_cifar100', fig5_res)
    
    # Exp 3: 
    num_epochs = 10

    fig6_res = fig6_exp(num_epochs)
    output_file = open('experiment_results/fig6_dict.json', 'w', encoding='utf-8')
    for dic in fig6_res:
        json.dump(dic, output_file) 
        output_file.write("\n")

