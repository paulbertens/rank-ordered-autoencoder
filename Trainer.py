import pickle
import threading
import time
from multiprocessing import Queue
import numpy as np
import DataLoader
from Visualizer import Visualizer
from RankOrderedAutoencoder import RankOrderNetwork


def data_loading(minibatch_size, data_iterator, shapeInput, exit_size):
    queue_train = Queue(maxsize=exit_size*10)
    queue_test = Queue(maxsize=exit_size*10)
    def start_loading():
        for e in range(exit_size):
            iterator_train = data_iterator(shapeInput, minibatch_size, shuffle=True, train=True)
            iterator_test = data_iterator(shapeInput, minibatch_size, shuffle=True, train=False)
            for new_input in iterator_train:
                while queue_train.full():
                    print('Queue full')
                    time.sleep(30)
                queue_train.put(new_input)
                new_input_test = iterator_test.next()
                queue_test.put(new_input_test)
        print('Exiting queue')

    t = threading.Thread(target=start_loading)
    t.daemon = True
    t.start()
    return queue_train, queue_test


def train(network, minibatch_size=5000, data_size=50000, n_epochs=60, shapeInput=(7, 7), data_iterator=DataLoader.iterate_cifar, load_previous=False):
    visualizer = Visualizer()
    errors_train = []
    errors_train_vector = []
    errors_test = []
    errors_test_vector = []
    sparsity_train = []
    sparsity_test = []
    currenti = 0
    if load_previous:
        try:
            state = pickle.load(open('network_state.pkl', 'rb'))
            network.weights = state[0]
            currenti = state[1] + 1
            errors_train = state[2]
            errors_train_vector = state[3]
            sparsity_train = state[4]
            errors_test = state[5]
            errors_test_vector = state[6]
            sparsity_test =  state[7]
            np.random.set_state(state[8])
            network.learning_rate = state[9]
            network.output_average = state[10]
            network.norm_limit = state[11]
            print('Reloading previous state')
        except:
            print('Starting from random weights')
    queue_train, queue_test = data_loading(minibatch_size, data_iterator, shapeInput, n_epochs-currenti)
    for i in range(currenti, n_epochs):
        print("----Epoch %i----" % i)
        total_error_train = 0
        total_error_train_vector = np.zeros_like(network.get_reconstruction_error_vector())
        total_sparsity_train = 0
        total_error_test = 0
        total_error_test_vector = np.zeros_like(network.get_reconstruction_error_vector())
        total_sparsity_test = 0
        t1 = time.time()
        j = 0
        for k in range(data_size/minibatch_size):
            new_input = queue_train.get()
            new_input_test = queue_test.get()
            for m in range(new_input.shape[0]):
                j += 1
                # train, update network
                network.update(new_input[m], train=True)
                # update statistics train
                total_error_train += network.get_reconstruction_error()
                total_error_train_vector += network.get_reconstruction_error_vector()
                total_sparsity_train += network.get_sparsity_error_term()
                # test, not modify weights
                network.update(new_input_test[m], train=False)
                # update statistics test
                total_error_test += network.get_reconstruction_error()
                total_error_test_vector += network.get_reconstruction_error_vector()
                total_sparsity_test += network.get_sparsity_error_term()

            print(i, j, total_error_train / j, total_sparsity_train / float(j))
            visualizer.saveNetworkPlots(network, i, calc_error_surface=False)
        if i % 1 == 0:  # measure only sometimes
            errors_train += [total_error_train / j]
            errors_train_vector += [total_error_train_vector / j]
            sparsity_train += [total_sparsity_train / float(j)]
            errors_test += [total_error_test / j]
            errors_test_vector += [total_error_test_vector / j]
            sparsity_test += [total_sparsity_test / float(j)]
            print(i, total_error_train / j, total_sparsity_train / float(j))
        print(time.time()-t1)
        if i % 1 == 0:
            pickle.dump((network.weights, i, errors_train, errors_train_vector, sparsity_train, errors_test,
                         errors_test_vector, sparsity_test, np.random.get_state(), network.learning_rate, network.output_average,
                         network.norm_limit), open('network_state.pkl', 'wb'))
            visualizer.saveFinalPlots(errors_train, errors_test, sparsity_train, sparsity_test, errors_train_vector, errors_test_vector, epoch=i)
            visualizer.saveNetworkPlots(network, i, calc_error_surface=True)
        if i > 0: # update learning rate
            previous_error = errors_train[i-1]
            if not errors_train[i]*1.01 < previous_error:
                network.learning_rate*=0.9
                print('New learning rate: ', network.learning_rate)

    print("Training END")

if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=2, suppress=True)
    network = RankOrderNetwork((7, 7, 3), (1, 13**2))
    train(network)