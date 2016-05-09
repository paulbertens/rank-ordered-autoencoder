import matplotlib
matplotlib.rcParams.update({'font.size': 16})
#matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'Paul'


class Visualizer():

    def __init__(self):
        pass

    def reshapeWeights(self, weights, normalize=True, modifier=None):
        # reshape the weights matrix to a grid for visualization
        n_rows = int(np.sqrt(weights.shape[1]))
        n_cols = int(np.sqrt(weights.shape[1]))
        kernel_size = int(np.sqrt(weights.shape[0]/3))
        weights_grid = np.zeros((int((np.sqrt(weights.shape[0]/3)+1)*n_rows), int((np.sqrt(weights.shape[0]/3)+1)*n_cols), 3), dtype=np.float32)
        for i in range(weights_grid.shape[0]/(kernel_size+1)):
            for j in range(weights_grid.shape[1]/(kernel_size+1)):
                index = i * (weights_grid.shape[0]/(kernel_size+1))+j
                if not np.isclose(np.sum(weights[:, index]), 0):
                    if normalize:
                        weights_grid[i * (kernel_size + 1):i * (kernel_size + 1) + kernel_size, j * (kernel_size + 1):j * (kernel_size + 1) + kernel_size]=\
                            (weights[:, index].reshape(kernel_size, kernel_size, 3) - np.min(weights[:, index])) / ((np.max(weights[:, index]) - np.min(weights[:, index])) + 1.e-6)
                    else:
                        weights_grid[i * (kernel_size + 1):i * (kernel_size + 1) + kernel_size, j * (kernel_size + 1):j * (kernel_size + 1) + kernel_size] =\
                        (weights[:, index].reshape(kernel_size, kernel_size, 3))
                    if modifier is not None:
                        weights_grid[i * (kernel_size + 1):i * (kernel_size + 1) + kernel_size, j * (kernel_size + 1):j * (kernel_size + 1) + kernel_size] *= modifier[index]

        return weights_grid

    def saveFinalPlots(self, errors_train, errors_test, sparsity_train, sparsity_test, errors_train_vector, errors_test_vector, epoch=0):
        #plot errors
        plt.figure(2, figsize=(10, 7))
        plt.clf()
        plt.plot(np.arange(len(errors_train)), errors_train, label='train error')
        plt.plot(np.arange(len(errors_train)), errors_test, label='test error')
        plt.colors()
        plt.legend()
        plt.title('Reconstruction error convergence')
        plt.xlabel('t')
        plt.ylabel('Reconstruction error')
        plt.savefig('plots/Reconstruction_errors_'+str(epoch)+'.pdf')

        #plot sparsity, real and non-zero
        plt.figure(3, figsize=(10, 7))
        plt.clf()
        plt.plot(np.arange(len(sparsity_train)), sparsity_train, label='train error')
        plt.plot(np.arange(len(sparsity_test)), sparsity_test, label='test error')
        plt.colors()
        plt.legend()
        plt.title('Objective function error convergence')
        plt.xlabel('t')
        plt.ylabel('E')
        plt.savefig('plots/Sparsity_'+str(epoch)+'.pdf')

        # plot reconstruction error output progression over time
        plt.figure(12, figsize=(10, 7))
        plt.clf()
        image=plt.imshow(np.clip(np.asarray(errors_train_vector).T, 0, 1), interpolation='nearest', aspect='auto', origin='lower')
        plt.xlabel('t')
        plt.ylabel('Output units \n (Rank Ordered)')
        plt.colors()
        plt.colorbar(image, label='reconstruction error')
        plt.title('Progressive reconstruction input error convergence')
        plt.savefig('plots/Reconstruction_errors_vector_' + str(epoch) + '.pdf')


    def saveNetworkPlots(self, network, epoch=0, calc_error_surface=False):
        # plot weights
        plt.figure(4, figsize=(10, 7))
        plt.clf()
        plt.imshow(self.reshapeWeights(network.weights), interpolation='nearest', vmin=0, vmax=1)
        plt.axis('off')
        plt.colors()
        plt.title('Learned filters')  # normalized
        plt.savefig('plots/Final_filters_' + str(epoch) + '.pdf')
        plt.pause(0.01)

        #plot progressive reconstruction
        plt.figure(5, figsize=(10, 7))
        plt.clf()
        reconstruction = np.ascontiguousarray(network.reconstruction[:, network.output_ranks])
        reconstruction[:, -1] = network.input#overwrite last one to show real image
        plt.imshow(self.reshapeWeights(reconstruction, normalize=False), interpolation='nearest', vmin=0, vmax=1)
        plt.axis('off')
        plt.colors()
        plt.title('Proggressive reconstruction')
        plt.savefig('plots/Progressive_reconstruction_' + str(epoch) + '.pdf')


        #plot unit outputs
        plt.figure(6, figsize=(10, 7))
        plt.clf()
        plt.imshow(self.reshapeWeights(network.weights, normalize=True, modifier=network.output.ravel()), interpolation='nearest', vmin=0, vmax=1)
        plt.axis('off')
        plt.colors()
        plt.title('Proggressive reconstruction filters used')  # normalized and possibly sorted
        plt.savefig('plots/Progressive_reconstruction_usedfilters_' + str(epoch) + '.pdf')


        #plot current input
        plt.figure(7, figsize=(5, 5))
        plt.clf()
        plt.imshow(network.input.reshape(network.input_shape), interpolation='nearest', vmin=0, vmax=1)
        plt.axis('off')
        plt.colors()
        plt.title('Current input')
        plt.savefig('plots/Current_input_' + str(epoch) + '.pdf')

        #plot reconstruction error surface over n samples
        if calc_error_surface:
            step_size = 5
            error_surface = np.zeros((network.weights.shape[0]/step_size, network.weights.shape[1]/step_size))
            arg_sorted_input = np.argsort(network.input, axis=1, kind='mergesort').ravel()[::-1].astype(np.int32)
            orig_input = np.copy(network.input)
            for i in range(0, error_surface.shape[0]):
                #drop inputs
                sorted_input_dropped = orig_input[:, arg_sorted_input]
                sorted_input_dropped[:, (i+1)*step_size:] = 0
                #execute
                network.input[:, arg_sorted_input] = sorted_input_dropped
                network.execute()
                network.rank_output()
                for j in range(0, error_surface.shape[1]):
                    #drop outputs
                    sorted_output_dropped = np.copy(network.output_raw[:, network.output_ranks])
                    sorted_output_dropped[:, (j+1)*step_size:] = 0
                    #TAKE ABS VALUE
                    network.output[:, network.output_ranks] = np.abs(sorted_output_dropped)
                    #reconstruct
                    network.reconstruct()
                    network.input = np.copy(orig_input)
                    error_surface[i, j]+= network.get_reconstruction_error()

            network.input = orig_input
            network.execute()
            network.rank_output()
            network.reconstruct()
            plt.figure(8, figsize=(10, 7))
            plt.clf()
            plt.colors()
            image=plt.matshow(np.minimum(1, error_surface), fignum=8)
            plt.colorbar(image, label='reconstruction error')
            plt.ylabel('active inputs \n (rank ordered)')
            plt.colors()
            plt.title('active outputs \n (rank ordered)')
            plt.savefig('plots/Real_error_surface_' + str(epoch) + '.pdf')

        # plot histogram outputs
        plt.figure(9, figsize=(10, 7))
        plt.clf()
        plt.hist(network.output[0], bins=network.output.shape[1])
        plt.title('Histogram output units')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('plots/Histogram_output_' + str(epoch) + '.pdf')

        # plot histogram outputs negative
        plt.figure(10, figsize=(10, 7))
        plt.clf()
        plt.hist(network.output_raw[0], bins=network.output.shape[1])
        plt.title('Histogram output units')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig('plots/Histogram_output_negative_' + str(epoch) + '.pdf')

        #plot sorted weights
        plt.figure(11, figsize=(10, 7))
        plt.clf()
        weights_sorted = np.ascontiguousarray(network.weights[:, np.argsort(network.output_average)[::-1]])
        plt.imshow(self.reshapeWeights(weights_sorted), interpolation='nearest', vmin=0, vmax=1)
        plt.axis('off')
        plt.colors()
        plt.title('Learned filters sorted')
        plt.savefig('plots/Sorted_filters_' + str(epoch) + '.pdf')