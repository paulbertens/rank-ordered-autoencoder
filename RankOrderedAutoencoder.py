import numpy as np


class RankOrderNetwork:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.input = np.zeros((output_shape[0], self.input_shape[0] * self.input_shape[1] *
                               self.input_shape[2]),dtype=np.float32)
        self.output = np.zeros(output_shape, dtype=np.float32)
        self.output_raw = np.zeros_like(self.output)
        self.output_error = np.zeros_like(self.output)
        self.output_average = np.zeros(self.output.shape[1], dtype=np.float32)
        self.weights = np.random.normal(0, np.sqrt(2.0 / (self.output.shape[1] + self.input.shape[1])),
                                        size=(self.input.shape[1], self.output.shape[1])).astype(np.float32)
        self.gradient = np.zeros_like(self.weights)
        self.reconstruction = np.zeros_like(self.weights)
        self.errors = np.zeros_like(self.weights)
        self.output_ranks = np.zeros(self.output.shape[1], dtype=np.int32)
        self.learning_rate = 1
        self.norm_limit = 0.1

    def execute(self):
        # compute output
        np.dot(self.input, self.weights, out=self.output_raw)
        self.output_raw /= self.weights.shape[0]
        self.activation(self.output_raw, out=self.output)

    def activation(self, X, out=None):
        return np.clip(X, 0, 1, out=out)

    def derivative(self, X, Y):
        return (np.sign(Y) != np.sign(X))

    def clip(self, X, out=None):
        return np.clip(X, -1, 1, out=out)

    def rank_output(self):
        self.output_ranks = np.argsort(self.output_raw, axis=1, kind='mergesort').ravel()[::-1].astype(np.int32)

    def reconstruct(self):
        # reconstruct progressively
        self.reconstruction[:, self.output_ranks] = np.cumsum(
            self.weights[:, self.output_ranks] * self.output[:, self.output_ranks], axis=1)
        self.activation(self.reconstruction, out=self.reconstruction)

    def reconstruction_error(self):
        # compute errors
        self.errors[:] = (self.input.T - self.reconstruction)

    def forward_prop(self):
        # backprop
        self.output_error = np.sum(self.errors * self.weights, axis=0).reshape(1, -1)
        self.output_error /= self.weights.shape[0]
        self.output_error *= self.derivative(self.output_raw, self.output_error)
        # clip gradient to not exceed zero
        self.output_error[self.output_raw > 0] = \
            np.maximum(-self.output_raw[self.output_raw > 0],self.output_error[self.output_raw > 0])
        self.output_error[self.output_raw < 0] = \
            np.minimum(-self.output_raw[self.output_raw < 0],self.output_error[self.output_raw < 0])

    def update_weights(self):
        # update gradient x
        self.gradient[:] = (self.errors * self.output)

    def update_weights_backwards(self):
        # update gradient y
        self.gradient += np.dot(self.input.reshape(-1, 1), self.output_error)

    def update_weights_final(self):
        # clip the gradient norm
        norm = np.sqrt(np.sum(self.gradient ** 2, axis=0))
        norm_check = norm > self.norm_limit
        self.gradient[:, norm_check] = ((self.gradient[:, norm_check]) / norm[norm_check]) * self.norm_limit
        # update weights
        self.weights += self.gradient * (self.learning_rate)
        # update output average for sorting weights
        self.output_average *= 0.99999
        self.output_average += self.output.ravel() * 0.00001

    def get_reconstruction_error(self):
        # the reconstruction error
        return np.sqrt(
            np.sum((self.reconstruction[:, self.output_ranks[-1]] - self.input.reshape(self.weights.shape[0])) ** 2))

    def get_sparsity_error_term(self):
        # the total error function being minimized
        return np.sum(np.cumsum(self.get_reconstruction_error_vector()[::-1]))

    def get_reconstruction_error_vector(self):
        # the reconstruction error as function of the sorted output.
        return np.sqrt(np.sum((self.input.T - self.reconstruction[:, self.output_ranks]) ** 2, axis=0))

    def update(self, new_input, train=True):
        # performs network updating, forward pass, reconstruction and optionally updating weights.
        self.input[:] = new_input.reshape(self.input.shape)
        self.execute()
        self.rank_output()
        self.reconstruct()
        if train:
            self.reconstruction_error()
            self.forward_prop()
            self.update_weights()
            self.update_weights_backwards()
            self.update_weights_final()
