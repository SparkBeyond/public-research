import tensorflow as tf
import numpy as np
import utilsSG


class SemanticGanNet():
    def __init__(self, number_of_bins, dimensions, batch_size, boundaries=[], numeric_input_mode=False, state_size=16, parameters_per_dimension_flag=False, num_of_lstm_layers=3):
        self.dimensions = dimensions
        self.state_size = state_size
        self.batch_size = batch_size
        self.weights, self.biases = [], []
        self.number_of_bins = number_of_bins
        self.boundaries = boundaries
        self.numeric_input_mode = numeric_input_mode
        self.parameters_per_dimension_flag = parameters_per_dimension_flag
        for i in range(dimensions if parameters_per_dimension_flag else 1):
            self.weights.append(tf.get_variable(name='W' + str(i), shape=(state_size, number_of_bins[i]), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32))
            self.biases.append(tf.Variable(tf.zeros(shape=number_of_bins[i], dtype=tf.float32), name='bias' + str(i), dtype=tf.float32))
        #self.rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=False)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(state_size)

        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell() for _ in range(num_of_lstm_layers)])

    def generate(self):
        output_probs, states, samples = [], [], []
        initial_state = state = self.stacked_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        samples = []
        for i in range(self.dimensions):
            param_index = i if self.parameters_per_dimension_flag else 0
            if i == 0:
                start_token = tf.zeros((self.batch_size, 1)) if self.numeric_input_mode else \
                    tf.tile(tf.one_hot(tf.constant([self.number_of_bins[0]], dtype=tf.int32), self.number_of_bins[0] + 1, 1.0, 0.0), (self.batch_size, 1))
                o_prob, state = self.stacked_lstm(start_token, state=initial_state)
            else:
                last_token = tf.one_hot(samples[-1], self.number_of_bins[0] + 1, 1.0, 0.0)
                o_prob, state = self.stacked_lstm(last_token, state=state)
            output_probs.append(tf.nn.softmax(tf.matmul(o_prob, self.weights[param_index]) + self.biases[param_index]))
            result = tf.squeeze(tf.multinomial(tf.log(output_probs[-1]), num_samples=1), axis=1)
            samples.append(result)
            states.append(state)
        return output_probs, states, samples

    def generate_with_supervision(self, X):
        outputs, states, samples = [], [], []
        input_size = tf.shape(X)[0]
        initial_state = state = self.stacked_lstm.zero_state(batch_size=input_size, dtype=tf.float32)
        samples = []
        for i in range(self.dimensions):
            param_index = i if self.parameters_per_dimension_flag else 0
            if i == 0:
                start_token = tf.zeros((input_size, 1)) if self.numeric_input_mode else \
                    tf.tile(tf.one_hot(tf.constant([self.number_of_bins[0]], dtype=tf.int32), self.number_of_bins[0] + 1, 1.0, 0.0), (input_size, 1))
                output, state = self.stacked_lstm(start_token, state=initial_state)

            else:
                next_token = tf.one_hot(X[:, i-1], self.number_of_bins[0] + 1, 1.0, 0.0) if not self.numeric_input_mode else \
                    utilsSG.reverse_discretiztion_tf(tf.expand_dims(X[:, i-1], -1), self.boundaries, dim=i)
                output, state = self.stacked_lstm(next_token, state=state)
            outputs.append(tf.nn.softmax(tf.matmul(output, self.weights[param_index]) + self.biases[param_index]))
            samples.append(X[:, i])
            states.append(state)
        samples = tf.stack(samples)
        return outputs, states, samples

    def dynamic_extrapolate_from_input(self, X, number_of_input_steps):
        X = tf.cast(X, tf.int64)
        output_probs, states, samples = [], [], []
        input_size = tf.shape(X)[0]
        initial_state = state = self.stacked_lstm.zero_state(batch_size=input_size, dtype=tf.float32)
        samples = []
        for i in range(self.dimensions):
            param_index = i if self.parameters_per_dimension_flag else 0
            if i == 0:
                start_token = tf.zeros((self.batch_size, 1)) if self.numeric_input_mode else \
                    tf.tile(tf.one_hot(tf.constant([self.number_of_bins[0]], dtype=tf.int32), self.number_of_bins[0] + 1, 1.0, 0.0), (input_size, 1))
                o_prob, state = self.stacked_lstm(start_token, state=initial_state)
            else:
                per_raw_condition = tf.less(i, number_of_input_steps)
                if not self.numeric_input_mode:
                    last_token = tf.where(per_raw_condition, tf.one_hot(X[:, i - 1], self.number_of_bins[0] + 1, 1.0, 0.0), tf.one_hot(samples[-1], self.number_of_bins[0] + 1, 1.0, 0.0))
                else:
                    last_token = tf.where(per_raw_condition, utilsSG.reverse_discretiztion_tf(tf.expand_dims(X[:, i - 1], -1), self.boundaries, dim=i), utilsSG.reverse_discretiztion_tf(tf.expand_dims(samples[-1], -1), self.boundaries, dim=i))
                o_prob, state = self.stacked_lstm(last_token, state=state)
            output_probs.append(tf.nn.softmax(tf.matmul(o_prob, self.weights[param_index]) + self.biases[param_index]))
            per_raw_condition = tf.less(i, number_of_input_steps)
            result = tf.where(per_raw_condition, X[:, i], tf.squeeze(tf.multinomial(tf.log(output_probs[-1]), num_samples=1), axis=1))
            samples.append(result)
            states.append(state)
        return output_probs, states, samples


class RnnDiscriminator():

    def __init__(self, state_size, dimensions, number_of_bins):
        self.state_size = state_size
        self.weights = tf.get_variable(name='W_D', shape=(state_size, 2), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        self.biases = tf.Variable(tf.zeros(shape=2, dtype=tf.float32), dtype=tf.float32)
        self.rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.state_size, activation=tf.nn.relu)
        self.number_of_bins = number_of_bins
        self.dimensions = dimensions
        self.session = None

    def forward(self, X):
        batch_size, dimensions = tf.shape(X)[0], tf.shape(X)[1]
        initial_state = self.rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        for i in range(self.dimensions+1):
            if i == 0:
                start_token = tf.tile(tf.one_hot(tf.constant([self.number_of_bins[0]], dtype=tf.int32), self.number_of_bins[0] + 1, 1.0, 0.0), (batch_size, 1))
                output, state = self.rnn_cell(start_token, state=initial_state)
            else:
                next_token = tf.one_hot(X[:, i-1], self.number_of_bins[0] + 1, 1.0, 0.0)
                output, state = self.rnn_cell(next_token, state=state)

        prediction = tf.nn.softmax(tf.matmul(output, self.weights) + self.biases)

        return prediction
