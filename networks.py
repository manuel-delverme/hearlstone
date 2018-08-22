import warnings
import numpy as np
import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, state_size, action_size, model_file=None):
        self.input_states = tf.placeholder(tf.float32, shape=[None, state_size, 1])
        # self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])

        # TODO: this should be resnets ? 

        with tf.name_scope('convolutions') as scope:
            self.conv1 = tf.layers.conv1d(
                    inputs=self.input_states,
                    filters=32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    data_format="channels_last",
                    activation=tf.nn.relu
            )
            self.conv2 = tf.layers.conv1d(
                    inputs=self.conv1,
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    data_format="channels_last",
                    activation=tf.nn.relu
            )

            self.conv3 = tf.layers.conv1d(
                    inputs=self.conv2,
                    filters=128,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    data_format="channels_last",
                    activation=tf.nn.relu
            )
            # 3-1 Action Networks
            self.action_conv = tf.layers.conv1d(
                    inputs=self.conv3,
                    filters=4,
                    kernel_size=1,
                    padding="same",
                    data_format="channels_last",
                    activation=tf.nn.relu
            )

        self.action_conv_flat = tf.contrib.layers.flatten(
            self.action_conv,
        )

        # 3-2 Full connected layer,
        # the output is the log probability of moves on each possible action
        self.action_fc = tf.layers.dense(
                inputs=self.action_conv_flat,
                units=action_size,
                activation=tf.nn.log_softmax
        )

        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv1d(
                inputs=self.conv3, filters=2,
                kernel_size=1,
                padding="same",
                data_format="channels_last",
                activation=tf.nn.relu
        )
        self.evaluation_conv_flat = tf.contrib.layers.flatten(self.evaluation_conv)

        self.evaluation_fc1 = tf.layers.dense(
                inputs=self.evaluation_conv_flat,
                units=64, activation=tf.nn.relu
        )

        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(
                inputs=self.evaluation_fc1,
                units=1,
                activation=tf.nn.tanh
        )

        # loss function
        # label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # predictions: the array containing the evaluation score of each state which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.evaluation_fc2)

        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, action_size])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))

        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        trainable_vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name.lower()])

        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
        )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, state, possible_actions):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        state = np.array(state)
        state = np.expand_dims(state, -1)
        state = np.expand_dims(state, 0)
        state = np.ascontiguousarray(state)
        import ipdb; ipdb.set_trace()
        act_probs, value = self.policy_value(state)
        for act, prob in zip(possible_actions, act_probs):
            print(act, prob)
        return act_probs, value

    def train(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={
                    self.input_states: state_batch,
                    self.mcts_probs: mcts_probs,
                    self.labels: winner_batch,
                    self.learning_rate: lr
        })
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)


