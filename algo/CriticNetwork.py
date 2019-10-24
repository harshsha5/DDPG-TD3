import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Activation,BatchNormalization
from tensorboardX import SummaryWriter
from keras import backend as K

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_critic_network(state_size, action_size, learning_rate):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    state_input = tf.placeholder("float",[None,state_size])
    action_input = tf.placeholder("float",[None,action_size])
    x_1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.relu)([state_input, action_input])  #VALIDATE
    x_2 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.relu)(x_1)          # See if adding Batch normalization helps
    value = tf.keras.layers.Dense(1, activation=tf.nn.linear)(x_2)                  # Add some weight initilization say Xavier
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.tau = tau
        model, _, _ = create_critic_network(state_size,action_size,learning_rate)
        self.critic_network = model
        self.target_critic_network = self.critic_network

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        """Computes dQ(s, a) / da.
        Note that tf.gradients returns a list storing a single gradient tensor,
        so we return that gradient, rather than the singleton list.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
        Returns:
            grads: a batched numpy array storing the gradients.
        """
        with tf.GradientTape() as t:
            t.watch(actions)
            Q = self.critic_network.predict([states, actions])
        return t.gradient(Q, actions)[0]


    def update_target(self):
        """Updates the target net using an update rate of tau."""
        self.target_critic_network = self.tau*self.critic_network + (1-self.tau)*self.target_critic_network
