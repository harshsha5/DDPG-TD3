import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.activations import relu, linear
from keras.initializers import RandomUniform
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
    state_input = Input(shape=(state_size,))
    action_input = Input(shape=(action_size,)) 
    combined_input= Concatenate()([state_input, action_input])
    x_1 = Dense(HIDDEN1_UNITS, activation=relu)(combined_input)  
    x_2 = Dense(HIDDEN2_UNITS, activation=relu)(x_1)          # See if adding Batch normalization helps
    value = Dense(1, activation=linear,kernel_initializer = RandomUniform(minval=-0.0003, maxval=0.0003, seed=None),bias_initializer=RandomUniform(minval=-0.0003, maxval=0.0003, seed=None))(x_2)                  # Add some weight initilization say Xavier
    model = tf.keras.Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model, state_input, action_input, value


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
        model, self.state_input, self.action_input, self.value = create_critic_network(state_size,action_size,learning_rate)
        self.critic_network = model

        target_model, _, _, _ = create_critic_network(state_size,action_size,learning_rate)
        self.target_critic_network = target_model
        self.target_critic_network.set_weights(self.critic_network.get_weights())

        self.action_grads = K.function([self.critic_network.input[0], self.critic_network.input[1]], K.gradients(self.critic_network.output, [self.critic_network.input[1]]))

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
        # actions = tf.convert_to_tensor(actions)
        # with tf.GradientTape() as t:
        #     # t.watch(actions)
        #     Q = self.critic_network.predict([states, actions], steps=1)
        # grad = K.gradients(Q, actions)[0]
        # import pdb; pdb.set_trace()
        # return grad
        return self.action_grads([states, actions])


    def update_target(self):
        """Updates the target net using an update rate of tau."""
        self.target_critic_network.set_weights([ tau * x + (1 - tau)*y for x,y,tau in zip(self.critic_network.get_weights(), self.target_critic_network.get_weights(), [self.tau]*len(self.critic_network.get_weights()))])
