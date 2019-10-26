import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from keras.activations import relu, linear, tanh
from keras.initializers import RandomUniform
from keras import backend as K

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400


def create_actor_network(state_size, action_size):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    state_input = Input(shape=(state_size,))                       
    x_1 = Dense(HIDDEN1_UNITS, activation=relu)(state_input)  
    x_2 = Dense(HIDDEN2_UNITS, activation=relu)(x_1)          # See if adding Batch normalization helps
    value = Dense(action_size, activation=tanh,kernel_initializer = RandomUniform(minval=-0.003, maxval=0.003, seed=None),bias_initializer=RandomUniform(minval=-0.003, maxval=0.003, seed=None))(x_2)                  # Add some weight initilization say Xavier
    model = tf.keras.Model(inputs=state_input, outputs=value)   #See if action needs to be scaled to action bound
#    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))         #CHANGE THIS LOSS
    return model, state_input


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.tau = tau
        model, _ = create_actor_network(state_size,action_size)
        self.actor_network = model

        target_model, _ = create_actor_network(state_size,action_size)
        self.target_actor_network = target_model
        self.target_actor_network.set_weights(self.actor_network.get_weights())
        self.batch_size = batch_size

        action_gradients = K.placeholder(shape=(None, action_size))
        params_gradient = [tf.scalar_mul(1/batch_size, x) for x in tf.gradients(self.actor_network.output, self.actor_network.trainable_weights, -action_gradients)]
        self.gradient_function = K.function([self.actor_network.input, action_gradients], outputs=[tf.ones(1)], updates=[tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(params_gradient, self.actor_network.trainable_weights))])

        self.sess = sess
        self.sess.run(tf.initialize_all_variables())


    def train(self, states, action_grads):
        """Updates the actor by applying dQ(s, a) / da.

        Args:
            states: a batched numpy array storing the state.
            actions: a batched numpy array storing the actions.
            action_grads: a batched numpy array storing the
                gradients dQ(s, a) / da.
        """
        self.gradient_function([states, action_grads])

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        self.target_actor_network.set_weights([ tau * x + (1 - tau)*y for x,y,tau in zip(self.actor_network.get_weights(), self.target_actor_network.get_weights(), [self.tau]*len(self.actor_network.get_weights()))])