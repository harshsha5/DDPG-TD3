import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model

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
    # state_input = Input(shape=[state_size])                       #I commented some already given code. See if doing this is fair
    state_input = tf.placeholder("float",[None,state_size])
    x_1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation=tf.nn.relu)(state_input)  #VALIDATE
    x_2 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation=tf.nn.relu)(x_1)          # See if adding Batch normalization helps
    value = tf.keras.layers.Dense(action_size, activation=tf.nn.tanh)(x_2)                  # Add some weight initilization say Xavier
    model = tf.keras.Model(inputs=state_input, outputs=value)
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))         #CHANGE THIS LOSS
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
        self.target_actor_network = self.actor_network

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
        with tf.GradientTape() as t:
            t.watch(states)
            mu = self.actor_network.predict([states])
        policy_grads = t.gradient(mu, states)[0]
        grad_J = (1/action_grads.shape[0]) * tf.reduce_sum(policy_grads*action_grads)           #SEE FINAL UPDATE HERE

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        self.target_actor_network = self.tau*self.actor_network + (1-self.tau)*self.target_actor_network
