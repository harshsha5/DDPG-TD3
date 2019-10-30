import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import argparse
import gym
import envs

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
parser.add_argument('--train',dest='train',type=int,default=1)
parser.add_argument('--model',dest='model_file',type=str)
parser.add_argument('--buffer_size',dest='buffer_size',type=int,default=1000000)
parser.add_argument('--batch_size',dest='batch_size',type=int,default=1024)
parser.add_argument('--gamma',dest='gamma',type=float,default=0.98)
parser.add_argument('--tau',dest='tau',type=float,default=0.05)
parser.add_argument('--lr_actor',dest='lr_actor',type=float,default=0.0001)
parser.add_argument('--lr_critic',dest='lr_critic',type=float,default=0.001)
parser.add_argument('--lr_critic2',dest='lr_critic2',type=float,default=0.001)
parser.add_argument('--noise_mu',dest='noise_mu',type=float,default=0.0)
parser.add_argument('--noise_sig',dest='noise_sig',type=float,default=0.05)
parser.add_argument('--eps',dest='eps',type=float,default=0.2)
parser.add_argument('--burn_in_size',dest='burn_in_size',type=float,default=5000)
parser.add_argument('--episodes',dest='episodes',type=int,default=10000)
parser.add_argument('--policy_update_frequency',dest='policy_update_frequency',type=int,default=1)
parser.add_argument('--td3',dest='td3',type=bool,default=False)
args = parser.parse_args()

BUFFER_SIZE = args.buffer_size
BATCH_SIZE = args.batch_size
GAMMA = args.gamma                   # Discount for rewards.
TAU = args.tau                     # Target network update rate.
LEARNING_RATE_ACTOR = args.lr_actor
LEARNING_RATE_CRITIC = args.lr_critic
LEARNING_RATE_CRITIC2 = args.lr_critic2
NOISE_MU = args.noise_mu
NOISE_SIGMA = args.noise_sig
EPSILON = args.eps
BURN_IN_MEMORY = args.burn_in_size
EPISODES = args.episodes
POLICY_UPDATE_FREQUENCY = args.policy_update_frequency

class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma,2)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name,is_TD3):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env

        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)
        self.batch_size = BATCH_SIZE
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.burn_in_memory_size = BURN_IN_MEMORY
        self.Critic = CriticNetwork(self.sess,state_dim,action_dim,self.batch_size,tau=TAU,learning_rate=LEARNING_RATE_CRITIC)
        self.noise_mu = NOISE_MU
        self.Noise_sigma = NOISE_SIGMA*(env.action_space.high[0] - env.action_space.low[0])
        self.Actor = ActorNetwork(sess=self.sess,state_size=state_dim,action_size=action_dim,batch_size=self.batch_size,tau=TAU,learning_rate=LEARNING_RATE_ACTOR)
        self.is_TD3 = is_TD3
        self.policy_update_frequency = 1 #If not using TD3 hard setting freq to 1
        self.Critic2 = None
        if(self.is_TD3):
            print("TD3 running")
            self.Critic2 = CriticNetwork(self.sess,state_dim,action_dim,self.batch_size,tau=TAU,learning_rate=LEARNING_RATE_CRITIC2)
            self.policy_update_frequency = POLICY_UPDATE_FREQUENCY
        else:
            print("DDPG running")

        # Defining a custom name for the Tensorboard summary.
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_path = "runs/DDPG_"+timestr+'/'
        self.writer = SummaryWriter(save_path)
        self.outfile = outfile_name
        self.action_range = 1

    def generate_burn_in(self):
        num_actions = self.env.action_space.shape[0]
        state = self.env.reset()
        state = np.array(state)
        done = False       
        for i in range(self.burn_in_memory_size):
            action = np.random.uniform(-1.0, 1.0, size=num_actions)         #Randomly generating actions for the buffer burn_in
            new_state, reward, done, info  = self.env.step(action)
            new_state = np.array(new_state)
            self.buffer.add(state,action,reward,new_state,done)
            state = new_state
            if(done):
                state = self.env.reset()
                state = np.array(state)
                done = False 

    def evaluate(self, num_episodes, num_iteration):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.Actor.actor_network.predict(s_t[None])[0]
                # import pdb; pdb.set_trace()
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    # plt.show()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards) , buf

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """
        self.generate_burn_in()
        for i in range(num_episodes):
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0
            store_states = []
            store_actions = []
            self.ActionNoise = EpsilonNormalActionNoise(self.noise_mu,self.Noise_sigma,EPSILON)
            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = np.clip(self.ActionNoise(self.Actor.actor_network.predict(s_t[None])[0]), -self.action_range, self.action_range) 
                # import pdb; pdb.set_trace()
                
                new_state, reward, done, info  = self.env.step(action)
                new_state = np.array(new_state)
                self.buffer.add(s_t,action,reward,new_state,done)
                transition_minibatch = np.asarray(self.buffer.get_batch(self.batch_size))
                
                target_actions = self.Actor.target_actor_network.predict(np.stack(transition_minibatch[:,3]))
                if(not self.is_TD3):
                    target_Qs = self.Critic.target_critic_network.predict([np.stack(transition_minibatch[:,3]),target_actions])
                else:
                    target_Q1s = self.Critic.target_critic_network.predict([np.stack(transition_minibatch[:,3]),target_actions])
                    target_Q2s = self.Critic2.target_critic_network.predict([np.stack(transition_minibatch[:,3]),target_actions])
                    target_Qs = np.minimum(target_Q1s ,target_Q2s)
                
                target_values = np.stack(transition_minibatch[:,2]) + GAMMA*target_Qs.reshape(-1)
                reward_indices = np.where(transition_minibatch[:,4]==True)[0]
                target_values[reward_indices] = target_values[reward_indices] - GAMMA*(target_Qs.reshape(-1)[reward_indices])

                # present_values = self.Critic.critic_network.predict([transition_minibatch[:,0][0][None],transition_minibatch[:,1][0][None]])
                history = self.Critic.critic_network.fit([np.stack(transition_minibatch[:,0]), np.stack(transition_minibatch[:,1])], target_values, batch_size=self.batch_size, epochs=1, verbose=0)
                if(self.is_TD3):
                    history2 = self.Critic2.critic_network.fit([np.stack(transition_minibatch[:,0]), np.stack(transition_minibatch[:,1])], target_values, batch_size=self.batch_size, epochs=1, verbose=0)               
                #Update Actor Policy
                
                if((step+1)%self.policy_update_frequency==0):
                    actor_actions = self.Actor.actor_network.predict(np.stack(transition_minibatch[:,0]))
                    action_grads = self.Critic.gradients(np.stack(transition_minibatch[:,0]), actor_actions)[0]
                    gradient_update = self.Actor.train(np.stack(transition_minibatch[:,0]), action_grads)
                # import pdb; pdb.set_trace()

                    self.Critic.update_target()
                    self.Actor.update_target()
                    if(self.is_TD3):
                        self.Critic2.update_target()
                
                loss += history.history['loss'][-1]
                loss += history2.history['loss'][-1]
                s_t = new_state
                step += 1
                total_reward += reward
            
            loss = loss/step
            
            self.writer.add_scalar('train/loss', loss, i)
            self.writer.add_scalar("Training Reward VS Episode", total_reward, i)

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(new_s)
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d" % (i, total_reward))
            print("\tTD loss = %.2f" % (loss,))
            print("\tSteps = %d; Info = %s" % (step, info['done']))
            if i % 100 == 0:
                successes, mean_rewards, std_rewards, buf = self.evaluate(10, i)
                image = tf.image.decode_png(buf.getvalue(), channels=3)
                image = image.eval(session=self.sess)
                self.writer.add_image('Performance', image, i, dataformats='HWC')
                self.writer.add_scalar('mean_reward', mean_rewards, i)
                self.writer.add_scalar('std_reward', std_rewards, i)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))

            
    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        raise NotImplementedError

def main():
    
    env = gym.make('Pushing2D-v0')
    # print(env.action_space.sample())
    # print(env.observation_space)
    print(env.action_space.high)
    print(env.action_space.low)
    # import pdb; pdb.set_trace()
    # num_states = env.observation_space.shape[0]
    # num_actions = env.action_space.shape[0]
    # print("Number of states: ",num_states)
    # print("Number of actions: ",num_actions)
    
    algo = DDPG(env, '../ddpg_log.txt', args.td3)
    algo.train(EPISODES, hindsight=False)


if __name__ == '__main__':
    main()