import gym
import envs
from algo.ddpg import DDPG


def main():
    env = gym.make('Pushing2D-v0')
    # print(env.action_space.sample())
    # print(env.observation_space)
    print(env.action_space.high)
    print(env.action_space.low)
    # num_states = env.observation_space.shape[0]
    # num_actions = env.action_space.shape[0]
    # print("Number of states: ",num_states)
    # print("Number of actions: ",num_actions)
    algo = DDPG(env, 'ddpg_log.txt')
    algo.train(1, hindsight=False)


if __name__ == '__main__':
    main()
