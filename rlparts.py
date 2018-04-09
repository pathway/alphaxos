'''
# Xs and Os and Deep Reinforcement Learning
### Deep Q-Learning using an Open AI Gym-like Environment

Concise working example of Deep Q-learning applied to an environment similar to gym/envs/toy_text/ in ai gym


#### References

- OpenAI gym: https://github.com/openai/gym/
- Keras-RL: https://github.com/keras-rl/keras-rl
- DQNs: https://www.nature.com/articles/nature14236
- Keras: https://keras.io/


Copyright (C) 2018  Pathway Intelligence Inc

Author: Robin Chauhan

License: The MIT License


'''

import numpy as np
from gym import Env, spaces

from gym.utils import seeding
from random import randint



class RandomAgent(object):
	'''
	RandomAgent:
	- pick a random action
	- if its not valid, keep picking new random until valid one found
	- return first valid action found
	'''
	env = None

	def __init__(self, action_space):
		self.action_space = action_space
		#print(self.action_space)

	def forward(self,observation):
		return self.act(observation,None,None)

	def act(self, observation, reward, done):
		rm = randint(0, self.action_space.n - 1)
		while not self.env.action_is_valid(rm):
			rm = randint(0, self.action_space.n - 1)

		return rm



class HumanAgent(object):
	'''
	Take in keyboard input to select square.
	'''

	def __init__(self, action_space):
		self.action_space = action_space
		#print(self.action_space)

	def forward(self,observation):
		return self.act(observation,None,None)

	def act(self, observation, reward, done):
		print(observation)
		mx = self.action_space.n
		rm = input("Move 1 to %s: " % str(mx))
		rm = int(rm)-int('1')
		return rm




class ChaosDqnAgent(object):
	'''
	This agent does Epsilon-greedy rollouts.

	The purpose is to ensure that duelling DQNs do not get trapped in mutual local minima.
	Adding a random element prevents competely closing off state pathways.

	Maybe more importantly, it also is an example of composing agents from others.

	It wraps "smart_agent" (eg. a DQN or other agent).
	'''

	smart_agent=None

	env = None

	def __init__(self, side, smart_agent,action_space,random_ratio=0.03):
		self.self_side=side
		self.action_space = action_space
		self.random_ratio = random_ratio
		self.smart_agent = smart_agent
		#print(self.action_space)

	def forward(self,observation):
		return self.act(observation,None,None)

	def act(self, observation, reward, done):
		if random.random() < self.random_ratio:
			return self.random_act(observation,None,None)
		else:
			return self.smart_agent.forward(observation)

	def random_act(self, observation, reward, done):
		rm = randint(0, self.action_space.n - 1)
		while not self.env.action_is_valid(rm):
			rm = randint(0, self.action_space.n - 1)
		return rm





import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Add, Lambda
from keras.optimizers import Adam
import keras.backend as K


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.layers import Dense, Input, GlobalMaxPooling1D, InputLayer

import random




class DummyProcessor(Processor):
    '''
    Stubs.
    '''

    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return reward #np.clip(reward, -1., 1.)




class Regime():

	env=None
	agent_factory_fn = None
	dqn_learner = None
	STEPS_FIT = None

	def init(self, env, CONTINUE_EXISTING_RUN, modelpath, agent_factory_fn,STEPS_FIT = 3000 ):

		# env = gym.make(args.env_id)
		self.env = env
		self.agent_factory_fn = agent_factory_fn
		self.STEPS_FIT = STEPS_FIT

		# opponent we consider part of the environment
		self.env.opponent_agent = RandomAgent(self.env.action_space)
		self.env.opponent_agent.env = self.env
		reward = 0
		done = False

		# store model in this file
		self.modelpath = modelpath

		self.dqn_learner = self.agent_factory_fn(1.0)

		try:
			# False: make a new model; True: continue training from before
			if CONTINUE_EXISTING_RUN:
				self.dqn_learner.load_weights(modelpath)
			print('loaded: ', modelpath)
		except:
			print('could not load: ', modelpath)
			pass

		self.random_agent = RandomAgent(self.env.action_space)
		self.random_agent.env = self.env
		self.human_agent = HumanAgent(self.env.action_space)


	def test_human(self):
		'''
		Set environment oppenent.agent to Human
		'''

		self.env.reset()
		board = self.env.obs()
		self.env.opponent_agent = self.human_agent  # RandomAgent(env.action_space)

		while True:

			# player 1
			# print("PLAYER 1")
			# rm = randint(0, env.wid*env.ht-1)
			# while not env.action_is_valid(rm):
			#	rm = randint(0, env.wid * env.ht-1)

			action = self.dqn_learner.forward(board)
			board, reward, done, info = self.env.step(action, verbose=True)
			# env.render_pretty()

			print("reward:", reward)

			if reward!=0.0:
				self.env.reset()
				board = self.env.obs()
				self.env.opponent_agent = self.human_agent  # RandomAgent(env.action_space)
				print('======= REMATCH! ========')


	def train_random(self):
		while True:
			self.env.opponent_agent = self.random_agent
			self.dqn_learner.fit(self.env, nb_steps=self.STEPS_FIT)  # callbacks=callbacks,
			self.dqn_learner.save_weights(self.modelpath, overwrite=True)
			self.dqn_learner.test(self.env, nb_episodes=200, visualize=True)


	def train_self(self):
		while True:
			while True:
				# reload the weights each loop, which were just saved in last loop
				try:
					dqn_bot = self.agent_factory_fn(-1.0)
					dqn_bot.load_weights(self.modelpath)
					print('loaded: ', self.modelpath)
				except:
					print('could not load: ', self.modelpath)
					dqn_bot = None
					pass
				random_ratio = 0.02
				chaos_agent = ChaosDqnAgent(-1.0, dqn_bot, self.env.action_space, random_ratio)
				chaos_agent.env = self.env

				self.env.opponent_agent = chaos_agent
				self.dqn_learner.fit(self.env, nb_steps=self.STEPS_FIT)  # callbacks=callbacks,
				self.dqn_learner.save_weights(self.modelpath, overwrite=True)
				self.dqn_learner.test(self.env, nb_episodes=200, visualize=True)

				# alternate with random agent, to prevent collusion
				if 1:
					self.env.opponent_agent = self.random_agent
					self.dqn_learner.fit(self.env, nb_steps=self.STEPS_FIT)  # callbacks=callbacks,
					self.dqn_learner.save_weights(self.modelpath, overwrite=True)
					self.dqn_learner.test(self.env, nb_episodes=200, visualize=True)

