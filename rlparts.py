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
import random

import pandas as pd
import datetime as dt

from pprint import pprint as ppp
from params import *










from rl.policy import *


class ValidGreedyQPolicy(Policy):
	"""Implement the epsilon greedy policy

	Eps Greedy policy either:

	- takes a random action with probability epsilon
	- takes current best action with prob (1 - epsilon)
	"""

	env = None

	def __init__(self):
		super(ValidGreedyQPolicy, self).__init__()

	def select_action(self, q_values):
		"""Return the valid action with highest Q value
		"""
		actions = np.argsort(q_values)

		for ai in actions:
			if self.env.action_is_valid(ai):
				return ai


class ValidEpsGreedyQPolicy(Policy):
	"""Implement the epsilon greedy policy

	Eps Greedy policy either:

	- takes a random action with probability epsilon
	- takes current best action with prob (1 - epsilon)
	"""

	env = None

	def __init__(self, eps=.1):
		super(ValidEpsGreedyQPolicy, self).__init__()
		self.eps = eps

	def select_action(self, q_values):
		"""Return the selected action

		# Arguments
			q_values (np.ndarray): List of the estimations of Q for each action

		# Returns
			Selection action
		"""
		assert q_values.ndim == 1
		nb_actions = q_values.shape[0]

		if np.random.uniform() < self.eps:
			action = randint(0, len(q_values) - 1)
			while not self.env.action_is_valid(action):
				action = randint(0, len(q_values) - 1)
			return action
		else:
			action = np.argmax(q_values)

			actions = np.argsort(q_values)
			for ai in actions:
				if self.env.action_is_valid(ai):
					return ai
			# if we got here, there are no valid actions.
			# the framework forces us to pick an action,
			# to handle rl/core.py line 209 if done: ....
			return actions[0]


	def get_config(self):
		"""Return configurations of EpsGreedyPolicy

		# Returns
			Dict of config
		"""
		config = super(ValidEpsGreedyQPolicy, self).get_config()
		config['eps'] = self.eps
		return config




def reflected_board(board):
	return board.T()

def reflected_action():
	return



import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Add, Lambda
from keras.optimizers import Adam
import keras.backend as K
from keras import regularizers


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from keras.layers import Dense, Input, GlobalMaxPooling1D, InputLayer

import datetime

import random

# Monkey-patching DQNAgent... Could just subclass

# TODO: Save training stats history with agent
# - game count
# - opponent for this block
# - stats vs random; self; chaos

def dqnagent_reload(self):
	print('Loading dqn weights from ',self.modelfile)
	try:
		r=self.load_weights(self.modelfile)
		return r
	except Exception as e:
		print(e)
DQNAgent.reload=dqnagent_reload

def dqnagent_save(self):
	print('Saving dqn weights to ',self.modelfile)
	return self.save_weights(self.modelfile,overwrite=True)
DQNAgent.save=dqnagent_save

import gzip

def dqnagent_save_memory( self ):
	print('Saving dqn memory to ',self.memoryfile)

	# Do not use deque to implement the memory. This data structure may seem convenient but
	# it is way too slow on random access. Instead, we use our own ring buffer implementation.
	mem = ( self.memory, self.memory.actions,
	self.memory.rewards,
	self.memory.terminals,
	self.memory.observations )

	fp = gzip.open(self.memoryfile, 'wb')
	cPickle.dump(mem, fp, protocol=-1)  # highest protocol means binary format
	fp.close()

	print('ok!')
DQNAgent.save_memory=dqnagent_save_memory


def dqnagent_load_memory( self ):
	print('Loading dqn memory from ',self.memoryfile)
	try:
		fp = gzip.open(self.memoryfile, 'rb')

		(self.memory.limit, self.memory.actions,
		 self.memory.rewards,
		 self.memory.terminals,
		 self.memory.observations) = cPickle.load( fp )  # highest protocol means binary format
		fp.close()

		return
	except Exception as e:
		print(e)
	print('ok!')
DQNAgent.reload_memory=dqnagent_load_memory



'''
class DummyProcessor(Processor):
    # '' '
    Stubs.
    # ' ''

    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        return batch

    def process_reward(self, reward):
        return reward #np.clip(reward, -1., 1.)

    def process_action(self, action):
        return action
'''

from keras.callbacks import TensorBoard

import _pickle as cPickle


from agents import *

class Regime():

	env=None
	agent_factory_fn = None
	dqn_learner = None
	STEPS_FIT = None
	agency = None

	#def get_dqn_agent(self):

	def make_and_add_agent(self,label,agent):
		agent.env = self.env
		self.agency.add_agent(label,agent)

	def init(self, env, CONTINUE_EXISTING_RUN, agent_factory_fn,STEPS_FIT = 3000 ):

		# env = gym.make(args.env_id)
		self.env = env
		self.agent_factory_fn = agent_factory_fn
		self.STEPS_FIT = STEPS_FIT

		self.agency = Agency()

		#dqn_kind = 'dqn3'
		dqn_kind = dqn_label


		# TODO: redesign dqn_learner so that:
		# its a class,
		# it remembers it own model file path
		# it can thus load and write itself
		# its model file path is not just based on the type of model (xos2, xos3 etc)
		#   because we should have multiple instances of agents using every model

		dqn_learner = self.agent_factory_fn(self.env, dqn_kind, load=CONTINUE_EXISTING_RUN,side_normalization_factor=1.0)
		dqn_learner.agent_kind = "dqn_learner"
		dqn_learner.agent_label =  str(random.randint(1000000, 9999999))
		self.agency.add_agent('dqn_learner',dqn_learner)


		dqn_opponent = self.agent_factory_fn(self.env, dqn_kind, load=CONTINUE_EXISTING_RUN,side_normalization_factor=-1.0)
		dqn_opponent.agent_kind = "dqn_opponent"
		dqn_opponent.agent_label =  str(random.randint(1000000, 9999999))
		self.agency.add_agent('dqn_opponent',dqn_opponent)

		dqn_bot = self.agent_factory_fn(self.env, dqn_kind,load=CONTINUE_EXISTING_RUN,side_normalization_factor= -1.0)
		dqn_bot.agent_label =  str(random.randint(1000000, 9999999))
		wrapper_agent = WrapperAgent(dqn_bot, self.env.action_space)
		wrapper_agent.env = self.env
		self.agency.add_agent('wrapper_opponent',wrapper_agent)


		dqn_bot = self.agent_factory_fn(self.env, dqn_kind,load=CONTINUE_EXISTING_RUN,side_normalization_factor= 1.0)
		dqn_bot.agent_label =  str(random.randint(1000000, 9999999))
		wrapper_agent = WrapperAgent(dqn_bot, self.env.action_space)
		wrapper_agent.env = self.env
		self.agency.add_agent('wrapper_protagonist',wrapper_agent)


		dqn_bot = self.agent_factory_fn(self.env, dqn_kind, load=CONTINUE_EXISTING_RUN,side_normalization_factor= -1.0)
		dqn_bot.agent_label =  str(random.randint(1000000, 9999999))

		# the new way to add agents... assumes easy class constructor?
		self.make_and_add_agent('chaos',ChaosDqnAgent(dqn_bot, self.env.action_space, regime_params['epsilon-chaos']))
		self.make_and_add_agent('delta-chaos',DeltaChaosDqnAgent(dqn_bot, self.env.action_space, regime_params['epsilon-chaos'],regime_params['delta_window']))
		self.make_and_add_agent('countchocula',CountChoculaAgent(self.env.action_space))
		self.make_and_add_agent('random',RandomAgent(self.env.action_space))
		self.make_and_add_agent('human',HumanAgent(self.env.action_space))

		print(self.agency.list_agents())
		# its wierd that we pick the turn_side on agent creation.  would be better to set it later.



	# opponent we consider part of the environment
	def install_opponent_kind(self,agent_kind):
		self.install_opponent( self.agency.find_agent_kind(agent_kind))

	def install_opponent(self,agent):
		self.env.opponent_agent = agent
		self.env.opponent_agent.env = self.env




	def test_human(self):
		'''
		Set environment oppenent.agent to Human
		'''
		self._test_human_init()

		self.install_opponent_kind("human")

		self.env.reset()
		board = self.env.obs()

		while True:

			# player 1
			# print("PLAYER 1")
			# rm = randint(0, env.wid*env.ht-1)
			# while not env.action_is_valid(rm):
			#	rm = randint(0, env.wid * env.ht-1)
			dqn_learner = self.agency.find_agent_kind("dqn_learner")

			action = dqn_learner.forward(board)

			board, reward, done, info = self.env.step(action, verbose=True)

			dqn_learner.backward(reward, done)

			# env.render_pretty()

			print("reward:", reward)

			if done:
				self.env.reset()
				board = self.env.obs()

				self.install_opponent_kind("human")
				print('======= REMATCH! ========')


	def _test_human_init(self):
		self.env.reset()
		dqn=self.agency.find_agent_kind('dqn_learner')
		dqn.training = True
		self.install_opponent(dqn)
		board = self.env.obs()
		return board







	def test_human_web_init(self,opponent_kind):
		'''
		Set environment oppenent.agent to Human
		'''
		opponent=self.agency.find_agent_kind(opponent_kind)
		self.install_opponent(opponent)
		self.env.reset()

		#??? does this work?
		#dqn_opponent.training = True

		return self.env.obs()


	def test_human_web_act(self,action):
		board, reward, done, info = self.env.step(action)
		#info['q_values']=None
		if hasattr(self.env.opponent_agent,'q_values'):
			info['q_values']=self.env.opponent_agent.q_values
		return board, reward, done, info


	def test_human_web_step(self):
		board = self.env.obs()
		# player 1
			# print("PLAYER 1")
		# rm = randint(0, env.wid*env.ht-1)
		# while not env.action_is_valid(rm):
		#	rm = randint(0, env.wid * env.ht-1)

		dqn_learner = self.agency.find_agent_kind('dqn_learner')

		action = dqn_learner.forward(board)

		board, reward, done, info = self.env.step(action, verbose=True)

		dqn_learner.backward(reward, done)

		# env.render_pretty()

		print("reward:", reward)








	def train_random(self):

		df = None

		a = self.agency.find_agent_kind('random')
		self.install_opponent(a)
		self.env.reset()

		SAVE_MEMORY_STEPS=regime_params['memory_iterations_per_save']
		c=0
		total_games=0
		total_steps=0
		while True:
			print('--- Training round vs random : ---------------')
			stats_learn=self.learner_learn( c % SAVE_MEMORY_STEPS==0 and c>0 )
			total_games+=stats_learn['games']
			total_steps+=stats_learn['moves']
			self.print_stats_learn(stats_learn)

			# do not include wrapper and chaos opponents in random training
			stats_eval=self.test_evaluate_learner(wrapper_opponent=False,chaos_opponent=False)
			a = self._get_stats_eval(stats_eval)

			a['total_games']=total_games
			a['total_steps']=total_steps
			a['regime']='random'
			a['ts']=dt.datetime.now().strftime("%Y%m%d-%H%M%S")
			print(a)

			if df is None:
				df = pd.DataFrame.from_dict( [a] )
				print('New dataframe')
			else:
				new_df = pd.DataFrame.from_dict( [a] )
				print('Appending dataframe')
				df=df.append(new_df, ignore_index=True)
				df.to_csv('train.csv')

			print(df.to_string())

			c+=1
			#import pdb; pdb.set_trace()
			#return




	def train_self(self, random_cotrain=False):

		train_csv_fn='train.csv'

		cc=0
		total_games=0
		total_steps=0

		opponent_agent = 'delta-chaos'

		# try continuing last session
		df = pd.DataFrame()
		try:
			print('Loading csv training history from ', train_csv_fn)
			df = df.from_csv(train_csv_fn)
			print('Loading csv training history')

			total_games = df.iloc[-1]['total_games']
			total_steps = df.iloc[-1]['total_steps']
		except Exception as e:
			print('Failed loading csv training history')
			print(e)

		while True:
			print(datetime.datetime.utcnow())

			# actually, we want to REBUILD the chaosagent, using new weights...
			agent = self.agency.find_agent_kind(opponent_agent)
			agent.reload()

			print('--- Training round vs %s: ---------------', opponent_agent)
			self.install_opponent(agent)
			stats_learn =self.learner_learn( (cc % regime_params['memory_iterations_per_save'])==0 and cc>0 )
			self.print_stats_learn(stats_learn)

			stats_eval=self.test_evaluate_learner()
			a = self._get_stats_eval(stats_eval)
			print(a)

			#print(stats['random']['fail_rate'])
			total_games+=stats_learn['games']   # this needs to be saved with the agent...
			total_steps+=stats_learn['moves']
			a['total_games']=total_games
			a['total_steps']=total_steps
			a['regime']='self'
			a['ts']=dt.datetime.now().strftime("%Y%m%d-%H%M%S")

			if df is None:
				df = pd.DataFrame.from_dict( [a] )
				print('New dataframe')
			else:
				new_df = pd.DataFrame.from_dict( [a] )
				print('Appending dataframe')
				df=df.append(new_df, ignore_index=True)
				df.to_csv(train_csv_fn)

			print(df.to_string())


			# alternate with random agent, to prevent collusion
			if random_cotrain:
				print('--- Training round vs random : ---------------')
				self.install_opponent_kind('random')
				self.learner_learn()
				stats_eval=self.test_evaluate_learner()
				a = self._get_stats_eval(stats_eval)
				print(a)

			cc+=1




	def learner_learn(self, save_memory=False ):

		dqn_learner = self.agency.find_agent_kind('dqn_learner')
		try:
			dqn_learner.reload()
			#print("loaded dqn weights")
		except Exception as e:
			print(e)
			print("could not load dqn weights: using randomly initialized agent")
			dqn_learner.save()

		'''
		tc = keras.callbacks.TensorBoard(log_dir='/tmp/tensorboard', histogram_freq=0, batch_size=regime_params['memory_batch_size'], write_graph=True, write_grads=False,
		            write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

		#tc.on_epoch_end(self, epoch, logs=None)
		callbacks = [tc, ]

		#callbacks_list=[tc,]
		'''

		dqn_learner.compile(Adam(lr=regime_params['learning_rate']), metrics=['mae'])
		dqn_learner.fit(self.env, nb_steps=regime_params['steps_per_iter'], verbose=0, ) #,verbose=2, visualize=True,)  # callbacks=callbacks,
		dqn_learner.save()

		if save_memory:
			dqn_learner.save_memory()

		st = self.env.epl.get_stats()
		print(st)
		self.env.epl.reset_stats()
		return st

		#dqn_learner.test(self.env, nb_episodes=20, visualize=False)











	def print_stats_learn(self,stats_learn):
		print('stats_learn', stats_learn)


	def _get_stats_eval(self, stats_eval ):
		o=['random','chaos','wrapper_opponent']
		a = {}
		for opponent_label in o:
			if opponent_label in stats_eval:
				k=['loss_rate','tie_rate','win_rate','invalid1_rate','invalid2_rate','fail_rate']
				for kk in k:
					a['v%s__%s' % (opponent_label,kk)] = stats_eval[ opponent_label][kk]
		return a






	def test_evaluate_learner(self,wrapper_opponent=True,chaos_opponent=True):
		print('Evaluation:')
		# reload wrapper weights for testing
		wrapper2 = self.agency.find_agent_kind('wrapper_protagonist')
		wrapper2.reload()

		TEST_ROUNDS = regime_params['test_rounds']

		round_stats={}

		# always test with random
		round_stats['random']=self.test_vs('wrapper_protagonist', 'random', rounds=TEST_ROUNDS)

		if wrapper_opponent:
			wrapper1 = self.agency.find_agent_kind('wrapper_opponent')
			wrapper1.reload()
			round_stats['wrapper_opponent'] =self.test_vs('wrapper_protagonist', 'wrapper_opponent', rounds=TEST_ROUNDS)

		if chaos_opponent:
			chaos = self.agency.find_agent_kind('chaos')
			chaos.reload()
			round_stats['chaos'] =self.test_vs('wrapper_protagonist', 'chaos', rounds=TEST_ROUNDS)
		return round_stats



	def test_vs(self,agent_kind_1,agent_kind_2,rounds=100):

		#print('----------------------')
		ident = {"player":agent_kind_1,"opponent":agent_kind_2,"pair":agent_kind_1+"|"+agent_kind_2}

		protagonist=self.agency.find_agent_kind(agent_kind_1)
		assert protagonist
		protagonist.reload()

		opponent=self.agency.find_agent_kind(agent_kind_2)
		assert opponent
		print (agent_kind_2)
		print(opponent)
		opponent.reload()
		self.install_opponent(opponent)


		for i in range(0,rounds):
			self.install_opponent(opponent)
			self.env.reset()
			# repeat?
			self.install_opponent(opponent)

			board = self.env.obs()
			done = False

			while not done:
				action = protagonist.forward(board)
				#print(agent1+" moves "+str(action))
				board, reward, done, info = self.env.step(action, verbose=False)
				#protagonist.backward(reward, done)

			#print (board,reward,done,info)
		st=self.env.epl.get_stats()
		#print(st)  # env.render_pretty()
		self.env.epl.reset_stats()

		st['ident']=ident
		return st


'''
Problem:
- modelfile is defined in a floating dict
- dont have it when needed to reload
- no common place for saving agent data

- solve this case first.

SOLNS:

- agent classes.  class has reload. 

'''
