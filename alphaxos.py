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

from rlparts import  *



class HumanAgentXos(object):
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
		#print(" 1 2 3\n 4 5 6\n 7 8 9")
		print(" 7 8 9\n 4 5 6\n 1 2 3")
		rm = input("Move 1-9: ")
		rm = int(rm)-int('1')

		# convert from keypad positions to matrix positions
		row = rm // 3
		col = rm - row * 3
		print ( row, col )
		row2 = 2-row

		rm2 = row2*3 + col

		return rm2



class BinaryBoardEnv(Env):
	'''
	'''
	def seed(self, seed=42):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		board = None

		self.board = np.zeros((self.wid, self.ht))
		self.self_side=1
		self.turn_side=randint(0,1)*-2 + 1
		self.done=False

		return self.board

	def obs(self):
		return self.board



	def render_pretty(self):
		for row in range(0,self.ht):
			for col in range(0,self.wid):
				val = self.board[row,col]
				if val==1:
					print('X',end='   ')
				elif val==-1:
					print('O',end='   ')
				else:
					print('-',end='   ')
			print("\n")

	def render(self,mode='human'):
		print(self.board)

	# full step, including "our" turn and opponent's turn
	def step(self, action, verbose=False):

		done=False

		if self.turn_side==1:

			# our player steps first
			board, reward, done, info = self._step(action,1)

			if verbose:
				env.render_pretty()

			if not done:
				# opponent turn
				#action = self.opponent_agent.act(self.board, reward, done)
				action = self.opponent_agent.forward(self.board)
				if verbose:
					env.render_pretty()

				board, reward, done, info = self._step(action,-1)
				if verbose:
					env.render_pretty()

			return board, reward, done, info

		else:

			# opponent turn
			# action = self.opponent_agent.act(self.board, reward, done)
			action = self.opponent_agent.forward(self.board)
			board, reward, done, info = self._step(action, -1)

			return board, reward, done, info


	def win_no(self):
		self.done=True
		#print("LOSE")
		return -1

	def win_yes(self):
		self.done=True
		#print("WIN")
		return 1




class XosEnv(BinaryBoardEnv):
	'''
	Connect-4 board and rules.
	'''

	board = None
	wid = 3
	ht = 3
	winlen = 3

	side = 1
	done = False

	opponent_agent = None

	invalid = 0 # consecutive invalid moves

	def __init__(self):

		self.nA = [ self.wid * self.ht ]
		self.nS = [ self.wid * self.ht ]

		self.action_space = spaces.Discrete(self.nA[0])
		self.observation_space = spaces.MultiDiscrete(self.nS)

		self.seed()
		self.reset()


	def action_is_valid(self,action):
		#print(action)
		if self.done:
			print("No more moves, Game over")
			return False

		if action<0 or action>self.wid*self.ht-1:
			print('Invalid action, move location out of range')
			return False   #

		row = action // self.wid
		col = action - row * self.wid
		if self.board[row,col] != 0:
			#print("Invalid move, not empty")
			self.invalid+=1
			if self.invalid > 10:
				#print("MANY")
				pass
			return False

		self.invalid=0

		return True

	def _step(self, action, side):

		reward = 0.0
		if not self.action_is_valid(action):
			print('invalid action penalty %d'% action)

			# dont punish us for opponent's move
			if side == -1:
				reward=0.0
			else:
				reward=-5.0 * self.turn_side
			self.done = True
		else:
			row = action // self.wid
			col = action - row * self.wid

			self.board[row,col] = side
			#print(row,col)

			w=self.wincheck()
			if w==1:
				reward =1.0
				self.done=True
			elif w==-1:

				reward =-3.0
				self.done=True
			elif w==-100: # board is full
				reward = 0
				self.done=True

		# change sides
		self.turn_side=self.turn_side* -1
		self.lastaction = action

		info = {}

		return (self.board, reward, self.done, info )

	def wincheck(self):
		#print("sums:")
		rs = np.sum( self.board, axis=0)
		cs = np.sum( self.board, axis=1)
		#print(rs)
		#print(cs)

		# check rows
		if np.max(rs)==self.winlen:
			return self.win_yes()
		if np.min(rs)== -1*self.winlen:
			return self.win_no()

		# check columns
		if np.max(cs)==self.winlen:
			return self.win_yes()
		if np.min(cs)==-1*self.winlen:
			return self.win_no()

		# check diagonals
		d1 = np.sum( np.diag( self.board ) )
		d2 = np.sum( np.diag(np.fliplr(self.board)) )

		if np.sum(d1)==self.winlen:
			return self.win_yes()
		if np.sum(d1)==-1*self.winlen:
			return self.win_no()
		if np.sum(d2)==self.winlen:
			return self.win_yes()
		if np.sum(d2)==-1*self.winlen:
			return self.win_no()

		# check for board full
		all = np.sum( np.sum(  np.abs( self.board), axis=0) )
		#print('sum:',all)
		if all == self.wid * self.ht:
			#print("DRAW")
			self.done=True
			return -100

		#print("No win yet")
		return 0




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



def get_dqn_agent(side=1.0):
	'''
	prepare a fresh agent
	'''

	processor = DummyProcessor()
	input_shape = (3,3)

	input_x = Input(shape=(1,) + input_shape)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	intput_x_sidenorm  = Lambda(lambda x: x * side)(input_x)

	input_x_flat = Flatten()(intput_x_sidenorm)
	x = Dense(200)(input_x_flat)
	x= Activation('relu')(x)
	x = Dense(40)(x)
	x = Activation('relu')(x)
	x= keras.layers.concatenate([x,input_x_flat,input_x_flat]) # highway
	x = Dense(env.action_space.n)(x)
	predictions = Activation('linear')(x)
	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())

	# see https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py
	memory = SequentialMemory(limit=50000, window_length=1)
	policy = EpsGreedyQPolicy(0.005)
	dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=1000,
	    target_model_update=1000, policy=policy, enable_double_dqn=True)
	dqn.compile(Adam(lr=1e-4), metrics=['mae'])
	return dqn

'''
TRAIN_RANDOM: Learn against RandomAgent
TRAIN_SELF: Learn against static replicas of itself (actually, a ChaosDqnAgent)

Typical training regime is:

1) TRAIN_RANDOM until it gets to static performance, understands bad moves, gets started
2) TRAIN_SELF to work on deeper levels
3) HUMAN

'''

if __name__ == "__main__":


	regime = Regime()
	env = XosEnv()
	modelpath= '/tmp/tictactoh3x3_model3.hd5'
	regime.init(env,True,modelpath,get_dqn_agent,STEPS_FIT = 3000)

	# mode = "TRAIN_RANDOM"
	# mode = "TRAIN_SELF"
	mode = "HUMAN"

	if mode=="HUMAN":
		regime.test_human()

	elif mode=="TRAIN_RANDOM":
		regime.train_random()

	elif mode == "TRAIN_SELF":
		regime.train_self()
