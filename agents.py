
from params import *

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
from keras.layers import Dense, Input, GlobalMaxPooling1D, InputLayer, Conv2D, Conv1D, Reshape,MaxPooling1D,  MaxPooling2D, BatchNormalization, Subtract
from keras import backend as K


import numpy as np
import random




class BaseAgent(object):
	agent_label=str(random.randint(1000000,9999999))

	def reload(self):
		pass

	def save(self):
		pass


class RandomAgent(BaseAgent):
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
		rm = random.randint(0, self.action_space.n - 1)
		while not self.env.action_is_valid(rm):
			rm = random.randint(0, self.action_space.n - 1)

		return rm



class CountChoculaAgent(BaseAgent):
	'''
	Always pick the first valid action (starting from smallest)
	'''

	env = None

	def __init__(self, action_space):
		self.action_space = action_space

	# print(self.action_space)

	def forward(self, observation):
		return self.act(observation, None, None)

	def act(self, observation, reward, done):
		for i in range(0, self.action_space.n):
			if self.env.action_is_valid(i):
				return i




class HumanAgent(BaseAgent):
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



class WrapperAgent(BaseAgent):
	'''
	It wraps "smart_agent" (eg. a DQN or other agent).
	smart_agent should have a load_weights()
	'''

	smart_agent = None
	env = None
	modelfile = None

	def __init__(self, smart_agent, action_space, random_ratio=0.03):
		self.action_space = action_space
		self.smart_agent = smart_agent

	def reload(self):
		self.smart_agent.reload()

	def save(self):
		self.smart_agent.save()

	def load_replay(self, memoryfile):
		pass

	def load_from_disk(model, agent_info):
		path = '/SRC/pathway/alphaxos/'
		modelfile = path + agent_info['modelfile']

		memorypath = path + agent_info['memoryfile']
		load_memory(model, memorypath)
		print('loaded: ', memorypath)

	# print(self.action_space)

	def forward(self, observation):
		return self.act(observation, None, None)

	def act(self, observation, reward, done):
		a = self.smart_agent.forward(observation)
		self.q_values = self.smart_agent.q_values
		return a




class ChaosDqnAgent(BaseAgent):
	'''
	This agent does Epsilon-greedy-ish rollouts.

	The purpose is to ensure that competing DQNs do not get trapped in mutual local minima.
	Adding a random element prevents competely closing off any state pathways.

	Maybe more importantly, it also is an example of composing agents from others.

	It wraps "smart_agent" (eg. a DQN or other agent).

	Note it is not precisely Epsilon-Greedy, but rather Valid-Epsilon-Greedy,
	its choice of random move are limited to valid moves given the current board state.
	'''

	smart_agent = None

	env = None

	def __init__(self, smart_agent, action_space, random_ratio=0.03):
		self.action_space = action_space
		self.random_ratio = random_ratio
		self.smart_agent = smart_agent

	# print(self.action_space)

	def forward(self, observation):
		return self.act(observation, None, None)

	def act(self, observation, reward, done):
		if random.random() < self.random_ratio:
			return self.random_act(observation, None, None)
		else:
			a = self.smart_agent.forward(observation)
			self.q_values = self.smart_agent.q_values
			return a

	def random_act(self, observation, reward, done):
		rm = random.randint(0, self.action_space.n - 1)
		while not self.env.action_is_valid(rm):
			rm = random.randint(0, self.action_space.n - 1)
		return rm




class DeltaChaosDqnAgent(BaseAgent):
	'''
	This agent does Epsilon-greedy-ish rollouts.

	The purpose is to ensure that competing DQNs do not get trapped in mutual local minima.
	Adding a random element prevents competely closing off any state pathways.

	Maybe more importantly, it also is an example of composing agents from others.

	It wraps "smart_agent" (eg. a DQN or other agent).

	Note it is not precisely Epsilon-Greedy, but rather Valid-Epsilon-Greedy,
	its choice of random move are limited to valid moves given the current board state.
	'''

	smart_agent = None

	env = None

	def __init__(self, smart_agent, action_space, random_ratio=0.03, delta_window=0.1):
		self.action_space = action_space
		self.random_ratio = random_ratio
		self.delta_window = delta_window
		self.smart_agent = smart_agent

	# print(self.action_space)

	def forward(self, observation):
		return self.act(observation, None, None)

	def act(self, observation, reward, done):
		if random.random() < self.random_ratio:
			# implement epsilon exploration
			return self.random_act(observation, None, None)
		else:
			a = self.smart_agent.forward(observation)

			# implement delta exploration
			self.q_values = self.smart_agent.q_values
			qq = self.q_values

			# find best; limit to within self.delta_window of best
			maxq = np.max(qq)
			threshq = maxq - self.delta_window

			# sort q values
			allowed_mask = qq > threshq
			indexes = np.arange(len(qq))

			allowed_indexes = indexes[allowed_mask]

			selected_index = np.random.choice(allowed_indexes, 1)
			a = selected_index[0]

			# set action
			self.smart_agent.recent_action = a
			return a

	def random_act(self, observation, reward, done):
		rm = random.randint(0, self.action_space.n - 1)
		while not self.env.action_is_valid(rm):
			rm = random.randint(0, self.action_space.n - 1)
		return rm





class TensorForceAgent(BaseAgent):
	'''
	It wraps "smart_agent" tensorforce ag (eg. a DQN or other agent).
	smart_agent should have a load_weights()
	'''

	smart_agent = None
	env = None
	modelfile = None

	def __init__(self, smart_agent, action_space, random_ratio=0.03):
		self.action_space = action_space
		self.smart_agent = smart_agent

	def reload(self):
		self.smart_agent.reload()

	def save(self):
		self.smart_agent.save()

	def load_replay(self, memoryfile):
		pass

	def load_from_disk(model, agent_info):
		path = '/SRC/pathway/alphaxos/'
		modelfile = path + agent_info['modelfile']

		memorypath = path + agent_info['memoryfile']
		load_memory(model, memorypath)
		print('loaded: ', memorypath)

	# print(self.action_space)

	def forward(self, observation):
		return self.act(observation, None, None)

	def act(self, observation, reward, done):
		a = self.smart_agent.forward(observation)
		self.q_values = self.smart_agent.q_values
		return a










class Agency(object):

	agents={}

	def add_agent(self,agent_kind,agent):
		self.agents[(agent_kind,agent.agent_label)]=agent

	def find_agent(self,agent_kind,agent_label):
		return self.agents[(agent_kind,agent_label)]

	def find_agent_kind(self,agent_kind):
		l=self.list_agents()
		matches = [ (a[0],a[1])  for a in l if a[0]==agent_kind ]
		if not matches:
			return None
		return self.find_agent(matches[0][0],matches[0][1])

	def list_agents(self):
		k = self.agents.keys()
		return k











def get_qfunction_approximator_xos0(out_width,side_normalization_factor):

	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)
	x = Reshape( ( 3,3,1) )(input_x)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model





def get_qfunction_approximator_xos(out_width,side_normalization_factor):
	'''
	prepare a fresh agent.
	side=-1: if you are opponent (within the env)
	opponent cannot learn.
	'''

	input_shape = (3,3)

	#input_x = Input(shape=(1,) + input_shape)
	input_x = Input(shape= (1,) + input_shape)
	#x = input_x
	x = Reshape( ( 3,3,1) )(input_x)

	#x = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x)
	#x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
	#x = MaxPooling2D(pool_size=(2, 2) )(x)
	#x = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x)
	#x = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(x)
	input_x_sidenorm  = Lambda(lambda z: z * side_normalization_factor)(x)
	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm)

	x = Dense(500)(input_x_sidenorm)
	x = BatchNormalization()(x)
	x= Activation('relu')(x)
	x = Dense(100)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x= keras.layers.concatenate([x,input_x_sidenorm_square]) # highway

	x = Dense(out_width)(x)
	#pre_predictions = Activation('linear')(x)
	predictions = Activation('linear')(x)

	#subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1], output_shape=lambda shapes: shapes[0])
	#predictions = Subtract()( [pre_predictions,input_x_sidenorm_square] )

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model


def get_qfunction_approximator_xos2(out_width,side_normalization_factor):
	'''
	prepare a fresh agent.
	side=-1: if you are opponent (within the env)
	opponent cannot learn.
	'''

	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)
	x = Reshape( ( 3,3,1) )(input_x)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(x)
	input_x_sidenorm  = Lambda(lambda z: z * side_normalization_factor)(x)

	x = Dense(500)(input_x_sidenorm)
	x = BatchNormalization()(x)
	x= Activation('relu')(x)
	x = Dense(100)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	#subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1], output_shape=lambda shapes: shapes[0])
	#predictions = Subtract()( [pre_predictions,input_x_sidenorm_square] )

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model




def get_qfunction_approximator_xos3(out_width,side_normalization_factor):
	'''
	prepare a fresh agent.
	side=-1: if you are opponent (within the env)
	opponent cannot learn.
	'''

	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)
	x = Reshape( ( 3,3,1) )(input_x)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(x)
	input_x_sidenorm  = Lambda(lambda z: z * side_normalization_factor)(x)

	x = Dense(81)(input_x_sidenorm)
	x = BatchNormalization()(x)
	x= Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	#subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1], output_shape=lambda shapes: shapes[0])
	#predictions = Subtract()( [pre_predictions,input_x_sidenorm_square] )

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model



def get_qfunction_approximator_xos4(out_width,side_normalization_factor):
	'''
	prepare a fresh agent.
	side=-1: if you are opponent (within the env)
	opponent cannot learn.
	'''

	input_shape = (3,3)

	#input_x = Input(shape=(1,) + input_shape)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	#x = input_x
	x = Reshape( ( 3,3,1) )(input_x_sidenorm )

	x = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2) )(x)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(x)

	x = Dense(27)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x= keras.layers.concatenate([x,input_x_sidenorm_square]) # highway

	x = Dense(out_width)(x)
	#pre_predictions = Activation('linear')(x)
	predictions = Activation('linear')(x)

	#subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1], output_shape=lambda shapes: shapes[0])
	#predictions = Subtract()( [pre_predictions,input_x_sidenorm_square] )

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model






def get_qfunction_approximator_xos5(out_width,side_normalization_factor):
	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	x = Reshape( ( 3,3,1) )(input_x_sidenorm )
	x = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu')(x)
	x = Flatten()(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(27)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x= keras.layers.concatenate([x,input_x_sidenorm_square]) # highway

	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model




def get_qfunction_approximator_xos6(out_width,side_normalization_factor):
	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	x = Reshape( ( 3,3,1) )(input_x_sidenorm )
	x1 = Conv2D(9, kernel_size=(3, 1), strides=(1, 1), activation='relu')(x)
	x1 = Flatten()(x1)

	x2 = Conv2D(9, kernel_size=(1, 3), strides=(1, 1), activation='relu')(x)
	x2 = Flatten()(x2)

	x= keras.layers.concatenate([x1,x2,input_x_sidenorm_flat])

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(27)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x= keras.layers.concatenate([x,input_x_sidenorm_square]) # highway

	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model



def get_qfunction_approximator_xos7(out_width,side_normalization_factor):
	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	flatten = Reshape( (9,1) )(input_x_sidenorm_flat)

	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	x1 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(flatten ))
	x2 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=2, activation='relu')(flatten ))
	x3 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=3, activation='relu')(flatten ))
	x4 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=4, activation='relu')(flatten ))

	x= keras.layers.concatenate([x1,x2,x3,x4]) # ,flatten ?
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model




def get_qfunction_approximator_xos8(out_width,side_normalization_factor):
	input_shape = (3,3)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	flatten = Reshape( (9,1) )(input_x_sidenorm_flat)

	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	x1 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=1, activation='relu')(flatten ))
	x2 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=2, activation='relu')(flatten ))
	x3 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=3, activation='relu')(flatten ))
	x4 = Flatten()(Conv1D(9, kernel_size=3, strides=1, dilation_rate=4, activation='relu')(flatten ))

	x= keras.layers.concatenate([x1,x2,x3,x4,input_x_sidenorm_flat])
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model









def get_qfunction_approximator_c4_1(out_width,side_normalization_factor):

	input_shape = (6,7)
	input_x = Input(shape= (1,) + input_shape)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.
	x = Flatten()(input_x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model





def get_qfunction_approximator_c4_2(out_width,side_normalization_factor):

	input_shape = (6,7)
	input_x = Input(shape= (1,) + input_shape)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.

	x = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu')(input_x)
	x = Flatten()(x)
	x = Dense(27)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model



def get_qfunction_approximator_c4_3(out_width,side_normalization_factor):

	input_shape = (6,7)
	input_x = Input(shape= (1,) + input_shape)

	# When instantiating agent network, multiply board
	# by -1 or +1 depending on which side agent is playing.
	# This allows agent to otherwise be ambivalent to side.

	x = Reshape( ( 6,7,1) )(input_x )

	x = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), activation='relu')(x)
	x = MaxPooling2D(pool_size=(1,1) )(x)
	x = Flatten()(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model




def get_qfunction_approximator_c4_4(out_width,side_normalization_factor):
	input_shape = (6,7)
	input_x = Input(shape= (1,) + input_shape)

	# skip connection
	input_x_sidenorm = Lambda(lambda z: z * side_normalization_factor)(input_x)
	input_x_sidenorm_flat  = Flatten()(input_x_sidenorm )
	flatten = Reshape( (42,1) )(input_x_sidenorm_flat)

	input_x_sidenorm_square = Lambda(lambda z: K.square(z))(input_x_sidenorm_flat)

	conv_count = 27
	x1 = Flatten()( MaxPooling1D(pool_size=(1) )(Conv1D(conv_count, kernel_size=4, strides=1, dilation_rate=1, activation='relu')(flatten )) )
	x2 = Flatten()(MaxPooling1D(pool_size=(1) )(Conv1D(conv_count, kernel_size=4, strides=1, dilation_rate=6, activation='relu')(flatten )) )
	x3 = Flatten()(MaxPooling1D(pool_size=(1) )(Conv1D(conv_count, kernel_size=4, strides=1, dilation_rate=7, activation='relu')(flatten )) )
	x4 = Flatten()(MaxPooling1D(pool_size=(1) )(Conv1D(conv_count, kernel_size=4, strides=1, dilation_rate=8, activation='relu')(flatten )) )

	x= keras.layers.concatenate([x1,x2,x3,x4,input_x_sidenorm_flat])
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dense(out_width)(x)
	predictions = Activation('linear')(x)

	model = keras.models.Model(inputs=input_x, outputs=predictions )

	print(model.summary())
	return model





dqn_agents ={
	'dqn0':{ 'qfn': get_qfunction_approximator_xos0, 'modelfile': 'xos33_dqn0.hd5', 'memoryfile': 'xos33_dqn0.mem', },
	'dqn2':{ 'qfn': get_qfunction_approximator_xos2, 'modelfile': 'xos33_dqn2.hd5', 'memoryfile': 'xos33_dqn2.mem', },
	'dqn3': {'qfn': get_qfunction_approximator_xos3, 'modelfile': 'xos33_dqn3.hd5', 'memoryfile': 'xos33_dqn3.mem', },
	'dqn4': {'qfn': get_qfunction_approximator_xos4, 'modelfile': 'xos33_dqn4.hd5', 'memoryfile': 'xos33_dqn4.mem', },
	'dqn5': {'qfn': get_qfunction_approximator_xos5, 'modelfile': 'xos33_dqn5.hd5', 'memoryfile': 'xos33_dqn5.mem', },
	'dqn6': {'qfn': get_qfunction_approximator_xos6, 'modelfile': 'xos33_dqn6.hd5', 'memoryfile': 'xos33_dqn6.mem', },
	'dqn7': {'qfn': get_qfunction_approximator_xos7, 'modelfile': 'xos33_dqn7.hd5', 'memoryfile': 'xos33_dqn7.mem', },
	'dqn8': {'qfn': get_qfunction_approximator_xos8, 'modelfile': 'xos33_dqn8.hd5', 'memoryfile': 'xos33_dqn8.mem', },
	#'dqn5': {'qfn': get_qfunction_approximator_xos5, 'modelfile': 'xos33_dqn5.hd5', 'memoryfile': 'xos33_dqn5.mem', },

	'c4_dqn1': {'qfn': get_qfunction_approximator_c4_1, 'modelfile': 'c4_dqn1.hd5', 'memoryfile': 'c4_dqn1.mem', },
	'c4_dqn2': {'qfn': get_qfunction_approximator_c4_2, 'modelfile': 'c4_dqn2.hd5', 'memoryfile': 'c4_dqn2.mem', },
	'c4_dqn3': {'qfn': get_qfunction_approximator_c4_3, 'modelfile': 'c4_dqn3.hd5', 'memoryfile': 'c4_dqn3.mem', },
	'c4_dqn4': {'qfn': get_qfunction_approximator_c4_4, 'modelfile': 'c4_dqn4.hd5', 'memoryfile': 'c4_dqn4.mem', },

}




def get_dqn_agent(env,dqn_agent_subtype,folder='/SRC/pathway/alphaxos/models/',load=False,load_mem=True,side_normalization_factor=1.0):

	agent_info = dqn_agents[dqn_agent_subtype]

	model = agent_info['qfn'](out_width=env.action_space.n,side_normalization_factor=side_normalization_factor)

	# see https://github.com/keras-rl/keras-rl/blob/master/examples/duel_dqn_cartpole.py
	memory = SequentialMemory(limit=100000, window_length=1)

	'''
	test_policy = ValidGreedyQPolicy()
	test_policy = ValidGreedyQPolicy()
	test_policy.env=env
	policy = ValidEpsGreedyQPolicy(0.1)
	policy.env=env
	'''

	policy = EpsGreedyQPolicy(regime_params['epsilon-train'])
	#policy = ValidEpsGreedyQPolicy(0.1)
	policy.env=env

	test_policy=None

	dqn = DQNAgent(model=model, batch_size=regime_params['memory_batch_size'], gamma=regime_params['gamma'], nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=regime_params['steps_warmup'],
	    target_model_update=regime_params['steps_target_model_update'], policy=policy, test_policy=test_policy, enable_double_dqn=True)
	dqn.compile(Adam(lr=regime_params['learning_rate']), metrics=['mae'])

	dqn.modelfile = folder+agent_info['modelfile']
	dqn.memoryfile = folder+agent_info['memoryfile']

	if load:
		dqn.reload()

	if load_mem:
		dqn.reload_memory()

	#dqn.env=env
	return dqn




