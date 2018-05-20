'''
# Xs and Os and Deep Reinforcement Learning
### Deep Q-Learning using an Open AI Gym-like Environment

Concise working example of Self-play via Deep Q-learning
Environment similar to gym/envs/toy_text/ in ai gym, but with two agents

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


from rlparts import  *
import random

from agents import *
from params import *
from envs import *



class HumanAgentXos(object):
	'''
	Take in keyboard input to select square.

	This agent is specific to the Xos environment and uses
	keypad numbering instead of direct enumeration
	'''
	agent_label="human-"+str(random.randint(1000000,9999999))

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

	if regime_params['env']=='xos':
		env = XosEnv()
	elif regime_params['env']=='c4':
		env = C4Env()
		env.win_vert=False
		env.win_diag=False
		env.win_horiz=True

	#modelpath= '/tmp/tictactoh3x3_model3.hd5' # pretty good player
	regime.init(env,True,get_dqn_agent,STEPS_FIT = regime_params['steps_per_iter'])

	#mode = "TRAIN_RANDOM"
	mode = "TRAIN_SELF"
	#mode = "HUMAN"
	#mode = "TEST"

	if mode=="HUMAN":
		regime.test_human()

	elif mode=="TRAIN_RANDOM":
		regime.train_random()

	elif mode == "TRAIN_SELF":
		regime.train_self()

	elif mode=="TEST":
		TEST_ROUNDS=regime_params['test_rounds']
		if 0:
			#regime.test_vs("random", "chaos", rounds=TEST_ROUNDS)
			regime.test_vs("chaos", "random", rounds=TEST_ROUNDS)
			#regime.test_vs("random", "chaos", rounds=TEST_ROUNDS)

		if 1:
			#ags=["random","countchocula","dqn_learner","dqn_opponent","wrapper_protagonist","wrapper_opponent","chaos"]
			ags=["random","countchocula","dqn_learner","wrapper_opponent","chaos","delta-chaos"]

			all=[]

			for a1 in ags:
				for a2 in ags:
					res = regime.test_vs(a1,a2,rounds=TEST_ROUNDS)
					print(res)

					all.append(res)

			print(all)
