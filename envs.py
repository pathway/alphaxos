from gym import Env, spaces
from gym.utils import seeding
from random import randint

import numpy as np

from params import *


REWARD_INVALID_ACTION = regime_params['reward_invalid_move']
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_TIE = 0.0



class Episodes:
	# ep=[]
	'''
	log:
	agent1  agent2  winner score moves_long

	summary:
		agent1 vs agent2 : games, wins, losses, ties, avf

	'''

	pair_stats = {}

	def _computed_stats_update(self, stat_dict):
		stat_dict['avg_moves'] = stat_dict['moves'] / stat_dict['games']
		stat_dict['avg_score'] = stat_dict['total_score'] / stat_dict['games']

		stat_dict['win_rate'] = stat_dict['wins'] / stat_dict['games']
		stat_dict['loss_rate'] = stat_dict['losses'] / stat_dict['games']
		stat_dict['tie_rate'] = stat_dict['ties'] / stat_dict['games']
		stat_dict['fail_rate'] = stat_dict['fails'] / stat_dict['games']
		stat_dict['invalid1_rate'] = stat_dict['invalid1'] / stat_dict['games']
		stat_dict['invalid2_rate'] = stat_dict['invalid2'] / stat_dict['games']

		return stat_dict

	def save_score(self, winner, score, moves_long, invalid_side):
		# saving whole record is wasteful. summarize.
		if not self.pair_stats:
			wins = 0
			losses = 0
			ties = 0
			total_score = score
			fails = 0
			inv1 = 0
			inv2 = 0

			if invalid_side != 0:
				if invalid_side == 1:
					inv1 = 1
				elif invalid_side == -1:
					inv2 = 1
				fails = 1
			else:
				if winner == 1:
					wins = 1
				elif winner == -1:
					losses = 1
				elif winner == 0:
					ties = 1
			pair_stat = {'games': 1, 'wins': wins, 'losses': losses, 'ties': ties, 'fails': fails,
			             'total_score': total_score, 'moves': moves_long, 'invalid1': inv1, 'invalid2': inv2}

			pair_stat = self._computed_stats_update(pair_stat)
			self.pair_stats = pair_stat
		else:
			pair_stat = self.pair_stats
			pair_stat['games'] += 1

			if invalid_side != 0:
				if invalid_side == 1:
					pair_stat['invalid1'] += 1
				elif invalid_side == -1:
					pair_stat['invalid2'] += 1
				pair_stat['fails'] += 1
			else:

				if winner == 1:
					pair_stat['wins'] += 1
				elif winner == -1:
					pair_stat['losses'] += 1
				elif winner == 0:
					pair_stat['ties'] += 1

			pair_stat['total_score'] += score
			pair_stat['moves'] += moves_long

			pair_stat = self._computed_stats_update(pair_stat)

			self.pair_stats = pair_stat
		return self.pair_stats

	def get_stats(self):
		return self.pair_stats

	def reset_stats(self):
		self.pair_stats = {}





class BinaryBoardEnv(Env):
	'''
	'''
	tumbler = False
	board = None
	done = False

	self_side=0
	turn_side=0

	total_reward=0


	def seed(self, seed=42):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def reset(self):
		board = None

		self.board = np.zeros((self.ht , self.wid))
		self.self_side=1
		self.turn_side=randint(0,1)*-2 + 1
		self.turn_count=0
		self.total_reward=0
		self.done=False

		# if opponent starts, then take first turn as part of reset
		# this simplifies step()
		if self.opponent_agent and self.turn_side==-1:
			# opponent starts

			# keras-rl ligo
			#action = self.opponent_agent.forward(self.board)

			# tensorforce lingo
			action = self.opponent_agent.act(self.board)
			board, reward, done, info = self._step(action, -1)
			self.turn_side=1
			self.turn_count+=1
			return self.obs() #, reward, done, info

		return self.obs()

	def obs(self):
		return self.board.flatten()



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
		print("----\n")

	def render(self,mode='human'):
		print(self.board)

	def execute(self,actions):
		state, step_reward, terminal, info = self.step(action=actions)
		return state, terminal, step_reward

	# full step, including "our" turn and opponent's turn
	def step(self, action, verbose=False):

		done=False

		if self.turn_side==1:

			# protagonist player steps first
			self.board, reward, done, info = self._step(action,1)

			if verbose:
				self.render_pretty()

			if not done:
				# opponent turn
				#action = self.opponent_agent.act(self.board, reward, done)
				# keras-rl ligo
				# action = self.opponent_agent.forward(self.board)

				# tensorforce lingo
				action = self.opponent_agent.act(self.board)

				self.board, reward, done, info = self._step(action,-1)
				if verbose:
					self.render_pretty()

			#if done:
			#	...

			#print(self.board)
			if self.tumbler:
				tmbl = random.randint(0,8)
				if tmbl==0:
					pass
				elif tmbl >= 4:
					# reflect
					self.board=np.flip(self.board, axis=0)
					tmbl =- 4

				if tmbl > 0:
					self.board = np.rot90(self.board, k=tmbl,axes=(0,1))

			return self.obs(), reward, done, info

		else:
			import pdb; pdb.set_trace()
			return None


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

	opponent_agent = None
	wid = 3
	ht = 3
	winlen = 3

	winner = None   # -1 = opponent_agent won, 1=regular "guest" agent won?

	side = 1

	invalid = 0 # consecutive invalid moves

	epl = Episodes()    # episode log.  env is not the cleanest place for it, but it works

	def __init__(self,opponent_agent=None):

		self.nA = [ self.wid * self.ht ]
		self.nS = [ self.wid * self.ht ]

		self.action_space = spaces.Discrete(self.nA[0])
		self.observation_space = spaces.MultiDiscrete(self.nS)

		self.seed()
		if opponent_agent:
			self.opponent_agent=opponent_agent
		self.reset()


	def action_is_valid(self,action):
		#print(action)
		if self.done:
			#print("No more moves, Game over")
			return False

		if action<0 or action>self.wid*self.ht-1:
			print('Invalid action, move location %d out of range' % action)
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
		invalid_side=0

		already_done = False
		if self.done == True:
			already_done=True
			pass

		if not self.action_is_valid(action):
			#print('invalid action penalty %d'% action)
			invalid_side = side

			# dont punish us for opponent's move
			if side == -1:
				reward=0.0
			else:
				reward=REWARD_INVALID_ACTION
			self.done = True
			self.winner = 0
		else:
			row = action // self.wid
			col = action - row * self.wid

			self.board[row,col] = side
			#print(row,col)

			w=self.wincheck()

			if 1: # enable winning? 0: no winning.  Just valid or invalid moves.
				if w==1:
					reward =REWARD_WIN
					self.winner=1
					self.done=True
				elif w==-1:
					self.winner=-1
					reward =REWARD_LOSE
					self.done=True
				elif w==-100: # board is full
					self.winner=0
					reward = REWARD_TIE
					self.done=True

		if not already_done:
			# count the turn as done
			self.total_reward += reward
			self.turn_count+=1

		# we just completed this episode?
		if not already_done and self.done:
			self.epl.save_score(self.winner,reward,self.turn_count,invalid_side)

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








class C4Env(BinaryBoardEnv):
	'''
	Connect-4 board and rules.
	'''

	board = None
	wid = 7
	ht = 6
	winlen = 4

	winner = None

	side = 1
	done = False

	win_diag = True
	win_vert = True
	win_horiz = True

	opponent_agent = None

	invalid = 0 # consecutive invalid moves

	epl = Episodes()    # episode log.  env is not the cleanest place for it, but it works

	def __init__(self,opponent_agent=None):

		self.nA = [ self.wid ]
		self.nS = [ self.wid * self.ht ]

		self.action_space = spaces.Discrete(self.nA[0])
		self.observation_space = spaces.MultiDiscrete(self.nS)

		self.seed()
		if opponent_agent:
			self.opponent_agent=opponent_agent
		self.reset()


	def action_is_valid(self,action):
		#print(action)
		if self.done:
			#print("No more moves, Game over")
			return False

		if action<0 or action>self.wid-1:
			print('Invalid action, move location out of range')
			return False   #

		if self.colfull(action):
			self.invalid+=1
			return False

		self.invalid=0
		return True


	def _step(self, action, side):

		reward = 0.0
		invalid_side=0

		already_done = False
		if self.done == True:
			already_done=True
			pass

		if not self.action_is_valid(action):
			#print('invalid action penalty %d'% action)
			invalid_side = side

			# dont punish us for opponent's move
			if side == -1:
				reward=0.0
			else:
				reward=REWARD_INVALID_ACTION
			self.done = True
			self.winner = 0
		else:
			# TODO:C4

			col = action
			for checkrow in range(0,self.ht):
				if self.board[self.ht-checkrow-1,col]==0:
					self.board[self.ht-checkrow-1,col] = side
					break

			#print(row,col)

			w=self.wincheck()

			if 1: # enable winning? 0: no winning.  Just valid or invalid moves.
				if w==1:
					reward =REWARD_WIN
					self.winner=1
					self.done=True
				elif w==-1:
					self.winner=-1
					reward =REWARD_LOSE
					self.done=True
				elif w==-100: # board is full
					self.winner=0
					reward = REWARD_TIE
					self.done=True

		if not already_done:
			# count the turn as done
			self.total_reward += reward
			self.turn_count+=1

		# we just completed this episode?
		if not already_done and self.done:
			self.epl.save_score(self.winner,reward,self.turn_count,invalid_side)

		# change sides
		self.turn_side=self.turn_side* -1
		self.lastaction = action

		info = {}

		return (self.board, reward, self.done, info )


	def colfull(self,col):
		c = self.board[:,col]
		if np.sum(np.absolute(c)) == self.ht:
			return True
		return False


	def _rowcheck(self,r):
		same=0
		elast=r[0]
		for ind in range(1,len(r)):
			el=r[ind]
			if el==elast and el!=0:
				same+=1
				#print(same)
			else:
				same=0
			if same>=self.winlen-1:
				return el
			elast = el
		return 0



	def wincheck(self):
		# check rows
		if self.win_horiz:
			for row in range(0,self.ht):
				d=self.board[row]
				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
					return self.win_no()

		# check columns
		if self.win_vert:
			for row in range(0,self.wid):
				d=self.board[:,row]
				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
					return self.win_no()

		print_diag=False

		# check diags
		if  self.win_vert:
			for diag_index in range(0,min(self.wid,self.ht) ):
				d=np.diag(self.board, diag_index)
				if print_diag:
					print(diag_index,d)

				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
					return self.win_no()
				d=np.diag(self.board, -1*diag_index)
				if print_diag:
					print(-1*diag_index,d)

				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
					return self.win_no()

			# check anti-diags
			bd=np.fliplr(self.board)
			for diag_index in range(0,min(self.wid,self.ht) ):
				d=np.diag(bd, diag_index)
				if print_diag:
					print(diag_index,d)

				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
					return self.win_no()
				d=np.diag(bd, -1*diag_index)
				if print_diag:
					print(-1*diag_index,d)

				cc= self._rowcheck(d)
				if cc==1:
					return self.win_yes()
				if cc==-1:
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
