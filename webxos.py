from flask import Flask
from flask import json
from flask import request

from alphaxos import *
from rlparts import *
from flask import render_template

#http://www.whatever-dude.com/wdimages/mikeposts/theo.jpg\

app = Flask(__name__)

regime=None

def webxos_reset(envtype):
	global board
	global regime
	global done
	global reward
	global info
	global op2
	global player

	regime = Regime()
	regime.init(env,True,get_dqn_agent,STEPS_FIT = regime_params['steps_per_iter'])
	op2 = 'random'
	done = None
	reward = 0
	info = {}
	player='human'

	board = regime.test_human_web_init(op2)


env = C4Env()
webxos_reset(env)



@app.route('/',methods=['GET', 'POST'])
def text_classification():
	global board
	global regime
	global done
	global reward
	global info
	global op2
	global player

	command = request.args.get('cmd',default=None,type=str)
	q_values = None

	if command == "reset":
		player_temp = request.args.get('op1', default=None, type=str)
		if player_temp:
			player=player_temp
		if not player:
			player='human'

		op2_temp = request.args.get('op2', default=None, type=str)
		if op2_temp:
			op2=op2_temp
		if not op2:
			op2='random'
		board = regime.test_human_web_init(op2) #''dqn_opponent')
		done = False
		reward = 0
	else:

		if player=='human':
			mv = request.args.get('move',default=None,type=int)
			print(mv)
		else:
			agent = regime.agency.find_agent_kind(player)
			mv = agent.forward(regime.env.obs() )

		if mv is not None and not done:
			board, reward, done, info = regime.test_human_web_act(mv)

	board_items=[]
	for col in range(0,env.ht):
		for row in range(0, env.wid):
			cell=board[col][row]
			board_items.append(cell)
	print(board)

	chars={
		'human': {'name':'Hooman','icon_url':'https://image.freepik.com/free-vector/caveman-computer_6460-47.jpg'},
		'random': {'name':'Twiki (Random)','icon_url': 'https://pbs.twimg.com/profile_images/586154475460685825/yNGfbiTP_400x400.jpg'},  # 'http://www.starwarshelmets.com/2007/ANH_HD_3po10.jpg'
		'chaos': {'name':'R2D2 (DQN+Random)','icon_url':'https://www.model-space.com/media/catalog/product/cache/2/thumbnail/1280x/9df78eab33525d08d6e5fb8d27136e95/r/2/r2d2.jpg'},
		'dqn_opponent': {'name':'Theo (DQN)','icon_url':'https://vignette.wikia.nocookie.net/buckrogers/images/1/1b/Theo.jpg/revision/latest?cb=20111026160842'},
		'wrapper_opponent': {'name':'Wrapped DQN opponent','icon_url':'http://www.mascotdesigngallery.com/wp/wp-content/uploads/2013/07/Mascot_Robot___Finished_by_MattWNelson.jpg'},
		'countchocula': {'name':'The Count','icon_url':'https://c1.staticflickr.com/9/8233/8539517000_9a44748db4_b.jpg'},
	}

	char = chars[op2]
	player_self=chars[player]
	if 'q_values' not in info or info['q_values'] is None:
		nq=env.action_space.n
		info['q_values']=[0,] * nq
	print(info['q_values'])

	#board_template='board.html'
	board_template='c4.html'

	return render_template(board_template, board=board_items,done=done,reward=reward,info=info,op=op2,player=player,player_self=player_self,char=char,q_values=info['q_values'],sqsz=100,wrapcount=env.wid)

