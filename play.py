import numpy as np
import os

def state_to_idx(state):
	# Maps each state to an integer between 0 and 3^9-1.
	idx = 0
	for i in range(3):
		for j in range(3):
			idx += state[i,j]*3**(3*i + j)
	return idx

def idx_to_state(idx):
	# Reverse mapping of "state_to_idx".
	state = np.zeros((3,3), dtype=np.int32)
	for i in range(3):
		for j in range(3):
			state[i,j] = idx % 3
			idx = idx // 3
	return state

def check(state):
	# Returns 1 if player 1 is the winner, 2 if player 2 is the winner,
	# 3 if the match is a draw and 0 if the game is not over yet.

	# Check the rows:
	for i in range(3):
		if state[i,0] == 1 and state[i,1] == 1 and state[i,2] == 1:
			return 1 # Player 1 is the winner
		elif state[i,0] == 2 and state[i,1] == 2 and state[i,2] == 2:
			return 2 # Player 2 is the winner

	# Check the columns:
	for j in range(3):
		if state[0,j] == 1 and state[1,j] == 1 and state[2,j] == 1:
			return 1
		elif state[0,j] == 2 and state[1,j] == 2 and state[2,j] == 2:
			return 2

	# Check the diagonals:
	d1 = state[0,0] == 1 and state[1,1] == 1 and state[2,2] == 1
	d2 = state[0,0] == 2 and state[1,1] == 2 and state[2,2] == 2
	d3 = state[0,2] == 1 and state[1,1] == 1 and state[2,0] == 1
	d4 = state[0,2] == 2 and state[1,1] == 2 and state[2,0] == 2

	if d1 == True or d3 == True:
		return 1
	elif d2 == True or d4 == True:
		return 2

	# Check if draw:
	if np.any(state == 0):
		return 0 # Game not over
	else:
		return 3 # Draw

class Agent:
	def __init__(self, sym, eps=0.1, alpha=0.5, V=None):
		self.eps = eps
		self.alpha = alpha
		self.sym = sym
		self.state_history = []
		
		# Initialize V:
		if V is None:
			self.V = np.zeros(3**9)
			for i in range(3**9):
				state = idx_to_state(i)
				status = check(state)
				if status == sym:
					self.V[i] = 1.0
				elif status == 0:
					self.V[i] = 0.5
		else:
			self.V = V

	def get_possible_moves(self, env):
		# Returns the list of possible moves.
		possible_moves = []
		current_state = env.get_state()
		for i in range(3):
			for j in range(3):
				if current_state[i,j] == 0:
					possible_moves.append((i,j))
		return possible_moves

	def take_action(self, env, epsgreedy=True):
		# Choose an action based on epsilon-greedy strategy.
		r = np.random.rand()
		current_state = env.get_state()
		best_state = None
		possible_moves = self.get_possible_moves(env)
		if r < self.eps and epsgreedy==True:
			# Take a random action:
			idx = np.random.choice(len(possible_moves))
			next_move = possible_moves[idx]
		else:
			# Choose the best action based on current values of states:
			best_value = -1
			# Find the best next state value:
			next_move = None
			for move in possible_moves:
				current_state[move[0], move[1]] = self.sym
				val = self.V[state_to_idx(current_state)]
				if val > best_value:
					best_value = val
					next_move = move
				current_state[move[0], move[1]] = 0
		# Make the move:
		env.make_move(next_move[0], next_move[1], self.sym)

	def update_state_history(self, env):
		# NOTE => Only do it after an action is taken!
		self.state_history.append(state_to_idx(env.get_state()))

	def update(self, env):
		# Update the values:
		# NOTE => Only do it after an episode!
		reward = env.reward(self.sym)
		target = reward
		for prev in reversed(self.state_history):
			value = self.V[prev] + self.alpha*(target - self.V[prev])
			self.V[prev] = value
			target = value
		self.state_history = []

	def save_values(self, filename):
		np.savetxt(filename + '.csv', self.V, delimiter=',')

class Human:
	def __init__(self, sym):
		self.sym = sym

	def take_action(self, i, j, env):
		return env.make_move(i,j,self.sym)


class Environment:
	def __init__(self):
		self.board = np.zeros((3,3), dtype=np.int32)
		self.winner = None
		self.ended = False

	def reward(self, sym):
		if not self.game_ended():
			return 0

		if self.winner == sym:
			return 1
		elif self.winner == None:
			return 0.25
		else:
			return 0

	def make_move(self, i, j, sym):
		# Put "sym" at the (i,j)-th cell:
		if self.board[i,j] == 0:
			self.board[i,j] = sym
			return True
		else:
			print('Invalid move!')
			return False

	def get_state(self):
		# Returns a copy of the current state:
		return np.copy(self.board)

	def game_ended(self):
		# Checks if the game has ended: If yes, then update self.ended and
		# self.winner.
		status = check(self.board)
		if status != 0:
			self.ended = True
			if status != 3:
				self.winner = status
		return self.ended

	def display_board(self):
		# Displays the board nicely:
		symbols = [' ', 'x', 'o']
		hline = '------------------'
		print(hline)
		for i in range(3):
			line = '| '
			for j in range(3):
				line = line + symbols[self.board[i,j]] + '  |  '
			print(line)
			print(hline)


def training_iteration(p1, p2, env):
	current_player = None
	while not env.game_ended():
		# Alternate between players:
		if current_player == p1:
			current_player = p2
		else:
			current_player = p1
		# Take action:
		current_player.take_action(env)
		p1.update_state_history(env)
		p2.update_state_history(env)
	p1.update(env)
	p2.update(env)

def play():
	while True:
		first = input('Do you want to make the first move? [yes/no] ').lower()
		if first == 'yes' or 'no':
			break
		else:
			print('Invalid input!')
	
	if first == 'no':
		p1 = Agent(sym=1, V=np.genfromtxt('player1.csv', delimiter=','))
		p2 = Human(sym=2)
		current_player = p2
	else:
		p2 = Human(sym=1)
		p1 = Agent(sym=2, V=np.genfromtxt('player2.csv', delimiter=','))
		current_player = p1

	env = Environment()
	
	while not env.game_ended():
		if current_player == p1:
			current_player = p2
			while True:
				move = input('Enter coordinates i, j: ')
				i, j = move.split(',')
				i = int(i)
				j = int(j)
				if p2.take_action(i,j,env) == True:
					break
			env.display_board()
		else:
			current_player = p1
			# Take action:
			print('Computer\'s move:')
			p1.take_action(env, False)
			env.display_board()
	
	if env.winner == None:
		print('Draw!')
	elif env.winner == 1:
		print('You have been DEFEATED!')
	else:
		print('Victory!')


def train():
	# Train the agent:
	p1 = Agent(sym=1)
	p2 = Agent(sym=2)
	T = 20000 # No. of episodes
	for t in range(T):
		training_iteration(p1,p2, Environment())
	# Save the values learnt:
	p1.save_values('player1')
	p2.save_values('player2')

if __name__ == '__main__':
	if not os.path.exists('player1.csv') or not os.path.exists('player2.csv'):
		print('training ...')
		train()
		print('training complete!')
	play()
