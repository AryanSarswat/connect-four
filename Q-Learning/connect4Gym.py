import gym
from gym import spaces
import numpy as np

class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}
    
    WIN_REWARD = 1
    LOSS_REWARD = -1
    DRAW_REWARD = 0
    
    def __init__(self, board_shape=(6,7)):
        super(Connect4Env, self).__init__()
        
        self.board_shape = board_shape
        self.action_space = spaces.Discrete(board_shape[1])
        self.observation_space = spaces.Box(low=-1, high=1, shape=board_shape, dtype=np.int8)
        
        self.current_player = 1
        self.valid_actions = np.ones(board_shape[1], dtype=np.int8)
        self.heights = np.ones(board_shape[1], dtype=np.int8) * (board_shape[0] - 1)
        self.board = np.zeros(board_shape, dtype=np.int8)
        self.done = False
        
        
    def step(self, action):
        if self.valid_actions[action] == 0:
            raise ValueError("Invalid action")
        
        self.board[self.heights[action], action] = self.current_player
        
        self.heights[action] -= 1
        if self.heights[action] < 0:
            self.valid_actions[action] = 0
            
        if self._isDone():
            done = 1
        else:
            done = 0
            self.change_player()
            
        reward = self._get_reward()
        observation = self._get_obs()
        
        return observation, reward, done, {}
            
    def _isDone(self):
        return np.all(self.valid_actions == 0) or self._get_winner() != 0

    def _get_obs(self):
        return self.board
    
    def _get_winner(self):
        if self.check_horizontal(self.current_player) or self.check_vertical(self.current_player) or self.check_diagonals(self.current_player):
            return self.current_player
        return 0
    
    def _get_reward(self):
        if self._isDone():
            winner = self._get_winner()
            if winner == 1:
                return self.WIN_REWARD
            elif winner == -1:
                return self.LOSS_REWARD
            else:
                return self.DRAW_REWARD
        else:
            return 0
        
    def check_horizontal(self, player):
        for i in range(6):
            for j in range(4):
                if np.all(self.board[i, j:j+4] == player):
                    return True
        return False
    
    def check_vertical(self, player):
        for i in range(3):
            for j in range(7):
                if np.all(self.board[i:i+4, j] == player):
                    return True
        return False
    
    def check_diagonals(self, player):
        for i in range(3):
            for j in range(4):
                if np.all(self.board[i:i+4, j:j+4].diagonal() == player):
                    return True
        for i in range(3):
            for j in range(3, 7):
                if np.all(np.fliplr(self.board[i:i+4, j-3:j+1]).diagonal() == player):
                    return True
        return False
        
    def change_player(self):
        self.current_player *= -1
    
    def reset(self):
        self.current_player = 1
        self.valid_actions = np.ones(self.board_shape[1], dtype=np.int8)
        self.heights = np.ones(self.board_shape[1], dtype=np.int8) * (self.board_shape[0] - 1)
        self.board = np.zeros(self.board_shape, dtype=np.int8)
        self.done = False
        
        return self.board
    
    def render(self, mode='human', close=False):
        pass
        