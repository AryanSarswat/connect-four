import torch
import math
import numpy as np
from gym import spaces
import gym
from tqdm import tqdm

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
        observation = self._get_canonical_board()
        
        return observation, reward, done, {}
    
    def get_valid_actions(self):
        return self.valid_actions
            
    def _isDone(self):
        return np.all(self.valid_actions == 0) or self._get_winner() != 0

    def _get_obs(self):
        return self.board
    
    def _get_canonical_board(self):
        p1 = np.where(self.board == 1, 1, 0)
        p2 = np.where(self.board == -1, 1, 0)
        player = np.ones(self.board_shape) * self.current_player
        stacked = np.stack([p1, p2, player], axis=2)
        return stacked
    
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
        
        return self._get_canonical_board()
    
    def render(self, mode='human', close=False):
        pass



from GeneticAlgorithm.geneticAgent import GeneticAgent
from Agents.randomAgent import RandomAgent
from Minimax.minimaxConnectFour import MinimaxConnectFour
from AlphaConnect.MCTS import MCTS
from AlphaConnect.AlphaConnect import AlphaConnect
from AlphaConnect.minimalConnectFour import Board
from QLearning.dqn import get_model
from mctsConnectFour import ConnectFour, MonteCarlo


class AlphaConnectAgent():
    def __init__(self, board, args, device):
        self.model = AlphaConnect()
        self.model.load_state_dict(torch.load('./AlphaConnect/latest_v1.pt'))
        self.model = self.model.to(device)
        self.mcts = MCTS(board, self.model, args, device)
        
        
    def get_action(self, board):
        action = self.mcts.get_best_action(board, self.model, 1)
        return action
    
class QLearningAgent():
    def __init__(self):
        self.model = get_model()
    
    def get_action(self, board, valid_moves):
        return self.model.act(board, 1, valid_moves)
    
class MinimaxAgent():
    def __init__(self):
        self.minimax = MinimaxConnectFour(1, 6)
        
    def get_action(self, board):
        value, move = self.minimax.minimax(board, 7, -math.inf, math.inf, True)
        return move

class MCTSAgent():
    def __init__(self, board):
        self.mcts = MonteCarlo(board)

    def get_action(self):
        return self.mcts.get_move()
    


FEATURE_WEIGHTS = [0.215, 0.948, 0.008, 0.411, 0.802, 0.897, 0.194, 0.109, 0.027, 0.449, 0.032, 0.954, 0.837] 

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {
        'batch_size' : 64,
        'num_simulations':25,
        'numIters':10,
        'numEps':10,
        'epochs':25,
        'checkpoint':'latest.pt',
        'lr' : 0.001,
    }
    
    
    NUM_TRIALS = 10
    
    P1_WIN = 0
    P2_WIN = 0
    
    game = Connect4Env()
    minimalistConnect4 = Board()
    mctsBoard = ConnectFour()
    
    P1 = QLearningAgent()
    P2 = MCTSAgent(mctsBoard)
    
    print(P1)
    print(P2)
    
    for i in tqdm(range(NUM_TRIALS), desc='Trials'):
        
        game = Connect4Env()
        minimalistConnect4 = Board()
        mctsBoard = ConnectFour()
        P2 = MCTSAgent(mctsBoard)
        
        while not minimalistConnect4.isDone():
            if game.current_player == 1:
                action = P1.get_action(game._get_canonical_board(), minimalistConnect4.column_heights)
            else:
                action = P2.get_action()
                
            game.step(action)
            minimalistConnect4 = minimalistConnect4.play_action(action)
            mctsBoard.play(action)
        
        if game._get_winner() == 1:
            P1_WIN += 1
        elif game._get_winner() == -1:
            P2_WIN += 1
    
    print(f"P1 Wins: {P1_WIN}")
    print(f"P2 Wins: {P2_WIN}")
    print(f"Draws: {(NUM_TRIALS - P1_WIN - P2_WIN)}", )
        
        
        

