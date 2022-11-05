from minimalConnectFour import Board
from AlphaConnect import AlphaConnect
import torch
from MCTS import MCTS
import numpy as np

# Current best is latest_v1.pt

if __name__ == "__main__":
    board = Board()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = {
        'batch_size' : 64,
        'num_simulations':150,
        'numIters':10,
        'numEps':10,
        'epochs':25,
        'checkpoint':'latest.pt',
        'lr' : 0.001,
    }
    
    network = AlphaConnect().to(device)
    network.load_state_dict(torch.load('./latest_v1.pt'))
    network.eval()
    mcts = MCTS(board, network, args, device)
    
    turn = 1    
    while not board.isDone():
        print(board)
        if turn == 1:
            action = mcts.get_best_action(board, network, 1)
            board = board.play_action(action)
        else:
            move = int(input('Enter move: '))
            board = board.play_action(move)
        turn = -1 * turn
    print(board)