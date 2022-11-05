from minimalConnectFour import Board
from AlphaConnect import AlphaConnect
from trainer import Trainer
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {
    'batch_size' : 256,
    'num_simulations':100,
    'numIters':50,
    'numEps':25,
    'epochs':100,
    'checkpoint':'latest_v5.pt',
    'lr' : 0.001,
}

game = Board()
model = AlphaConnect().to(device)

trainer = Trainer(game, model, args, device)
trainer.learn()