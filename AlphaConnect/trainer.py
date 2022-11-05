import os
import numpy as np
from random import shuffle
from tqdm import tqdm

import torch
import torch.optim as optim

from MCTS import MCTS
from minimalConnectFour import Board

class Trainer:
    
    def __init__(self, game, model, args, device):
        self.game = game
        self.model = model
        self.args = args
        self.device = device
        self.mcts = MCTS(game, model, args, device)
        
    def execute_episode(self):
        train_examples = []
        current_player = 1
        episode_step = 0 
        state = Board()
        
        while True:
            episode_step += 1
            
            self.mcts = MCTS(self.game, self.model, self.args, self.device)
            root = self.mcts.run(self.model, state, to_play=1)
            
            action_probs = [0] * 7
            for k, v in root.children.items():
                action_probs[k] = v.visit_count
                
            action_probs = np.array(action_probs)
            action_probs = action_probs / np.sum(action_probs)
            
            train_examples.append([state.get_canonical_form(), current_player, action_probs])
            
            action = root.select_action(temperature=0.5)
            state = state.play_action(action)
            reward = state.get_score()
            
            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_prob in train_examples:
                    ret.append([hist_state, hist_current_player, hist_action_prob, reward * ((-1) ** (hist_current_player != current_player))])
                
                return ret
            
    def learn(self):
        for i in tqdm(range(self.args['numIters']), desc="Num of Iterations"):
            train_examples =  []
            
            for eps in tqdm(range(self.args['numEps']), desc="Num episodes"):
                train_examples.extend(self.execute_episode())
            
            shuffle(train_examples)
            self.train(train_examples)
            self.save_checkpoint(folder='.', filename=self.args['checkpoint'])
            
    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=1e-5)
        
        pi_losses = []
        v_losses = []
        
        for epoch in range(self.args['epochs']):
            self.model.train()
            
            batch_idx = 0
            
            while batch_idx < int(len(examples)) / self.args['batch_size']:
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards = np.empty((self.args['batch_size'], 3, 6, 7))
                pis = np.empty((self.args['batch_size'], 7))
                vs = np.empty((self.args['batch_size'], 1))
                for ind, idx in enumerate(sample_ids):
                    boards[ind] = np.array(examples[idx][0])
                    pis[ind] = np.array(examples[idx][2])
                    vs[ind] = np.array(examples[idx][3])
                
                boards = torch.Tensor(boards).to(self.device)
                target_pis = torch.Tensor(pis).to(self.device)
                target_vs = torch.Tensor(vs).to(self.device)
                
                out_pi, out_v = self.model(boards)
                
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                
                total_loss = l_pi + l_v
                
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                batch_idx += 1
            
            print(f"Epoch {epoch + 1} / {self.args['epochs']}:")
            print(f"pi_loss: {np.mean(pi_losses)}")
            print(f"v_loss: {np.mean(v_losses)}")
            
    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1).mean()
        return loss
    
    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2)/targets.size()[0]
        return loss
    
    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)