import numpy as np
import math
import torch

class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = dict()
        self.state = None
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = np.array([action for action in self.children.keys()])
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action
    
    def select_child(self):
        best_score = -float('inf')
        best_action = -1
        best_child = None
        for action, child in self.children.items():
            score = ucb_score(self, child, 1)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def expand(self, state, to_play, action_probs):
        self.to_play = to_play
        self.state = state
        for a, p in enumerate(action_probs):
            if p != 0:
                self.children[a] = Node(prior=p, to_play=to_play*-1)
                
    def __repr__(self):
        prior = "{:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state, prior, self.visit_count, self.value())
        
def ucb_score(parent, child, c):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    
    if child.visit_count > 0:
        value_score = -child.value()
    else:
        value_score = 0
    
    return value_score + c * prior_score

class MCTS:
    def __init__(self, game, model, args, device):
        self.game = game
        self.model = model
        self.args = args
        self.device = device
        
    def run(self, model, state, to_play):
        root = Node(0, to_play)

        fwdPass = torch.FloatTensor(state.get_canonical_form()).to(self.device)
        if model != None:
            action_probs, value = model(fwdPass)
            action_probs = action_probs.detach().cpu().numpy()[0]
            valid_moves = state.get_valid_moves_mask()
            action_probs = action_probs * valid_moves
            action_probs /= action_probs.sum()
            root.expand(state, to_play, action_probs)
        else:
            actions_probs = np.ones(self.game.action_size) * state.get_valid_moves_mask()
            actions_probs /= actions_probs.sum()
            root.expand(state, to_play, actions_probs)
        
        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]
            
            # Select
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
                
            parent = search_path[-2]
            state = parent.state
            
            next_state = state.play_action(action)
            
            value = next_state.get_score()
            
            if value is None:
                if self.model != None:
                    fwdPass = torch.FloatTensor(next_state.get_canonical_form()).to(self.device)
                    action_probs, value = model(fwdPass)
                    valid_moves = next_state.get_valid_moves_mask()
                    action_probs = action_probs.detach().cpu().numpy()[0]
                    action_probs = action_probs * valid_moves
                    action_probs /= action_probs.sum()
                    node.expand(next_state, parent.to_play * -1, action_probs)
                else:
                    action_probs = np.ones(self.game.action_size) * next_state.get_valid_moves_mask()
                    action_probs /= action_probs.sum()
                    node.expand(next_state, parent.to_play * -1, action_probs)
            self.backpropagate(search_path, value, parent.to_play * -1)
            
        return root
    
    def backpropagate(self, search_path, value, to_play):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            
    def get_best_action(self, board, model, to_play):
        root = self.run(model, board, to_play)
        return root.select_child()[0]
            
            
                
