from dqn import get_model
import os
import torch
from connect4Gym import Connect4Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')

if __name__ == "__main__":
    model = get_model()
    
    env = Connect4Env()
    
    env.reset()
    
    while not env._isDone():
        print(env._get_obs())
        if env.current_player == 1:
            action = model.act(env._get_canonical_board(), 1, env.get_valid_actions())
        else:
            action = int(input("Enter action: "))
        env.step(action)
        
    print(f"The winner is {env._get_winner()}")