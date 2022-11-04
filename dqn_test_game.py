from GameEnvironment.connectFour import ConnectFour
from Agents.randomAgent import RandomAgent
from keras.models import load_model
import numpy as np

def get_player_selection():
    while True:
        try:
            selection = input('Select column [0-6]: ')
            selection = int(selection)
            return selection
        except Exception as e:
            print(e)

def get_dqn_action(model, observation, available_actions):
    state = np.array([observation])
    state = np.expand_dims(state, axis=3)
    actions = model(state)
    action_scores = np.copy(actions)
    action = np.argmax(actions)
    # # Prevent choosing an invalid action
    while available_actions[action] == -1:
        action_scores = np.delete(action_scores, np.argmax(action_scores))
        next_largest = np.max(action_scores)
        action = np.where(actions[0] == next_largest)[0][0]
    return action

'''
This code file gives an example of how to use the ConnectFour class to allow 2 agents to play against each other
'''
if __name__ == '__main__':
    # Play an infinite number of games: End with ctrl + c
    print('Loading DQN agent...')
    dqn_model = load_model('./Agents/DQN/dqn_c4_model_2.h5')
    print('DQN Agent loaded!')
    while True:
        connect_four = ConnectFour(1) # Intitalise game with AGENT as player 1
        # connect_four = ConnectFour(2) # Intitialise game with AGENT as player 2
        random_agent = RandomAgent()
        game_result = 0
        while not connect_four.is_done:
            available_actions = connect_four.get_available_actions() # A value of -1 indicates a column that is not valid
            turn_number = connect_four.get_turn_number()
            game_state = connect_four.get_state()   # 2D numpy array: 1 = player 1 tokens, 2 = player 2 tokens
            # Is HUMAN turn
            if turn_number % 2 == 1:
                print('YOUR TURN')
                connect_four.print_board()
                print(f'Available columns: {str(available_actions)}')
                selection = get_player_selection()
                game_result = connect_four.play_turn(selection)
            # Is AI turn
            elif turn_number % 2 == 0:
                # selection = random_agent.select_random_column(available_actions)
                selection = get_dqn_action(dqn_model, connect_four.get_state(), connect_four.available_actions)
                game_result = connect_four.play_turn(selection)
                print(f'AI TURN OVER - Selected column: {str(selection)}')
        print(f'[Game Finished] Result for Agent: {str(game_result)}')
        print()

