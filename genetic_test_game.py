from GameEnvironment.connectFour import ConnectFour
from Agents.randomAgent import RandomAgent
from GeneticAlgorithm.geneticAgent import GeneticAgent

def get_player_selection():
    while True:
        try:
            selection = input('Select column [0-6]: ')
            selection = int(selection)
            return selection
        except Exception as e:
            print(e)

'''
This code file gives an example of how to use the ConnectFour class to allow 2 agents to play against each other
'''
if __name__ == '__main__':
    # Play an infinite number of games: End with ctrl + c
    feature_weights = [0.215, 0.948, 0.008, 0.411, 0.802, 0.897, 0.194, 0.109, 0.027, 0.449, 0.032, 0.954, 0.837]   # best
    # feature_weights = [0.271, 0.802, 0.086, 0.314, 0.072, 0.802, 0.604, 0.129, 0.246, 0.118, 0.049, 0.562, 0.944]
    # feature_weights = [0.638, 0.894, 0.198, 0.688, 0.723, 0.25, 0.408, 0.496, 0.286, 0.077, 0.063, 0.386, 0.69]
    # feature_weights = [0.368, 0.936, 0.134, 0.24, 0.205, 0.181, 0.724, 0.58, 0.486, 0.09, 0.149, 0.344, 0.82]
    # feature_weights = [0.772, 0.315, 0.023, 0.681, 0.205, 0.764, 0.955, 0.153, 0.189, 0.058, 0.228, 0.426, 0.585]
    agent_number = 1    # Set for agent to be player 1 or player 2
    opponent_number = 2
    if agent_number == 2:
        opponent_number = 1
    agent = GeneticAgent(agent_number, feature_weights)
    random_agent = RandomAgent()
    count = 0
    max_games = 1000
    accumulated_outcomes = []
    to_continue = True
    while count < max_games and to_continue:
        connect_four = ConnectFour(agent_number)
        connect_four.print_board()
        game_result = 0
        while not connect_four.is_done:
            available_actions = connect_four.get_available_actions() # A value of -1 indicates a column that is not valid
            turn = connect_four.player_turn
            game_state = connect_four.get_state()   # 2D numpy array: 1 = player 1 tokens, 2 = player 2 tokens

            if turn == agent_number:
                selection = agent.get_action(game_state, available_actions)           # Trained Genetic Agent
                print(f'[AI SELECTION] AI has selected column: {str(selection)}')
                game_result = connect_four.play_turn(selection)
            else:
                # --------- MOVE SELECTION --------- #
                print(f'[PLAYER TURN] Token number - {str(opponent_number)}')
                selection = get_player_selection()                                      # For human player
                # selection = random_agent.select_random_column(available_actions)      # Random agent
                # ----------- PLAY MOVE ----------- #
                game_result = connect_four.play_turn(selection)
            connect_four.print_board()
        print(f'[Game Finished] Result for Agent: {str(game_result)}')
        # if game_result == -1:
        #     to_continue = False
        count += 1
        accumulated_outcomes.append(game_result)
        print()
    print('ALL GAMES PLAYED - FINAL OUTCOME')
    num_wins = 0
    num_loss = 0
    num_draw = 0
    for outcome in accumulated_outcomes:
        if outcome == 1:
            num_wins += 1
        elif outcome == 0:
            num_draw += 1
        elif outcome == -1:
            num_loss += 1
    print(f'WINS: {str(num_wins)}')
    print(f'LOSS: {str(num_loss)}')
    print(f'DRAW: {str(num_draw)}')
            

