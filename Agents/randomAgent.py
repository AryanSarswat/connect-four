import random

'''
The RandomAgent class is one which randomly chooses a column to 
set their token for the connect 4 game
'''
class RandomAgent:
    def get_action(self, available_actions):
        column_number = 0
        available_columns = []
        for action in available_actions:
            if action != -1:
                available_columns.append(column_number)
            column_number += 1
        rand_idx = random.randrange(len(available_columns))
        return available_columns[rand_idx]