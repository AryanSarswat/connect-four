import math
import numpy as np
from pandas import *

class MinimaxConnectFour:
    def __init__(self, agent_player_number, search_depth=4, height=6, width=7):
        self.row_count = height
        self.column_count = width
        self.window_length = 4
        self.game_state = np.zeros((height,width))
        self.game_state = self.game_state.astype(int)
        self.search_depth = search_depth
        self.agent_player_number = agent_player_number  # Determines if agent will be player 1 or 2
        self.opponent_player_number = 0
        if self.agent_player_number == 1:
            self.opponent_player_number = 2
        else:
            self.opponent_player_number = 1
        self.game_reward = 0
        # Start game
        #self.play_game()

    def print_board(self):
        print('------ GAME STATE ------')
        print(DataFrame(self.game_state))

    def play_game(self):
        # Reset game
        self.game_state = np.zeros((self.row_count,self.column_count))
        self.game_state = self.game_state.astype(int)
        current_player_turn = 1
        is_over = False
        print('----------- CONNECT FOUR -----------')
        print('[NEW GAME] Starting a new connect 4 game!')
        print(f'Player Number: {str(self.opponent_player_number)}, AI Number: {str(self.agent_player_number)}')
        print(f'[SETUP INFO] Minimax search depth - {str(self.search_depth)} levels')
        print()
        self.print_board()
        while not is_over:
            available_actions = self.get_available_actions(self.game_state)
            selected_column = 0
            if current_player_turn == self.agent_player_number:
                value, selected_column = self.minimax(self.game_state, self.search_depth, -math.inf, math.inf, True)
                print(f'[AI Turn] - AI selected column number: {str(selected_column)}')
                print(f'[AI Turn Value] {str(value)}') 
            else:
                print(f'[Player turn] - Your token number: {str(self.opponent_player_number)}')
                print(available_actions)
                try:
                    selected_column = input('Select column: ')
                    selected_column = int(selected_column)
                except Exception as e:
                    print(e)
                    continue
            row = available_actions[selected_column]
            if row == -1:
                print('[FATAL ERROR] Invalid column selected! - Please try again')
                continue
            self.drop_piece(self.game_state, row, selected_column, current_player_turn)
            self.print_board()
            new_available_actions = self.get_available_actions(self.game_state)
            is_ai_win, is_player_win, is_draw = self.is_end_of_game(self.game_state, new_available_actions)

            if is_ai_win:
                print('[GAME OVER] AI has won!')
                is_over = True
            elif is_player_win:
                print('[GAME OVER] Player has won! Congratulations!')
                is_over = True
            elif is_draw:
                print('[GAME OVER] Draw!')
                is_over = True
            # Roll over players
            if current_player_turn == 1:
                current_player_turn = 2
            else:
                current_player_turn = 1
        return

    def get_available_actions(self, board):
        available_actions = []
        for column in range(0, self.column_count): # Number of columns
            available_row_index = -1
            for row in range(self.row_count - 1, -1, -1): # Decreasing in row
                if board[row][column] == 0:
                    available_row_index = row
                    break
            available_actions.append(available_row_index)
        return available_actions

    def score_window(self, window, player_token):
        score = 0
        opponent_token = self.agent_player_number
        if player_token == self.agent_player_number:
            opponent_token = self.opponent_player_number

        if window.count(player_token) == 4:
            score += 100
        elif window.count(player_token) == 3 and window.count(0) == 1:
            score += 10
        elif window.count(player_token) == 2 and window.count(0) == 2:
            score += 5

        # Blocking score
        if window.count(opponent_token) == 3 and window.count(0) == 1:
            score -= 80
        return score

    def calculate_board_score(self, board, player_token):
        score = 0
        # Prioritise centre
        center_array = [int(i) for i in list(board[:, self.column_count//2])]
        center_count = center_array.count(player_token)
        score += center_count * 10
        # Horizontal
        for row in range(self.row_count):
            row_array = [int(i) for i in list(board[row,:])]
            for column in range(self.column_count - 3):
                window = row_array[column:column + self.window_length]
                score += self.score_window(window, player_token)
        # Vertical
        for column in range(self.column_count):
            col_array = [int(i) for i in list(board[:, column])]
            for row in range(self.row_count - 3):
                window = col_array[row:row + self.window_length]
                score += self.score_window(window, player_token)
        # Diagonal 1
        for row in range(self.row_count - 3):
            for column in range(self.column_count - 3):
                window = [board[row + i][column + i] for i in range(self.window_length)]
                score += self.score_window(window, player_token)
        # Diagonal 2
        for row in range(self.row_count - 3):
            for column in range(self.column_count - 3):
                window = [board[row + 3 - i][column + i] for i in range(self.window_length)]
                score += self.score_window(window, player_token)
        return score

    def drop_piece(self, board, row, column, token):
        board[row][column] = token

    def check_if_winning_move(self, board, token):
        for c in range(self.column_count - 3):
            for r in range(self.row_count):
                if board[r][c] == token and board[r][c+1] == token and board[r][c+2] == token and board[r][c+3] == token:
                    return True
        for c in range(self.column_count):
            for r in range(self.row_count - 3):
                if board[r][c] == token and board[r+1][c] == token and board[r+2][c] == token and board[r+3][c] == token:
                    return True
        for c in range(self.column_count - 3):
            for r in range(self.row_count - 3):
                if board[r][c] == token and board[r+1][c+1] == token and board[r+2][c+2] == token and board[r+3][c+3] == token:
                    return True
        for c in range(self.column_count - 3):
            for r in range(3, self.row_count):
                if board[r][c] == token and board[r-1][c+1] == token and board[r-2][c+2] == token and board[r-3][c+3] == token:
                    return True
        return False

    def is_end_of_game(self, board, valid_actions):
        is_draw = True
        for action in valid_actions:
            if action > -1:
                is_draw = False
                break
        return (self.check_if_winning_move(board, self.agent_player_number), self.check_if_winning_move(board, self.opponent_player_number), is_draw)

    # Determines the minimax move
    def minimax(self, board, depth, alpha, beta, maximising_player):
        valid_actions = self.get_available_actions(board)
        # print(valid_actions)
        is_agent_win, is_opponent_win, is_draw = self.is_end_of_game(board, valid_actions)
        is_over = is_agent_win or is_opponent_win or is_draw
        # Terminal conditions
        if depth == 0 or is_over:
            if is_over:
                if is_agent_win:
                    # print('REACHED TERMINAL STATE AI WIN')
                    return (10000, None)
                elif is_opponent_win:
                    # print('REACHED TERMINAL STATE OPPONENT WIN')
                    return (-10000, None)
                else:
                    # print('REACHED TERMINAL STATE DRAW')
                    return (0, None) # Draw
            else:   # Depth of 0 - Return score of current board instead
                # print('MAX DEPTH REACHED')
                # print(board)
                score = self.calculate_board_score(board, self.agent_player_number)
                return (score, None)
        
        if maximising_player:
            value = -math.inf
            best_column = 0
            # For each position that is available in connect 4
            column_number = 0
            # print('MAX TURN')
            for open_row_index in valid_actions:
                if open_row_index == -1:    # Non valid column
                    column_number += 1
                    continue
                board_copy = board.copy()
                self.drop_piece(board_copy, open_row_index, column_number, self.agent_player_number)
                new_score = self.minimax(board_copy, depth-1, alpha, beta, False)[0]
                if new_score > value:
                    value = new_score
                    best_column = column_number
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
                column_number += 1
            return value, best_column
        else: # Minimising player
            value = math.inf
            best_column = 0
            # For each position that is available in connect 4
            column_number = 0
            # print('MIN TURN')
            for open_row_index in valid_actions:
                if open_row_index == -1:    # Non valid column
                    column_number += 1
                    continue
                board_copy = board.copy()
                self.drop_piece(board_copy, open_row_index, column_number, self.opponent_player_number)
                new_score = self.minimax(board_copy, depth-1, alpha, beta, True)[0]
                if new_score < value:
                    value = new_score
                    best_column = column_number
                beta = min(beta, value)
                if alpha >= beta:
                    break
                column_number += 1
            return value, best_column

if __name__ == '__main__':
    MinimaxConnectFour(1, 6)
    # Can play either as P1 or P2, 6-7 turn lookahead is possible. 8 is pushing it a bit