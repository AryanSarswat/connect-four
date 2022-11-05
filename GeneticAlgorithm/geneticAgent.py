import math
import random

'''
The GeneticAgent class is an implementation for an agent to play connect4
The agent will choose moves by calculating the values of each column, and selecting the column with the highest value

Genetic algorithm will be used to determine how we weigh the importance of each feature, to generate the values of each action

player_number: integer, either 1 or 2
feature_weights: list/array of values of weights for each
'''
class GeneticAgent:
    def __init__(self, player_number, feature_weights, board_height=6, board_width=7):
        self.player_number = player_number          # Player number corresponds to board token
        self.opponent_number = 0
        # ------ Set Opponent Number ------ #
        if self.player_number == 1:
            self.opponent_number = 2
        elif self.player_number == 2:
            self.opponent_number = 1
        # ------ Genetically Created Feature Weights ------ #
        self.feature_weights = feature_weights
        # print(self.feature_weights)
        # ------ Board indexes ------ #
        self.bottom_index = board_height - 1        # Bottom index = largest number
        self.top_index = 0
        self.rightmost_index = board_width - 1
        self.leftmost_index = 0
        self.middle_column_index = math.floor(board_width / 2)      # Board should always be an odd number width

    # Calculate the scores for each column, weighted by the provided feature weights generated from genetic algorithm
    def calculateColumnScore(self, features_array):
        score = 0
        if self.feature_weights == None or len(self.feature_weights) != len(features_array):
            # print('[INVALID FEATURE WEIGHTS] Invalid feature weights detected. Generating list of weight 1')
            self.feature_weights = [1] * len(features_array)
        for i in range(len(features_array)):
            score += features_array[i] * self.feature_weights[i]
        return score
    
    # --------- Calculate score of each column, and select best --------- #
    def get_action(self, board, actions):
        column_number = 0
        column_scores = []
        for row_index in actions:
            if row_index == -1: # Invalid action, skip
                column_number += 1
                column_scores.append(-1)
                continue
            # Available actions are non -1
            features = []
            # Get number of own player tokens in the 7 different directions, without being blocked by the opponent token
            token_counts = self.get_tokens_for_directions(board=board, player_number=self.player_number, selected_column=column_number, selected_row=row_index)
            # ------- Check Features ------- #
            features.append(self.check_if_winning_move(board=board, selected_column=column_number, selected_row=row_index, player_number=self.player_number))     # Check if winning move available
            features.append(self.check_if_blocking_move(board=board, selected_column=column_number, selected_row=row_index))    # Check if token blocks opponent win
            features.append(self.check_if_row_of_two(token_counts=token_counts))    # Check if moves results in the creation of at least a row of 2
            features.append(self.check_if_row_of_three(token_counts=token_counts))  # Check if moves results in the creation of at least a row of 3
            features.append(self.creates_multiple_row_of_two(token_counts=token_counts))
            features.append(self.creates_mutiple_row_of_three(token_counts=token_counts))
            features.append(self.check_if_middle_column(column_number))
            features.append(self.check_if_edge_column(column_number))
            features.append(self.check_if_other_column(column_number))
            features.append(self.check_if_possible_inevitable_horizontal_win(board=board, current_row=row_index, current_column=column_number, player_number=self.player_number))
            features.append(self.check_if_blocking_possible_inevitable_win(board=board, current_row=row_index, current_column=column_number))
            features.append(self.check_if_inevitable_win_move(player_number=self.player_number, board=board, current_column=column_number, actions_list=actions))
            features.append(self.check_if_blocking_inevitable_win_move(board=board, current_column=column_number, actions_list=actions))

            # ------- End of Features Gen for Column ------- #
            # print(f'[FEATURE EXTRACTION] Extracted feature array for column: {str(column_number)}')
            # print(features)
            column_number += 1
            column_scores.append(self.calculateColumnScore(features_array=features))
        # Get best action. Account for multiple columns with the same score
        max_score = max(column_scores)
        actions = []
        column_number = 0
        for score in column_scores:
            if score == max_score:
                actions.append(column_number)
            column_number += 1
        best_action = 0
        if len(actions) > 1:
            # Randomly choose
            best_action_index = random.randrange(len(actions))
            best_action = actions[best_action_index]
        else:
            best_action = actions[0]
        # print('[COLUMN SCORES]')
        # print(column_scores)
        # print(actions)
        # print(f'Best action selected: {str(best_action)}')
        return best_action

    # Check if putting token in column results in a win for agent
    def check_if_winning_move(self, board, selected_column, selected_row, player_number):
        if self.check_horizontal_win(board=board, selected_column=selected_column, selected_row=selected_row, player_number=player_number) == 4:
            # print('[WIN MOVE CHECKER] HORIZONTAL win detected')
            return 1
        elif self.check_vertical_win(board=board, selected_column=selected_column, selected_row=selected_row, player_number=player_number) == 4:
            # print('[WIN MOVE CHECKER] VERTICAL win detected')
            return 1
        elif self.check_diagonalone_win(board=board, selected_column=selected_column, selected_row=selected_row, player_number=player_number) == 4:
            # print('[WIN MOVE CHECKER] DIAGONAL 1 win detected')
            return 1
        elif self.check_diagonaltwo_win(board=board, selected_column=selected_column, selected_row=selected_row, player_number=player_number) == 4:
            # print('[WIN MOVE CHECKER] DIAGONAL 2 win detected')
            return 1
        return 0

    '''
    Check if putting token in column results in blocking the opponent from winning
    We run check_if_winning_move() but for the opponent instead - Simulating if we do not place the token at this time, at this column,
    the opponent would be able to place their token here in the next step, and win the game
    '''
    def check_if_blocking_move(self, board, selected_column, selected_row):
        return self.check_if_winning_move(board=board, selected_column=selected_column, selected_row=selected_row, player_number=self.opponent_number)

    # Checks if placing token at selected column creates at least a line of 2
    def check_if_row_of_two(self, token_counts):
        for count in token_counts:
            if count >= 1:
                return 1
        return 0

    def creates_multiple_row_of_two(self, token_counts):
        direction_count = 0
        for count in token_counts:
            if count >= 2:
                direction_count += 1
        if direction_count >= 2:
            return 1
        return 0

    def check_if_row_of_three(self, token_counts):
        for count in token_counts:
            if count >= 2:
                return 1
        return 0

    def creates_mutiple_row_of_three(self, token_counts):
        direction_count = 0
        for count in token_counts:
            if count >= 3:
                direction_count += 1
        if direction_count >= 2:
            return 1
        return 0

    def check_if_middle_column(self, current_column):
        if current_column == self.middle_column_index:
            return 1
        return 0

    def check_if_edge_column(self, current_column):
        if current_column == self.leftmost_index or current_column == self.rightmost_index:
            return 1
        return 0

    def check_if_other_column(self, current_column):
        middle_column_result = self.check_if_middle_column(current_column)
        edge_column_result = self.check_if_edge_column(current_column)
        if middle_column_result == 1 or edge_column_result == 1:
            return 0
        return 1

    def check_if_possible_inevitable_horizontal_win(self, board, current_row, current_column, player_number):
        if self.check_if_edge_column(current_column) == 1:
            return 0    # Cannot form inevitable win when placing on edge tiles
        # If adjacent is to the right of our current column
        if board[current_row][current_column + 1] == player_number:
            # Check is empty to left of column
            left_index = current_column - 1
            if left_index < self.leftmost_index: # Out of index
                return 0
            elif board[current_row][left_index] != 0:
                return 0
            # Check to right of adjacent
            right_index = current_column + 2
            if right_index > self.rightmost_index: # Out of index
                return 0
            elif board[current_row][right_index] != 0:
                return 0
            return 1
        # Adjacent to the left of our current column
        elif board[current_row][current_column - 1] == player_number:
            # Check is empty to right of column
            right_index = current_column + 1
            if right_index > self.rightmost_index: # Out of index
                return 0
            elif board[current_row][right_index] != 0:
                return 0
            # Check to left of adjacent
            left_index = current_column - 2
            if left_index < self.leftmost_index: # Out of index
                return 0
            elif board[current_row][left_index] != 0:
                return 0
            return 1
        return 0

    # Check if placing on column prevents the opponent from getting an inevitable win
    def check_if_blocking_possible_inevitable_win(self, board, current_row, current_column):
        return self.check_if_possible_inevitable_horizontal_win(board=board, current_row=current_row, current_column=current_column, player_number=self.opponent_number)
    
    '''
    First check if row we placing on has 2 adjacent already
    Then check if adjacent columns are on same row as current
    Then check if they are 0
    '''
    def check_if_inevitable_win_move(self, player_number, board, current_column, actions_list):
        current_row = actions_list[current_column]
        # Check if row already has 2 adjacent
        left_count = 0
        left_index = current_column - 1
        right_count = 0
        right_index = current_column + 1
        if current_column != self.leftmost_index:
            while left_index >= self.leftmost_index:
                if board[current_row][left_index] == player_number:
                    left_count += 1
                else:
                    break   # Encountered empty or opponent
                left_index -= 1
        if current_column != self.rightmost_index:
            while right_index <= self.rightmost_index:
                if board[current_row][right_index] == player_number:
                    right_count += 1
                else:
                    break   # Encountered empty or opponent
                right_index += 1
        total_sum = left_count + right_count
        if total_sum != 2:
            return 0    # Not a valid inevitable win move
        # No more valid space on either side
        if left_index < self.leftmost_index or right_index > self.rightmost_index:
            return 0
        # Check if next move is at same height
        adjacent_left_row = actions_list[left_index]
        adjacent_right_row = actions_list[right_index]
        if adjacent_left_row != current_row or adjacent_right_row != current_row:
            return 0
        # print('======== INEVITABLE WIN DETECTED ========')
        # print(f'FOR COLUMN: {str(current_column)}')
        return 1

    def check_if_blocking_inevitable_win_move(self, board, current_column, actions_list):
        return self.check_if_inevitable_win_move(self.opponent_number, board, current_column, actions_list)
        

    # ------------ Tokens in direction count functions: Continues when encountering blank spaces ------------ #

    def get_tokens_for_directions(self, player_number, board, selected_column, selected_row):
        counts = []
        counts.append(self.count_tokens_to_left(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_right(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_bottom(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_diagonal_botleft(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_diagonal_botright(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_diagonal_topleft(player_number, board, selected_column, selected_row))
        counts.append(self.count_tokens_to_diagonal_topright(player_number, board, selected_column, selected_row))
        # print(f'[TOKEN COUNTS] Counts for tokens in 7 directions for player {str(self.player_number)}')
        # print(counts)
        return counts

    # Calculates the number of unblocked tokens to left - Can include enoty spaces
    def count_tokens_to_left(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_column == self.leftmost_index:
            return count
        curr_column = selected_column - 1
        while curr_column >= self.leftmost_index:
            if board[selected_row][curr_column] == player_number:
                count += 1
                curr_column -= 1
            elif board[selected_row][curr_column] == 0: # Empty space
                curr_column -= 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_right(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_column == self.rightmost_index:
            return count
        curr_column = selected_column + 1
        while curr_column <= self.rightmost_index:
            if board[selected_row][curr_column] == player_number:
                count += 1
                curr_column += 1
            elif board[selected_row][curr_column] == 0: # Empty space
                curr_column += 1
            else: # Is opponent tile
                return count
        return count

    '''
    Due to logic of gravity, can never have empty spaces below - Only own, or opponent tokens
    Extended, we don't have to check upwards to count number of tokens
    '''
    def count_tokens_to_bottom(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.bottom_index:
            return count
        curr_row = selected_row + 1
        while curr_row <= self.bottom_index:
            if board[curr_row][selected_column] == player_number:
                count += 1
                curr_row += 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_topleft(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.top_index or selected_column == self.leftmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row - 1
        curr_column = selected_column - 1
        while curr_row >= self.top_index and curr_column >= self.leftmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row -= 1
                curr_column -= 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row -= 1
                curr_column -= 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_topright(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.top_index or selected_column == self.rightmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row - 1
        curr_column = selected_column + 1
        while curr_row >= self.top_index and curr_column <= self.rightmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row -= 1
                curr_column += 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row -= 1
                curr_column += 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_botleft(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.bottom_index or selected_column == self.leftmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row + 1
        curr_column = selected_column - 1
        while curr_row <= self.bottom_index and curr_column >= self.leftmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row += 1
                curr_column -= 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row += 1
                curr_column -= 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_botright(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.bottom_index or selected_column == self.rightmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row + 1
        curr_column = selected_column + 1
        while curr_row <= self.bottom_index and curr_column <= self.rightmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row += 1
                curr_column += 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row += 1
                curr_column += 1
            else: # Is opponent tile
                return count
        return count

    # --------------- Winning move calculation functions: Must be CONSECUTIVE ---------------- #

    # Checks to the left and right of last placed piece to determine if 4 in a row has been achieved
    def check_horizontal_win(self, board, selected_column, selected_row, player_number):
        left_count = 0
        right_count = 0
        # Calculate number of pieces to left
        if selected_column != self.leftmost_index:
            curr_column = selected_column - 1
            while curr_column >= self.leftmost_index:
                if board[selected_row][curr_column] == player_number:
                    left_count += 1
                else:
                    break # Not continuous matching token
                curr_column -= 1
        # Calculate number of pieces to right
        if selected_column != self.rightmost_index:
            curr_column = selected_column + 1
            while curr_column <= self.rightmost_index:
                if board[selected_row][curr_column] == player_number:
                    right_count += 1
                else:
                    break # Not continuous matching token
                curr_column += 1
        return 1 + left_count + right_count
    
    # Checks to the top and bottom of last placed piece to determine if 4 in a row has been achieved
    def check_vertical_win(self, board, selected_column, selected_row, player_number):
        up_count = 0
        down_count = 0
        # Calculate number of pieces to bottom
        if selected_row != 5:
            curr_row = selected_row + 1
            while curr_row <= self.bottom_index:
                if board[curr_row][selected_column] == player_number:
                    down_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
        # Calculate number of pieces to top
        if selected_row != 0:
            curr_row = selected_row - 1
            while curr_row >= self.top_index:
                if board[curr_row][selected_column] == player_number:
                    up_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
        return 1 + down_count + up_count

    def check_diagonalone_win(self, board, selected_column, selected_row, player_number):
        diagonal_down_right_count = 0
        diagonal_up_left_count = 0
        # Calculate pieces to diagonal bottom right of current piece
        if selected_column != self.rightmost_index and selected_row != self.bottom_index:   # Ensure there are still spaces to right, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column + 1
            while curr_row <= self.bottom_index and curr_column <= self.rightmost_index:
                if board[curr_row][curr_column] == player_number:
                    diagonal_down_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column += 1
        # Calculate pieces to diagonal top left of current piece
        if selected_column != 0 and selected_row != 0: # Ensure there are still spaces to left, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= self.top_index and curr_column >= self.leftmost_index:
                if board[curr_row][curr_column] == player_number:
                    diagonal_up_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column -= 1
        return 1 + diagonal_down_right_count + diagonal_up_left_count

    def check_diagonaltwo_win(self, board, selected_column, selected_row, player_number):
        diagonal_down_left_count = 0
        diagonal_up_right_count = 0
        # Calculate pieces to diagonal bottom left of current piece
        if selected_column != self.leftmost_index and selected_row != self.bottom_index:   # Ensure there are still spaces to left, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column - 1
            while curr_row <= self.bottom_index and curr_column >= self.leftmost_index:
                if board[curr_row][curr_column] == player_number:
                    diagonal_down_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column -= 1
        # Calculate pieces to diagonal top right of current piece
        if selected_column != self.rightmost_index and selected_row != self.top_index: # Ensure there are still spaces to right, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= self.top_index and curr_column <= self.rightmost_index:
                if board[curr_row][curr_column] == player_number:
                    diagonal_up_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column += 1
        return 1 + diagonal_down_left_count + diagonal_up_right_count