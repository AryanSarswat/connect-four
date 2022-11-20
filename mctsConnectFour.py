from __future__ import division
import datetime
from math import log, sqrt
from random import choice
import numpy as np
from pandas import *
from copy import deepcopy
from abc import ABC

class Game(ABC):
    """
    Abstract base class for defining the game for the `MonteCarlo` class
    """
    def get_current_player(self):
        # Takes the game state and returns the current player's
        # number.
        pass

    def play(self, move):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
        pass

    def get_legal_actions(self):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        pass

    def get_winner(self):
        # Takes a sequence of game states representing the full
        # game history.  If the game is now won, return the player
        # number.  If the game is still ongoing, return zero.  If
        # the game is tied, return a different distinct value, e.g. -1.
        pass

    def print(self):
        pass


class ConnectFour(Game):
    """
    Represents the state of the Connect 4 playing environment.
    Pieces for each player is defined by the integers: 1 for player 1 and 2 for player 2.
    """
    def __init__(self, height=6, width=7):
        self.height = height
        self.width = width
        self.game_state =  np.zeros((height,width)).astype(int)
        self.columns_state = [height - 1] * width
        self.player_turn = 1    # Current player's turn (1 or 2)
        self.turns_left = height * width
        self.is_done = False
        self.is_draw = False

    def print(self):
        print('------ GAME STATE ------')
        print(DataFrame(self.game_state))
        print('------------------------')

    def get_current_player(self):
        return self.player_turn

    def play(self, move):
        self.__update_state(move)
        self.turns_left -= 1

        # Game has terminated
        if self.__has_winner():
            self.is_done = True
            return self
        elif self.turns_left == 0:
            self.is_draw = True
            self.is_done = True
            return self
        
        # Set player turn
        if self.player_turn == 1:
            self.player_turn = 2
        else:
            self.player_turn = 1
        return self

    def get_legal_actions(self):
        actions = []
        for c in range(self.width):
            if self.columns_state[c] >= 0:
                actions.append(c)
        return actions

    def get_winner(self):
        if self.is_draw:
            return -1
        if self.is_done:
            return self.player_turn
        return 0

    def __has_winner(self):
        return (self.__check_horizontal_win() or self.__check_vertical_win() or 
            self.__check_diagonal_win_one() or self.__check_diagonal_win_two())

    def __update_state(self, selected_column):
        r, c = self.columns_state[selected_column], selected_column
        self.game_state[r][c] = self.player_turn
        self.columns_state[selected_column] -= 1
        return

    def __check_horizontal_win(self):
        player_token = self.player_turn
        for r in range(self.height):
            count = 0
            for c in range(self.width):
                if self.game_state[r][c] == player_token:
                    count += 1
                else:
                    count = 0

                if count == 4:
                    return True
        return False
    
    def __check_vertical_win(self):
        player_token = self.player_turn
        for c in range(self.width):
            count = 0
            for r in range(self.height):
                if self.game_state[r][c] == player_token:
                    count += 1
                else:
                    count = 0
                
                if count == 4:
                    return True
        return False

    # Calculates diagonal from top left down to bottom right
    def __check_diagonal_win_one(self):
        for start_row in range(self.height):
            count = 0
            r, c = start_row, 0
            while 0 <= r < self.height and 0 <= c < self.width:
                if self.game_state[r][c] == self.player_turn:
                    count += 1
                else:
                    count = 0
                r += 1
                c += 1

                if count == 4:
                    return True
        
        for start_col in range(1, self.width):
            count = 0
            r, c = 0, start_col
            while 0 <= r < self.height and 0 <= c < self.width:
                if self.game_state[r][c] == self.player_turn:
                    count += 1
                else:
                    count = 0
                r += 1
                c += 1

                if count == 4:
                    return True
            
        return False

    # Calculates diagonal from top right down to bottom left
    def __check_diagonal_win_two(self):
        for start_row in range(self.height):
            count = 0
            r, c = start_row, self.width - 1
            while 0 <= r < self.height and 0 <= c < self.width:
                if self.game_state[r][c] == self.player_turn:
                    count += 1
                else:
                    count = 0
                r += 1
                c -= 1

                if count == 4:
                    return True
        
        for start_col in range(1, self.width):
            count = 0
            r, c = 0, start_col
            while 0 <= r < self.height and 0 <= c < self.width:
                if self.game_state[r][c] == self.player_turn:
                    count += 1
                else:
                    count = 0
                r += 1
                c -= 1

                if count == 4:
                    return True
            
        return False

    def __eq__(self, __o: object) -> bool:
        return (np.array_equal(self.game_state, __o.game_state) and 
                self.player_turn == __o.player_turn)

    def __hash__(self) -> int:
        return hash((tuple(self.game_state.flatten().tolist()), self.player_turn))


class MonteCarlo(object):
    """
    A Monte Carlo Tree Search agent.

    Init args:
    `game` (`Game`): Game to play.
    `simulation_time` (`int`): Time in seconds allowed for the agent to return the next move.
    `max_moves` (`int`): Maximum number of moves allowed before a simulated game is stopped.
    `C` (`float`): Constant used in UCT calculation.
    """
    def __init__(self, game: Game, simulation_time=1, max_moves=42, C=1.4):
        self.game = game
        self.simulation_time = datetime.timedelta(seconds=simulation_time)
        self.max_moves = max_moves
        self.wins = {}
        self.plays = {}
        self.C = C

    def get_move(self):
        """
        Calculates and returns the best move.
        """
        legal_actions = self.game.get_legal_actions()
        if len(legal_actions) == 0:
            return
        if len(legal_actions) == 1:
            return legal_actions[0]

        games = 0
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.simulation_time:
            self.__run_simulation()
            games += 1

        move = self.__get_best_move()

        print("Number of simulations:", games, "Time used:", datetime.datetime.utcnow() - begin)
        # self.__print_stats()

        return move

    def __print_stats(self):
        moves_states = self.__get_next_states(self.game)
        player = self.game.get_current_player()
        stats =  [(100 * self.wins.get((player, s), 0) / self.plays.get((player, s), 1),
                    self.wins.get((player, s), 0),
                    self.plays.get((player, s), 0), m)
                    for m, s in moves_states]
        for x in sorted(stats, reverse=True):
            print("{3}: {0:.2f}% ({1} / {2})".format(*x))
        return

    def __get_next_states(self, curr_state: Game):
        legal_actions = curr_state.get_legal_actions()
        moves_states = [(move, deepcopy(curr_state).play(move)) for move in legal_actions]
        return moves_states

    def __run_simulation(self):
        """
        Simulates a game from the current state.
        """
        visited_states = set()
        state = self.game
        player = self.game.get_current_player()

        expand = True
        for _ in range(self.max_moves):
            moves_states = self.__get_next_states(state)

            if all(self.plays.get((player, s)) for m, s in moves_states):
                # Selection
                state = self.__choose_next_state(player, moves_states)
            else:
                # Possible expansion
                _, state = choice(moves_states)

            if expand and (player, state) not in self.plays:
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
                expand = False

            visited_states.add((player, state))

            player = state.get_current_player()
            winner = state.get_winner()
            if winner:
                break
        
        # Backup
        self.__update_stats(visited_states, winner)

    def __choose_next_state(self, player, moves_states):
        log_total = log(sum(self.plays[(player, s)] for m, s in moves_states))
        _, _, best_state = max(((self.wins[(player, s)] / self.plays[(player, s)]) + 
                                self.C * sqrt(log_total / self.plays[(player, s)]),
                                m, s)
            for m, s in moves_states)
        return best_state
    
    def __update_stats(self, visited_states, winner):
        for player, state in visited_states:
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if player == winner:
                self.wins[(player, state)] += 1
        return

    def __get_best_move(self):
        player = self.game.get_current_player()
        moves_states = self.__get_next_states(self.game)
        _, move = max((self.wins.get((player, s), 0) / self.plays.get((player, s), 1), m)
                        for m, s in moves_states)
        return move

def start(game, agent):
    is_over = False
    opponent_player_number, agent_player_number = 1, 2
    print('----------- CONNECT FOUR -----------')
    print('[NEW GAME] Starting a new connect 4 game!')
    print(f'Player Number: {str(opponent_player_number)}, AI Number: {str(agent_player_number)}')
    print()
    game.print()
    while not is_over:
        available_actions = game.get_legal_actions()
        move = 0
        if game.get_current_player() == agent_player_number:
            move = agent.get_move()
            print(f'[AI Turn] - AI selected column number: {str(move)}')
        else:
            print(f'[Player turn] - Please select one of the following columns: {available_actions}')
            try:
                move = input('Select column: ')
                move = int(move)
            except Exception as e:
                print(e)
                continue
        if not move in available_actions:
            print('[FATAL ERROR] Invalid column selected! - Please try again')
            continue

        game.play(move)
        game.print()

        result = game.get_winner()
        if result == agent_player_number:
            print('[GAME OVER] AI has won!')
            is_over = True
        elif result == opponent_player_number:
            print('[GAME OVER] Player has won! Congratulations!')
            is_over = True
        elif result == -1:
            print('[GAME OVER] Draw!')
            is_over = True
    return

if __name__ == '__main__':
    game = ConnectFour()
    agent = MonteCarlo(game) # Change simulation time and constant C here 
    start(game, agent)

    