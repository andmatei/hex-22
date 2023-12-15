import gymnasium as gym
import numpy as np

from gymnasium import spaces
from engine.Board import Board
from engine.Colour import Colour
from engine.EndState import EndState
from engine.Move import Move
from engine.Game import BaseGame

class HexEnv(gym.Env, BaseGame):
    def __init__(self, board_size):
        self.turn = 1
        self.board = Board(board_size)
        self.current_player = Colour.RED
        self.start_time = 0
        self.has_swaped = False

        self.action_space = spaces.MultiDiscrete([board_size + 1, board_size + 1])
        self.observation_space = spaces.Box(
            low=0, 
            high=2, 
            shape=(board_size, board_size), 
            dtype=int
        )

    def reset(self):
        self.board.reset()
        self.turn = 1
        self.current_player = Colour.RED
        self.start_time = 0
        self.has_swaped = False

        return self.board, {}

    def step(self, action: Move):
        # assert self.action_space.contains(
        #     action
        # ), f"{action!r} ({type(action)}) invalid"
        assert not self.board.has_ended(), \
            """You are calling 'step()' even though this
            environment has already returned terminated = True. You
            should always call 'reset()' once you receive 'terminated =
            True' -- any further steps are undefined behavior."""

        reward = 0
        terminated = False
        truncated = False
        end_state = None

        if not action.is_valid_move(self):
            self.flip_turn()
            reward = -1000
            truncated = True
            end_state = EndState.BAD_MOVE
        
        else:
            reward = self.get_reward(action, self.current_player)
            if action.is_swap():
                self.has_swaped = True
                self.current_player = Colour.opposite(self.current_player)
            else:
                action.move(self.board)
            self.flip_turn()

        return self.board, reward, terminated, truncated, {
            "last_move": action,
            "player": self.current_player,
            "turn": self.turn,
            "end_state": end_state
        }
        
    def get_reward(self, action: Move, player: Colour):
        pass

    def flip_turn(self):
        self.turn += 1
        self.current_player = Colour.opposite(self.current_player)

    def get_board(self):
        return self._board
    
    def get_player(self):
        return self._player
    
    def get_turn(self):
        return self._turn

