import numpy as np

from abc import ABC, abstractmethod
from engine import Move, Board, Colour

class BaseAgent(ABC):
    def __init__(self, board_size: int, colour: Colour):
        self.board_size = board_size
        self.colour = colour

    @abstractmethod
    def act(self, observation: Board) -> Move:
        pass

class TrainableAgent(BaseAgent):
    def __init__(self, board_size: int, colour: Colour):
        super().__init__(board_size, colour)

    @abstractmethod
    def learn(self, reward, next_observation, done):
        pass

    def reset(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    def _get_colour_channel(self):
        if self.colour == Colour.RED:
            return np.ones([self.board_size, self.board_size])
        return np.zeros([self.board_size, self.board_size])


class PlayableAgent(BaseAgent):
    pass