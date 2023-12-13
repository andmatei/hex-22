from abc import ABC, abstractmethod

from engine import Move, Board

class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, board_size: int):
        self.board_size = board_size

    @abstractmethod
    def act(self, observation: Board) -> Move:
        pass


class LearnableAgent(BaseAgent):
    @abstractmethod
    def learn(self, reward, next_observation, done):
        pass

    def reset(self):
        pass