from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, action_space, observation_space):
        self._action_space = action_space
        self._observation_space = observation_space

    @abstractmethod
    def act(self, observation):
        pass