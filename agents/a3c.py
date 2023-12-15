import torch
import numpy as np

from agent import TrainableAgent
from engine import Board, Move, Colour
from model.a3c.model import ActorCritic


class A3CAgent(TrainableAgent):
    def __init__(self,
                board_size: int,
                colour: Colour,
                learning_rate = 1e-4,
                discount = 1.0,
                t_max = 20):
        super().__init__(board_size, colour)

        self.learning_rate = learning_rate 
        self.discount = discount
        self.t_max = t_max

        self.shared_model = ActorCritic(
            4, 
            board_size ** 2 + 1
        )
        self.shared_model.share_memory()
        self._worker_agents = []

        self.reset()

    def act(self, observation: Board) -> Move:
        red, blue, empty = observation.to_channels()
        turn = self._get_colour_channel()

        channels = np.stack([red, blue, empty, turn], axis=-1)
        prob, _, = self.shared_model(torch.from_numpy(channels))
        return prob.max(1)[1].data

    def create_async_learner(self):
        worker_agent = _A3CWorkerAgent(
            self.shared_model,
            self.board_size,
            self.learning_rate,
            self.discount,
            self.t_max
        )

        return worker_agent
    
    def save(self, path: str):
        torch.save(self.shared_model.state_dict(), path)

    def load_model(self, path: str):
        self.shared_model.load_state_dict(torch.load(path))


class _A3CWorkerAgent(TrainableAgent):
    def __init__(self,
            shared_model,
            board_size,
            learning_rate = 1e-4,
            discount = 1.0,
            t_max = 20):
        super().__init__(board_size)

        self.shared_model = shared_model
        self.learning_rate = learning_rate 
        self.discount = discount
        self.t_max = t_max

        self.local_model = ActorCritic(
            board_size * board_size, 
            board_size * board_size + 1)
        self.optimizer = torch.optim.Adam(
            self.shared_model.parameters(), lr=self.learning_rate)

        self.local_model.load_state_dict(self.shared_model.state_dict())

        self.rewards = []
        self.values = []
        self.log_probs = []
        self.entropies = []

    def act(self, observation: Board):
        actor, critic = self.local_model(observation)


if __name__ == "__main__":
    agent = A3CAgent(11, Colour.RED)
    board = Board()
    print(agent.act(board))