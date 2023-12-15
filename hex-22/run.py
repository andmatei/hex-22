from agents.a3c import A3CAgent
from engine.board import Board
from engine.colour import Colour

if __name__ == "__main__":
    agent = A3CAgent(11, Colour.RED)
    b = Board.from_string(
        "0R000B00000,0R000000000,0RBB0000000,0R000000000,0R00B000000," +
        "0R000BB0000,0R0000B0000,0R00000B000,0R000000B00,0R0000000B0," +
        "0R00000000B", bnf=True
    )
    print(agent.act(b))