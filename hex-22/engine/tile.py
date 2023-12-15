from engine.colour import Colour


class Tile:
    """The class representation of a tile on a board of Hex."""

    # number of neighbours a tile has
    NEIGHBOUR_COUNT = 6

    # relative positions of neighbours, clockwise from top left
    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

    def __init__(self, x: int, y: int, colour: Colour=None):
        super().__init__()

        self.x = x
        self.y = y
        self.colour = colour

        self.visited = False

    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y

    def set_colour(self, colour: Colour):
        self.colour = colour

    def get_colour(self) -> Colour:
        return self.colour

    def visit(self):
        self.visited = True

    def is_visited(self) -> bool:
        return self.visited

    def clear_visit(self):
        self.visited = False

    def reset(self) :
        self.colour = None
        self.visited = False
