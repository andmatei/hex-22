from math import sqrt, log

inf = float('inf')

class Node:
    def __init__(self, move = None, parent = None):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = []
        self.outcome = None

    def add_children(self, children):
        self.children += children

    def set_outcome(self, outcome):
        self.outcome = outcome

    def value(self, explore):
        if(self.N == 0):
            if(explore == 0):
                return 0
            else:
                return inf
        else:
            return self.Q / self.N + explore * sqrt(2 * log(self.parent.N) / self.N)
        

