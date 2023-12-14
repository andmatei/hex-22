from random import choice
import socket
import time
import math
import copy
import heapq
import numpy as np


BAD = 1000  
class NaiveAgent():


    HOST = "127.0.0.1"
    PORT = 1234

    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]
    


    def run(self):
        
        self._board_size = 0
        self._colour = ""
        self._turn_count = 1
        self._choices = []
        self._best_move = []
        self._seen_board = dict()
        self.eval_count = 0
        self.depth = 2
        self.Swaped = True
        self.timerstart = 0
        self.timerlast = 0
        self.total_time = 0 

        
        
        states = {
            1: NaiveAgent._connect,
            2: NaiveAgent._wait_start,
            3: NaiveAgent._make_move,
            4: NaiveAgent._wait_message,
            5: NaiveAgent._close
        }

        res = states[1](self)
        while (res != 0):
            res = states[res](self)

    def _connect(self):

        
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((NaiveAgent.HOST, NaiveAgent.PORT))

        return 2

    def _wait_start(self):

        
        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "START"):
            self._board_size = int(data[1])
            self._board = np.full((self._board_size,self._board_size), "0")
            self._colour = data[2]


            if (self._colour == "R"):
                return 3
            else:
                return 4

        else:
            print("ERROR: No START message received.")
            return 0

    def _make_move(self):
        self.timerstart = time.time()
        if (self.total_time > 220):
            self.depth = 1 

        if (self._turn_count == 1):
            msg = f"{int(math.sqrt(self._board_size)-1)},{int(math.sqrt(self._board_size)-1)}\n"
            self.Swaped = False
        elif (self._turn_count == 2 and self.CheckSwap()):
            msg = "SWAP\n"   
        else:
            newBoard = copy.deepcopy(self._board)
            value = self.MinMax(newBoard, True, self.depth, -math.inf, math.inf, [0,0])

            msg = f"{self._best_move[0]},{self._best_move[1]}\n"
            
        self.timerlast = time.time()
        roundTimer = self.timerlast - self.timerstart
        self.total_time = self.total_time + roundTimer
        self._s.sendall(bytes(msg, "utf-8"))

        return 4

    def _wait_message(self):
        
        self._turn_count += 1
        
        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "END" or data[-1] == "END"):
            return 5
        else:

            if (data[1] == "SWAP"):
                self._colour = self.opp_colour()
                for x in range(self._board_size):
                    for y in range(self._board_size):
                        if self._board[x][y] != "0":
                            if (self._board[x][y] == "R"):
                                self._board[x][y] = "B"
                            else: 
                                self._board[x][y] = "R"
                            break
            else:
                x, y = data[1].split(",")
                if (data[-1] == self._colour):
                    self._board[int(x)][int(y)] = self.opp_colour()
                else: 
                    self._board[int(x)][int(y)] = self._colour
            if (data[-1] == self._colour):
                return 3

        return 4

    def _close(self):

        self._s.close()
        return 0

    def opp_colour(self):
        
        if self._colour == "R":
            return "B"
        elif self._colour == "B":
            return "R"
        else:
            return "None"
    def opp_this_colour(self, colour):
        
        if colour == "R":
            return "B"
        elif colour == "B":
            return "R"
        else:
            return "0"

    def MinMax(self, startBoard, maximizingPlayer, depth, alpha, beta, currentMove ):
        alpha = alpha
        beta = beta
        iterationBestMove = []


        if depth == 0 or  not (any("0" in subl for subl in startBoard)):
            turn_colour = self.opp_colour
            if maximizingPlayer:
                turn_colour = self._colour

            if np.array2string(self._board) in self._seen_board:
                return self._seen_board[np.array2string(self._board)]
            else:
                score =  self.eval_dijkstra(turn_colour)
                self._seen_board[np.array2string(self._board)] = score
                return score
             
        
    
  
        if maximizingPlayer:
            bestValue= -math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                self._board[move[0]][move[1]] = self._colour
                value = self.MinMax(self._board, False, depth-1, alpha, beta, move)
                self._board[move[0]][move[1]] = "0"
                if value > bestValue:
                    iterationBestMove = move
                    bestValue = value
                alpha = max(alpha, bestValue)
                if bestValue >= beta:
                    break
            self._best_move = iterationBestMove
            return bestValue
        else: 
            bestValue= math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                self._board[move[0]][move[1]] = self.opp_colour()
                value = self.MinMax(self._board, True, depth-1, alpha, beta, move)
                self._board[move[0]][move[1]] = "0"
                if value < bestValue:
                    iterationBestMove = move
                    bestValue = value
                beta = min(beta, bestValue)
                if bestValue <= alpha:
                        break
            self._best_move = iterationBestMove
            return bestValue

    def eval_dijkstra(self, color):

       weights = {
         'num_pieces': 1.5,
         'dijkstra_score': 1.5,
         'center_control': 1.5,
         'mobility': 1.5,
         'edge_control': 1.5,  # New feature: control of edges
         'corner_control': 1.5
        }

       num_pieces = np.sum(self._board == color)
       dijkstra_score = self.get_dijkstra_score(color)
       center_control = self.get_center_control(color)
       mobility = self.get_mobility(color)
       edge_control = self.get_edge_control(color)  # Implement this method
       corner_control = self.get_corner_control(color)

    # Calculate the final heuristic value
       heuristic = (
          weights['num_pieces'] * num_pieces +
          weights['dijkstra_score'] * dijkstra_score +
          weights['center_control'] * center_control +
          weights['mobility'] * mobility +
          weights['edge_control'] * edge_control +
         weights['corner_control'] * corner_control
       )

       return heuristic
        #self.eval_count += 1
        
        #return  self.get_dijkstra_score(self.opp_this_colour(color)) - self.get_dijkstra_score(color)
    def get_edge_control(self, color):
    # Calculate the control of edges for the given color
      edge_count = 0
      for i in range(self._board_size):
          if self.is_color((i, 0), color) or self.is_color((i, self._board_size - 1), color):
             edge_count += 1
          if self.is_color((0, i), color) or self.is_color((self._board_size - 1, i), color):
             edge_count += 1
      return edge_count

    def get_corner_control(self, color):
    # Calculate the control of corners for the given color
       corner_count = 0
       corners = [(0, 0), (0, self._board_size - 1), (self._board_size - 1, 0), (self._board_size - 1, self._board_size - 1)]
       for corner in corners:
          if self.is_color(corner, color):
             corner_count += 1
       return corner_count

    def get_center_control(self, color):
    # In Hex, the concept of center control is different
    # You can check if there are pieces forming a path from one side to the opposite side
      path_exists = self.check_path(color)
      center_control = 1 if path_exists else 0
      return center_control

    def get_mobility(self, color):
    # Mobility in Hex can be calculated by counting the number of empty neighboring cells
      moves = self.get_moves(self._board)
      color_moves = [move for move in moves if self.is_color(move, color)]
      mobility = sum(1 for move in color_moves if self.is_empty(move))
      return mobility

    def check_path(self, color):
    # Helper method to check if there is a path from one side to the opposite side for a given color
      start, end = (0, 0), (self._board_size - 1, self._board_size - 1) if color == "B" else (0, self._board_size - 1)
      visited = set()

      def dfs(coord):
         visited.add(coord)
         if coord[1] == end[1]:
            return True
         neighbors = self.get_neighbors(coord)
         color_neighbors = [neighbor for neighbor in neighbors if self.is_color(neighbor, color) and neighbor not in visited]
         return any(dfs(neighbor) for neighbor in color_neighbors)

      return any(dfs((0, i)) for i in range(self._board_size))

    def dijkstra_update(self, color, scores, updated):

        
        updating = True
        while updating: 
            updating = False
            for i, row in enumerate(scores): 
                for j, point in enumerate(row):  
                    if not updated[i][j]: 
                        neighborcoords = self.get_neighbors((i,j))  
                        for neighborcoord in neighborcoords:
                            target_coord = tuple(neighborcoord)
                            path_cost = BAD  
                            if self.is_empty(target_coord):
                                path_cost = 1
                            elif self.is_color(target_coord, color):
                                path_cost = 0
                            
                            if scores[target_coord] > scores[i][j] + path_cost: 
                                scores[target_coord] = scores[i][j] + path_cost 
                                updated[target_coord] = False 
                                updating = True 
        return scores
    def get_dijkstra_score(self, color):

        scores = an_array = np.full((self._board_size, self._board_size), BAD)
        updated = an_array = np.full((self._board_size, self._board_size), True)
        alignment = (1, 0) if color == "B" else (0, 1)


        for i in range(self._board_size):
            newcoord = tuple([i * j for j in alignment])

            updated[newcoord] = False
            if self.is_color(newcoord, color):
                scores[newcoord] = 0
            elif self.is_empty(newcoord):
                scores[newcoord] = 1
                scores[newcoord] = BAD

        scores = self.dijkstra_update(color, scores, updated)


        results = [scores[alignment[0] * i - 1 + alignment[0]][alignment[1]*i - 1 + alignment[1]]
                   for i in range(self._board_size)] 
        best_result = min(results)

        return best_result 

    def is_empty(self, coordinates):

        return self._board[coordinates] == "0"

    def is_color(self, coordinates, color):

        return self._board[coordinates] == color



    def get_moves(self, board):

        moves = np.where(board == "0")
        moves = list(zip(moves[0], moves[1]))

        return moves

    def update_board(self, updateboard, x,y, colour):
        updateboard[x][y] = colour
        return updateboard   


    def get_neighbors(self, coordinates):

        (x,y) = coordinates
        neighbors = []
        if x-1 >= 0: neighbors.append((x-1,y))
        if x+1 < self._board_size: neighbors.append((x+1,y))
        if x-1 >= 0 and y+1 <= self._board_size-1: neighbors.append((x-1,y+1))
        if x+1 < self._board_size and y-1 >= 0: neighbors.append((x+1,y-1))
        if y+1 < self._board_size: neighbors.append((x,y+1))
        if y-1 >= 0: neighbors.append((x,y-1))

        return neighbors

    def get_heuristic(self, move):
        if self._board.tobytes() in self._seen_board:
            return self._seen_board[self._board.tobytes()]
        else:
            myPath = self.beam(self._colour, move)
            oppPath = self.beam(self.opp_colour(), move)
            self._seen_board[self._board.tobytes()] = myPath - oppPath
            return self._seen_board[self._board.tobytes()]


    def beam(self, colour, move): 
        count = 0
        if(colour == "R"):
            count = (self._board[:, move[1]] == "R").sum()
        else:
            count = (self._board[move[0], :] == "B").sum()

        return count
    

    def CheckSwap(self):
        moves = np.where(self._board == self.opp_colour())
        moves = list(zip(moves[0], moves[1]))[0]
        swap_border = math.sqrt(self._board_size)
        if (( moves[0] > swap_border and moves[0] < (self._board_size + swap_border)) and ( moves[1] > swap_border and moves[1] < (self._board_size + swap_border))):
            return True
        else:
            return False
            


if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
