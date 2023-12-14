import socket
from random import choice
import time
import math
import copy
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

       if self.total_time > 220:
          self.depth = 1

       best_move = None
       for current_depth in range(1, self.depth + 1):
          move = self.perform_search(current_depth)
          if move:
             best_move = move
          else:
             break  # Stop searching if no move found within the time limit

       if best_move and 0 <= best_move[0] < self._board_size and 0 <= best_move[1] < self._board_size:
            msg = f"{best_move[0]},{best_move[1]}\n"
       else:
            msg = f"{int(math.sqrt(self._board_size)-1)},{int(math.sqrt(self._board_size)-1)}\n"

       self.timerlast = time.time()
       roundTimer = self.timerlast - self.timerstart
       self.total_time += roundTimer
       self._s.sendall(bytes(msg, "utf-8"))

       return 4
    
    def perform_search(self, depth):
      alpha = -math.inf
      beta = math.inf
      best_move = None

      moves = self.get_moves_ordered(self._board)
      for move in moves:
         new_board = self.update_board(self._board, move[0], move[1], self._colour)
         value = self.MinMax(new_board, False, depth, alpha, beta, move)
         if value > alpha:
            alpha = value
            best_move = move

      return best_move

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
             x, y = map(int, data[1].split(","))
             if (data[-1] == self._colour):
                self._board[x, y] = self.opp_colour()
             else: 
                 self._board[x, y] = self._colour
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

    def MinMax(self, startBoard, maximizingPlayer, depth, alpha, beta, currentMove):
        alpha = alpha
        beta = beta
        iterationBestMove = []

        transposition_key = np.array2string(startBoard)
        if transposition_key in self._seen_board:
           return self._seen_board[transposition_key]

        if depth == 0 or not (any("0" in subl for subl in startBoard)):
             turn_colour = self.opp_colour()
             if maximizingPlayer:
                turn_colour = self._colour

             if np.array2string(startBoard) in self._seen_board:
                return self._seen_board[np.array2string(startBoard)]
             else:
                score = self.eval_dijkstra(turn_colour)
                self._seen_board[np.array2string(startBoard)] = score
                return score

        if maximizingPlayer:
           bestValue = -math.inf
           moves = self.get_moves_ordered(startBoard)
           for move in moves:
              newBoard = self.update_board(copy.deepcopy(startBoard), move[0], move[1], self._colour)
              value = self.MinMax(newBoard, False, depth - 1, alpha, beta, move)
              if value > bestValue:
                 iterationBestMove = move
                 bestValue = value
              alpha = max(alpha, bestValue)
              if bestValue >= beta:
                 break
           self._best_move = iterationBestMove
           return bestValue
        else:
           bestValue = math.inf
           moves = self.get_moves_ordered(startBoard)
           for move in moves:
              updatedBoard = self.update_board(copy.deepcopy(startBoard), move[0], move[1], self.opp_colour())
              value = self.MinMax(updatedBoard, True, depth - 1, alpha, beta, move)
              if value < bestValue:
                 iterationBestMove = move
                 bestValue = value
              beta = min(beta, bestValue)
              if bestValue <= alpha:
                 break
           self._best_move = iterationBestMove
           return bestValue

   # def get_moves_ordered(self, board):
        # Get moves and sort them based on historical information (descending order)
      #  moves = self.get_moves(board)
      #  moves.sort(key=lambda move: -self._seen_board.get(np.array2string(self.update_board(copy.deepcopy(board), move[0], move[1], self._colour)), 0))
      #  return moves
    def get_moves_ordered(self, board):
        # Get moves and sort them based on the number of empty neighbors (simple heuristic)
        moves = self.get_moves(board)
        moves.sort(key=lambda move: -len(self.get_empty_neighbors(board, move)))
        return moves
    
    def get_empty_neighbors(self, board, coordinates):
        empty_neighbors = []
        for neighbor in self.get_neighbors(coordinates):
            if self.is_empty(neighbor, board):
                empty_neighbors.append(neighbor)
        return empty_neighbors

    
    def eval_dijkstra(self, color):

        self.eval_count += 1
        return  self.get_dijkstra_score(self.opp_this_colour(color)) - self.get_dijkstra_score(color)

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

    def update_board(self, current_board, x, y, colour):
      new_board = copy.deepcopy(current_board)
      new_board[x][y] = colour
      return new_board
   


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
      if len(moves[0]) > 0:
        moves = list(zip(moves[0], moves[1]))[0]
        swap_border = math.sqrt(self._board_size)
        if ((swap_border < moves[0] < (self._board_size + swap_border)) and
                (swap_border < moves[1] < (self._board_size + swap_border))):
            return True
      return False
            


if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()





