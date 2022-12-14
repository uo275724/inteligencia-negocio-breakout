import torch
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from collections import deque
from breakout_IA import BreakoutGameAI
from model import Linear_QNet, QTrainer
import torch
from helper import plot
if torch.cuda.is_available():
    print("GPU")  
    dev = "cuda:0" 
else:  
    print("CPU")
    dev = "cpu" 
FRAMES = 1
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Direction(Enum):
    RIGHT = 1
    LEFT = -1

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(3, 32, 3)
        if(dev == "cpu"):
            self.model.load_state_dict(torch.load("model\model.pth",map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load("model\model.pth"))
        
        self.model.eval()


    def get_state(self, game):
        Point = namedtuple('Point', 'x, y') 
        paddel_position = Point(game.player_paddle.rect.x,game.player_paddle.rect.y)
        ball_position = Point(game.ball.rect.x, game.ball.rect.y)
        state = [paddel_position.x, ball_position.x, ball_position.y]
        return np.array(state, dtype=int)

    def get_action(self, state):
        final_move = [0,0,0] # [IZDA, QUIETO, DCHA]
        state0 = torch.tensor(state, dtype=torch.float).to(device=dev)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move

def test():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BreakoutGameAI()
    
    

    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)
       
        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    test()