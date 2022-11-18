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

MAX_MEMORY = 100_000_000_000
BATCH_SIZE = 1000000
LR = 0.001

class Direction(Enum):
    RIGHT = 1
    LEFT = -1

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1000 # randomness
        self.gamma = 0.5 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(13, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        Point = namedtuple('Point', 'x, y') 
        paddel_position = Point(game.player_paddle.x,game.player_paddle.y)
        ball_position = Point(game.ball.x, game.ball.y)
        
        dir_l = game.player_paddle.direction == Direction.LEFT
        dir_r = game.player_paddle.direction == Direction.RIGHT
        
        state = [
            # Ball Left
            (paddel_position.x-game.player_paddle.width/2 > ball_position.x),

            # Ball right
            (paddel_position.x+game.player_paddle.width/2 < ball_position.x),

            # Ball top
            (paddel_position.x+game.player_paddle.width/2 > ball_position.x) and
            (paddel_position.x-game.player_paddle.width/2 < ball_position.x),
            
            # paddle direction
            # (paddel_position.y +20 > ball_position.y),
            # Se mueve a la derecha
            (game.ball.speed_x > 0),
            # Se mueve a la izquierda
            (game.ball.speed_x < 0),
            # Se mueve arriba
            (game.ball.speed_y > 0),
            # Se mueve abajo
            (game.ball.speed_y < 0),
            dir_l,
            dir_r,
            (dir_l == (game.ball.speed_x <0)),
            (dir_r == (game.ball.speed_x >0)),
            (paddel_position.x > (game.screen.get_width()-game.player_paddle.width )),
            (paddel_position.x < (game.player_paddle.width))
            ]
        """
        Implementar posición ladrillos
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        """
            

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 200 - self.n_games
        final_move = [0,0,0] # [IZDA, QUIETO, DCHA]
        #final_move = [0,0] # [IZDA, DCHA]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
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
        #state_old = agent.get_state(game)

        # get move
        final_move = random.choice([[1,0,0],[0,1,0],[0,0,1]])

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        '''
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        '''
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            #agent.train_long_memory()
            
            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def train():
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

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
    #test()