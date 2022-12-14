import pygame
from pygame.locals import *
from enum import Enum
import numpy as np
import random

pygame.init()

screen_width = 600
screen_height = 600
# define font
font = pygame.font.SysFont('Constantia', 30)

# define colours
bg = (234, 218, 184)
# block colours
block_red = (242, 85, 96)
block_green = (86, 174, 87)
block_blue = (69, 177, 232)
# paddle colours
paddle_col = (142, 135, 123)
paddle_outline = (100, 100, 100)
# text colour
text_col = (78, 81, 139)

cols = 6
rows = 6
fps = 120 #10000
is_random = False # True for random ball moves

class Direction(Enum):
    RIGHT = 1
    LEFT = -1


class BreakoutGameAI:
    # Funcion que inicializa variables locales, como la screen o la pelota.
    # Al terminar llama a reset
    def __init__(self):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Breakout')

        # define game variables
        self.clock = pygame.time.Clock()

        # Create the objects
        # create a wall
        self.wall = self.wall()
        self.wall.create_wall()

        # create paddle
        self.player_paddle = self.paddle()

        # create ball
        self.ball = self.game_ball(self.player_paddle.x + (self.player_paddle.width // 2), self.player_paddle.y - self.player_paddle.height)

        self.reset()

        

    # Devuelve el juego al estado inicial
    # (Nota: ahora mismo solo es lo que estaba justo antes del bucle inicial)
    def reset(self):
        self.game_over = 0
        self.frame_iteration = 0
        self.score = 0

        
        #Reset objects
        self.live_ball = True
        self.ball.reset(self.player_paddle.x + (self.player_paddle.width // 2), self.player_paddle.y - self.player_paddle.height)
        self.player_paddle.reset()
        self.wall.create_wall()


        self.run = True

    # function for outputting text onto the screen
    def draw_text(self, text, font, text_col, x, y):
        img = font.render(text, True, text_col)
        self.screen.blit(img, (x, y))

    # Contiene el bucle principal tal cual estaba.
    # Es para poder probar que todo funcione bien hasta ahora al utilizar esta clase como main
    def bucle_juego_prueba(self):
        # BUCLE PRINCIPAL DEL JUEGO
        while self.run:

            self.clock.tick(fps)

            self.screen.fill(bg)

            # draw all objects
            self.wall.draw_wall(self)
            self.player_paddle.draw(self)
            self.ball.draw(self)

            if self.live_ball:
                # draw paddle
                self.player_paddle.move()
                # draw ball
                self.game_over = self.ball.move(self)
                if self.game_over != 0:
                    self.live_ball = False

            # print player instructions
        
            #if self.game_over == 0:
                #self.draw_text('CLICK ANYWHERE TO START', font, text_col, 100, screen_height // 2 + 100)
            if self.game_over == 1:
                #self.draw_text('YOU WON!', font, text_col, 240, screen_height // 2 + 50)
                #self.draw_text('CLICK ANYWHERE TO START', font, text_col, 100, screen_height // 2 + 100)
                pygame.quit()
                quit()
            elif self.game_over == -1:
                #self.draw_text('YOU LOST!', font, text_col, 240, screen_height // 2 + 50)
                #self.draw_text('CLICK ANYWHERE TO START', font, text_col, 100, screen_height // 2 + 100)
                print("Perdiste :C")
                pygame.quit()
                quit()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #Aqui solo se llega si cierras la ventana
                    self.run = False
                    print("Cerraste la ventana")
                """
                if  self.live_ball == False: # and event.type == pygame.MOUSEBUTTONDOWN :
                    self.live_ball = True
                    self.ball.reset(self.player_paddle.x + (self.player_paddle.width // 2), self.player_paddle.y - self.player_paddle.height)
                    self.player_paddle.reset()
                    self.wall.create_wall()
                """


            pygame.display.update()

        pygame.quit()
    '''TEST PLAY PARA IA POR HACER'''
    def play_step(self, action):
        #self.frame_iteration += 1
        # 1. Mirar si se intent?? cerrar la ventana
        self.clock.tick(fps)
        self.screen.fill(bg)
        self.wall.draw_wall(self)
        self.player_paddle.draw(self)
        self.ball.draw(self)
        
        
        
        gameover = False
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    #self.run = False
                    pygame.quit()
                    quit()


        # 2. Moverse seg??n la acci??n
        #self.player_paddle._move(action) # update the head

        # TODO: El resto est?? sin hacer, auque tiene algunas cosas hechas
        
        # 3. check if game over
        if self.live_ball:
                # draw paddle
                self.player_paddle._move(action)
                # draw ball
                self.game_over = self.ball.move(self)
                if self.game_over != 0:
                    self.live_ball = False
                    
        
            
        #if self.ball.rect.colliderect(self.player_paddle):
            #reward = 1000000
        if self.game_over == -1:
            gameover = True
            reward = -100
            return reward, gameover, self.score
        if self.game_over == 1:
            gameover = True
            reward = 1000
            self.score+=69000
            return reward, gameover, self.score

        
        '''
        if ((self.player_paddle.x+self.player_paddle.width/2 > self.ball.x) and
            (self.player_paddle.x-self.player_paddle.width/2 < self.ball.x) and (self.player_paddle.y < self.ball.y)):
            reward += 5000
        '''
        # 5. update ui and clock
        # draw all objects
        #print("Posici??n paleta x: {} ".format(self.player_paddle.rect.x))
        #print("Posici??n Bola x: {}".format(self.ball.rect.x))
        
        if (((self.player_paddle.rect.x+(game.player_paddle.width/2)*0.9 >= self.ball.rect.x) and
            (self.player_paddle.rect.x-(game.player_paddle.width/2)*0.9 <= self.ball.rect.x) )):
            aux = 100
            
            # La recompensa es mayor si la pelota est?? cerca de la paleta
            aux = (self.ball.rect.y/screen_height) * (aux)

            reward = aux
            #print("EST?? DEBAJOOOOOOOOOOOOOO")
        else:
            # M??s castigo si est?? lejos (en horizontal)
            aux = abs(self.player_paddle.rect.x - self.ball.rect.x) * -1

            # El castigo es m??s severo si adem??s la pelota est?? cerca del suelo
            aux = (self.ball.rect.y/screen_height) * (aux)
            # aux = (self.ball.rect.y/screen_height) * (aux)

            reward = aux
        #print("Reward: {}".format(reward))

        pygame.display.update()
        # 6. return game over and score
        return reward, gameover, self.score
    '''TEST PLAY PARA IA POR HACER'''
    # brick wall class
    class wall():
        def __init__(self):
            self.width = screen_width // cols
            self.height = 50

        def create_wall(self):
            self.blocks = []
            # define an empty list for an individual block
            block_individual = []
            for row in range(rows):
                # reset the block row list
                block_row = []
                # iterate through each column in that row
                for col in range(cols):
                    # generate x and y positions for each block and create a rectangle from that
                    block_x = col * self.width
                    block_y = row * self.height
                    rect = pygame.Rect(block_x, block_y, self.width, self.height)
                    # assign block strength based on row
                    if row < 2:
                        strength = 3
                    elif row < 4:
                        strength = 2
                    elif row < 6:
                        strength = 1
                    # create a list at this point to store the rect and colour data
                    block_individual = [rect, strength]
                    # append that individual block to the block row
                    block_row.append(block_individual)
                # append the row to the full list of blocks
                self.blocks.append(block_row)

        def draw_wall(self, super):
            for row in self.blocks:
                for block in row:
                    # assign a colour based on block strength
                    if block[1] == 3:
                        block_col = block_blue
                    elif block[1] == 2:
                        block_col = block_green
                    elif block[1] == 1:
                        block_col = block_red
                    pygame.draw.rect(super.screen, block_col, block[0])
                    pygame.draw.rect(super.screen, bg, (block[0]), 2)

    # paddle class
    class paddle():
        def __init__(self):
            self.reset()

        def move(self):
            # reset movement direction
            self.direction = 0
            key = pygame.key.get_pressed()
            if key[pygame.K_LEFT] and self.rect.left > 0:
                self.rect.x -= self.speed
                self.direction = -1
            if key[pygame.K_RIGHT] and self.rect.right < screen_width:
                self.rect.x += self.speed
                self.direction = 1
        
        def _move(self, action):
            # Igual que el anterior, pero en vez de usando las keys, la accion
            # Accion = [IZDA, QUIETO, DCHA] ser?? uno donde toque
            # reset movement direction (por defecto)
            self.direction = 0
            
            if np.array_equal(action, [1,0,0]) and self.rect.left > 0:
                # IZQUIERDA
                self.rect.x -= self.speed
                self.direction = -1
            if np.array_equal(action, [0,1,0]):
                # QUIETO
                self.direction = 0
            if np.array_equal(action, [0,0,1]) and self.rect.right < screen_width:
                # DERECHA
                self.rect.x += self.speed
                self.direction = 1
            '''
            if np.array_equal(action, [1,0]) and self.rect.left > 0:
                # IZQUIERDA
                self.rect.x -= self.speed
                self.direction = -1
            if np.array_equal(action, [0,1]) and self.rect.right < screen_width:
                # DERECHA
                self.rect.x += self.speed
                self.direction = 1
            '''
        def draw(self,super):
            pygame.draw.rect(super.screen, paddle_col, self.rect)
            pygame.draw.rect(super.screen, paddle_outline, self.rect, 3)

        def reset(self):
            # define paddle variables
            self.height = 20
            self.width = int(screen_width / cols)
            self.x = int((screen_width / 2) - (self.width / 2))
            self.y = screen_height - (self.height * 2)
            self.speed = 10
            self.rect = Rect(self.x, self.y, self.width, self.height)
            self.direction = 0

    # ball class
    class game_ball():
        def __init__(self, x, y):
            self.reset(x, y)

        def move(self, super):

            # collision threshold
            collision_thresh = 5

            # start off with the assumption that the wall has been destroyed completely
            wall_destroyed = 1
            row_count = 0
            for row in super.wall.blocks:
                item_count = 0
                for item in row:
                    """
                    SCORE SE SUMA AQU??
                    """
                    # check collision
                    if self.rect.colliderect(item[0]):
                        # check if collision was from above
                        if abs(self.rect.bottom - item[0].top) < collision_thresh and self.speed_y > 0:
                            self.speed_y *= -1
                        # check if collision was from below
                        if abs(self.rect.top - item[0].bottom) < collision_thresh and self.speed_y < 0:
                            self.speed_y *= -1
                        # check if collision was from left
                        if abs(self.rect.right - item[0].left) < collision_thresh and self.speed_x > 0:
                            self.speed_x *= -1
                            
                        # check if collision was from right
                        if abs(self.rect.left - item[0].right) < collision_thresh and self.speed_x < 0:
                            self.speed_x *= -1
                            
                            
                        # reduce the block's strength by doing damage to it
                        if super.wall.blocks[row_count][item_count][1] > 1:
                            super.wall.blocks[row_count][item_count][1] -= 1
                            super.score += 10

                        else:
                            super.wall.blocks[row_count][item_count][0] = (0, 0, 0, 0)
                            super.score += 5

                        print("Score: {}".format(super.score))
                    # check if block still exists, in whcih case the wall is not destroyed
                    if super.wall.blocks[row_count][item_count][0] != (0, 0, 0, 0):
                        wall_destroyed = 0
                    # increase item counter
                    item_count += 1
                # increase row counter
                row_count += 1
            # after iterating through all the blocks, check if the wall is destroyed
            if wall_destroyed == 1:
                self.game_over = 1

            # check for collision with walls
            if self.rect.left < 0 or self.rect.right > screen_width:
                self.speed_x *= -1


            # check for collision with top and bottom of the screen
            if self.rect.top < 0:
                self.speed_y *= -1

            if self.rect.bottom > screen_height:
                self.game_over = -1

            # look for collission with paddle
            if self.rect.colliderect(super.player_paddle):
                # check if colliding from the top
                if abs(self.rect.bottom - super.player_paddle.rect.top) < collision_thresh and self.speed_y > 0:
                    self.speed_y *= -1

                    # Esto hace que la pelota cambie ligeramente de direcci??n, como si se le diese efecto
                    if(not is_random):
                        self.speed_x += super.player_paddle.direction
                    # Pues en vez de eso, que sea random:
                    else:
                        self.speed_x = random.randint(-self.speed_max, self.speed_max)

                    # Respetar el limite de velocidad
                    if self.speed_x > self.speed_max:
                        self.speed_x = self.speed_max

                    elif self.speed_x < 0 and self.speed_x < -self.speed_max:
                        self.speed_x = -self.speed_max 

                else:
                    self.speed_x *= -1


            self.rect.x += self.speed_x

            self.rect.y += self.speed_y

            #print("Dentro de bolaX {}".format(self.rect.x))
            return self.game_over

        def draw(self, super):
            pygame.draw.circle(super.screen, paddle_col, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad),
                            self.ball_rad)
            pygame.draw.circle(super.screen, paddle_outline, (self.rect.x + self.ball_rad, self.rect.y + self.ball_rad),
                            self.ball_rad, 3)

        def reset(self, x, y):
            self.ball_rad = 10
            self.x = x - self.ball_rad
            self.y = y
            self.rect = Rect(self.x, self.y, self.ball_rad * 2, self.ball_rad * 2)
            self.speed_x = 4
            self.speed_y = -4
            self.speed_max = 5
            self.game_over = 0


if __name__ == '__main__':
    game = BreakoutGameAI()
    game.bucle_juego_prueba()
else: 
    game = BreakoutGameAI()