# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:15:32 2019

@author: Gerry
"""
import numpy as np
import pygame
from my_module import snake_env

def in_player(keys,action):
    #Action=0, 1, 2 or 3 corresponding to left, up, right, down
#    global action
    if keys[pygame.K_LEFT]:
        action=0
    elif keys[pygame.K_UP]:
        action=1
    elif keys[pygame.K_RIGHT]:
        action=2
    elif keys[pygame.K_DOWN]:
        action=3
    return action

def in_player_WASD(keys, action2):
#    global action2
    if keys[pygame.K_a]:
        action2=0
    elif keys[pygame.K_w]:
        action2=1
    elif keys[pygame.K_d]:
        action2=2
    elif keys[pygame.K_s]:
        action2=3
    return action2


def reflexAgent(obs,player):#input observation
    #obs[0]=snake.x,obs[1]=snake.y, obs[4]=food.x, obs[5]=food.y
    if player==1:
        i=7
    elif player==2:
        i=0
    else:
        print("Invalid Player!")
    action=2
    if obs[4]==obs[7-i]:
        action=1    
    return action

def render():
    "Adjustables"
    opponent_is_Bot=True
    player1_is_Bot=True
    env=snake_env(multiplayer=True,simple_observation=True,\
                  periodic_boundaries_mode=True, win_width=20*10,win_height=20*10)
    vel=1
    delay_last_frame=1000
    time_delay=120 
    
    
    pygame.init() 
    pygame.display.set_caption("Snaketron")
    win=pygame.display.set_mode((env.win_width,env.win_height))
    pygame.time.delay(500)
    
    env.setup()
    observation=env.observation
    done=False
    action=2
    action2=2
    while not done:    
        pygame.time.delay(time_delay-vel*10)
        "-----Input------" 
        keys =pygame.key.get_pressed()                
        if player1_is_Bot:
            action=np.random.randint(0,4)
            #action=reflexAgent(observation,1)
        else:
            action=in_player(keys,action)
        if env.multiplayer and not opponent_is_Bot:
            action2=in_player_WASD(keys,action2)
        elif env.multiplayer and opponent_is_Bot:
            action2=np.random.randint(0,4)
            #action2=reflexAgent(observation,2)
        if not env.multiplayer:   
            [observation,reward,done]=env.step(action)
        else:
            [observation,reward,reward2,done]=env.step(action,action2)
               
        pygame.draw.rect(win,(0,0,0),(0,0,env.win_width,env.win_height))
        if env.special_food_active:
            env.specialfood.draw(win)
        env.food.draw(win)
        env.snake.draw(win,r=180,g=70,b=0)
        for i in range(env.snake.num_parts):
            env.parts[i].draw(win,r=200,g=100,b=0)
        if env.multiplayer:
            env.snake2.draw(win,r=0,g=0,b=180)
            for i in range(env.snake2.num_parts):
                env.parts2[i].draw(win,r=0,g=0,b=200)                   
        pygame.display.update()
        for event in pygame.event.get():            
            if event.type ==pygame.QUIT:                #Close window if X is pressed
                done=True
    print("\nSnake Length Player1: ", len(env.parts))
    if env.multiplayer:
        print("Snake Length Player2: ", len(env.parts2))
        if  env.snake.lost and  env.snake2.lost:
            print("\nBoth lost!")
        elif env.snake.lost:
            print("\nPlayer2 won!")
        elif env.snake2.lost:
            print("\nPlayer1 won!")
    
    pygame.time.delay(delay_last_frame)
    pygame.quit()

render()