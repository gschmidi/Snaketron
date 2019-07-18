# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:46:56 2019

@author: Asus X556
"""
import numpy as np
import pygame
from random import randrange, choice
import itertools 
from my_module_1 import snake_env
from collections import defaultdict

global action
def in_player(keys):
    #Action=0, 1, 2 or 3 corresponding to left, up, right, down
    
    if keys[pygame.K_LEFT]:
        return 0
    elif keys[pygame.K_UP]:
        return 1
    elif keys[pygame.K_RIGHT]:
        return 2
    elif keys[pygame.K_DOWN]:
        return 3
    

    

def in_player_WASD(keys):
    global action2
    if keys[pygame.K_a]:
        action2=0
    elif keys[pygame.K_w]:
        action2=1
    elif keys[pygame.K_d]:
        action2=2
    elif keys[pygame.K_s]:
        action2=3
        
 
def EpsilonGreedy(state, ep): 
    
    if randrange(100) <= (1-ep)*100:
        best_action = state.index(max(state))
    else:
        best_action = randrange(0,3)
    

 
   
    return best_action 
        

def qLearning(env, num_episodes, discount_factor = 0.5, 
                            alpha = 0.6, epsilon = 0.1):
    
    vel=10
    delay_penultimate_frame=1000
    delay_last_frame=1000
    time_delay=120 
    pygame.init()
    pygame.display.set_caption("Snaketron")
    win=pygame.display.set_mode((env.win_width,env.win_height)) #define window
    pygame.time.delay(500)
    
    State=0

       
  
    
    done = False
    Q={}
    
    for i in range(num_episodes): 
        num_step=0
        env.setup()
        state=tuple(env.observation)
        a = np.arange(10)

        Q[state]= np.random.choice(a, 4)
        action = randrange(4)

        while not done:  
            
            pygame.time.delay(time_delay-vel*10)

            [observation,reward, done,]= env.step(action)

            next_state=tuple(observation)
            if reward == 0:
                num_step += 1
                reward= -(0.1*max(next_state[8], next_state[9], next_state[10], next_state[11]))
                print(reward)
            else:
                num_step= 0
            #ingresa el estado si no está previamente 
            if Q.get(next_state) == None:
        
                Q[next_state]= [randrange(5),randrange(5),randrange(5), randrange(5)]

            next_action = EpsilonGreedy(Q[next_state],epsilon)
            Q[state][action]= Q[state][action] + alpha*(reward + discount_factor \
            *(Q[next_state][next_action])- Q[state][action])
            ##print (Q[state][action])
            
            action = next_action
            state=next_state
            #Actualizar estadado -Valor """
            if num_step == 500 or env.snake.lost:
                break
            pygame.draw.rect(win,(0,0,0),(0,0,env.win_width,env.win_height))
            if env.special_food_active:
              env.specialfood.draw(win)
            env.food.draw(win)
            env.snake.draw(win,r=180,g=70,b=0)
            for i in range(env.snake.num_parts):
                env.parts[i].draw(win,r=200,g=100,b=0)
                             
            pygame.display.update()
            time_temp=0

            for event in pygame.event.get():            
                if event.type ==pygame.QUIT: #Close window if X is pressed
                     done=True
          
            
    return Q 
          





"------------Main-------------"

env=snake_env(win_width=500,win_height=500,\
                  special_food_mode=True,special_food_time=40,special_food_frequency=5,\
                  multiplayer=False,tron_mode=True,special_food_frequency_multi=100,\
                  collision_with_head=True, simple_observation=False,\
                  periodic_boundaries_mode=True,no_tail_mode=False, POV_mode=True , starving_mode=False,\
                  starving_limit=100,zerosum=False)

pygame.init()
pygame.display.set_caption("Snaketron")
win=pygame.display.set_mode((env.win_width,env.win_height)) #define window
pygame.time.delay(500)

qLearning(env,1000)
pygame.quit()


