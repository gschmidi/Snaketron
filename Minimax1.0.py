# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:03:26 2019

@author: Paul Gaillard
"""
#import statistics
import numpy as np
from my_module import snake_env
import copy
#import sys
#sys.setrecursionlimit(10000)

def first_agent():
    env=snake_env(multiplayer=True,win_width=100,win_height=100)
    env.setup()
    reward1=0
    reward2=0
    snake_size=20
    done = False
    perdu1=0
    perdu2=0
    nbIte=4
    for i in range(5):
        action=np.random.randint(3)
        action2=np.random.randint(3)
        [observation, reward_temp, reward_temp2, done]=env.step(action,action2)
        reward1+=reward_temp
        reward2+=reward_temp2
        parts2 = copy.deepcopy(env.parts2)
        parts1 = copy.deepcopy(env.parts)
        sum0=0
        #print("Test 1")
        if(action!=2):
                for j in range(0,4):
                    if(action2!=j+2 or action2!=j-2):
                        sum0 += f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],parts1,observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],parts2,0,j,nbIte)
        sum1=0
        #print("Test 2")
        if(action!=3):
                for j in range(0,4):
                    if(action2!=j+2 or action2!=j-2):
                        sum1 += f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],parts1,observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],parts2,1,j,nbIte)
        sum2=0
        #print("Test 3")
        if(action!=0):
                for j in range(0,4):
                    if(action2!=j+2 or action2!=j-2):
                        sum2 += f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],parts1,observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],parts2,2,j,nbIte)
        sum3=0
        #print("Test 4")
        if(action!=1):
                for j in range(0,4):
                    if(action2!=j+2 or action2!=j-2):
                        sum3 += f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],parts1,observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],parts2,3,j,nbIte)
        
        print("SUM",sum0,sum1,sum2,sum3)
        '''x_dist = (observation[0]-observation[7])/snake_size
        y_dist = (observation[1]-observation[8])/snake_size
        if(abs(x_dist)>abs(y_dist)):
            if(x_dist>0):
                action=2
            else:
                action=0
        else:
            if(y_dist>0):
                action=3
            else:
                action=1
        print("-")
        print("S1 : Obs1 and 2",observation[0],observation[1])
        print("S2 : Action", action)
        print("S2 : Obs1 and 2",observation[7],observation[8])
        print("S2 : Action", action2)
        print("Dist x : ",x_dist,"Dist y : ",y_dist)
        print("Reward1", reward1,"Reward2", reward2)
        print("size 1 ", observation[6],"size 2 ", observation[11])'''
        if(env.snake.lost or env.snake2.lost):
            if(env.snake.lost and env.snake2.lost):
                print("los 2 han perdido")
            elif(env.snake.lost):
                #testt = f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],copy.deepcopy(env.parts),observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],copy.deepcopy(env.parts2),0,0,nbIte)
                print("1 ha perdido")
                perdu1+=1
            elif(env.snake2.lost):
                #testt = f(env,observation[6]*env.snake_size,observation[0],observation[1],observation[2],observation[3],copy.deepcopy(env.parts),observation[11]*env.snake_size,observation[7],observation[8],observation[9],observation[10],copy.deepcopy(env.parts2),0,0,nbIte)
                print("2 ha perdido ")
                perdu2+=1
            env=snake_env(multiplayer=True,win_width=200,win_height=200)
            env.setup()
    print("Perdu 1 : ",perdu1,"Perdu 2 : ",perdu2)
    #print("Stat 1 =",(stat1_test/stat1)*100,"%")
    #print("Stat 2 =",(stat2_test/stat2)*100,"%")
        
def f(env,s1_size,s1_x,s1_y,s1_vx,s1_vy,parts11,s2_size,s2_x,s2_y,s2_vx,s2_vy,parts22,action1,action2,i):

        i-=1
        if(i==0):
            return 0
        else:
            win=False
            lose=False
            #Snake1 move
            (parts1,s1_x,s1_y,s1_vx,s1_vy) = moveSnake(env,action1,parts11,s1_x,s1_y,s1_vx,s1_vy)
            #Snake2 move
            (parts2,s2_x,s2_y,s2_vx,s2_vy) = moveSnake(env,action2,parts22,s2_x,s2_y,s2_vx,s2_vy)
            
            #test if snake 2 lose
            for i in range(len(parts1)):
                if s2_x==parts1[i].x and s2_y==parts1[i].y:
                    win=True
                    
            #test if snake 1 lose
            for i in range(len(parts2)):
                if s1_x==parts2[i].x and s1_y==parts2[i].y:
                    lose=True
                    
            if win:
                return 1
            elif lose:
                return -1
            else:
                sumf=0
                for k in range(0,4):
                    if(action1!=k+2 or action1!=k-2):
                        for j in range(0,4):
                            if(action2!=j+2 or action2!=j-2):
                                sumf += f(env,s1_size,s1_x,s1_y,s1_vx,s1_vy,parts1,s2_size,s2_x,s2_y,s2_vx,s2_vy,parts2,k,j,i)
                return sumf

def moveSnake(env,action,parts,s_x,s_y,s_vx,s_vy):
        s_vx=0
        s_vy=0
        if(action==0):
            for i in range(len(parts)):
                parts[i].x -= env.snake_size
            s_x -= env.snake_size
            s_vx = -env.snake_size
        elif(action==1):
            for i in range(len(parts)):
                parts[i].y -= env.snake_size
            s_y -= env.snake_size
            s_vy = -env.snake_size
        elif(action==2):
            for i in range(len(parts)):
                parts[i].x += env.snake_size
            s_x += env.snake_size
            s_vx = env.snake_size
        elif(action==3):
            for i in range(len(parts)):
                parts[i].y += env.snake_size
            s_y += env.snake_size
            s_vy = env.snake_size
        return(parts,s_x,s_y,s_vx,s_vy)
                
first_agent()