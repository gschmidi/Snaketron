# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:09:22 2019

@author: Gerry
"""

import numpy as np
import pygame
"-------------------------------Code--------------------------------------------"

class snake_env(): 
    """
    Setup: Reset Game
    Step(action): Executes one frame following the action
        in:    
            -Action=0, 1, 2 or 3 corresponding to left, up, right, down
        out:
            -Observations[0:12]=[snake.x,snake.y,snake.v_x,snake.v_y,food.x,food.y, len(self.parts),snake2.x,snake2.y,\
                                snake2.v_x,snake2.v_y, len(parts2)]
            The rest of the observation vector correspond to the position of the snake tail
            if simple_observation==True, len(observation)=4, [snake.x,snake.y,food.x,food.y]
            -Reward
            -done=True if game ended         
    """
    def __init__ (self,snake_size=20,win_width=20*30,win_height=20*20,\
                  special_food_mode=True,special_food_time=40,special_food_frequency=5,\
                  multiplayer=False,tron_mode=True,special_food_frequency_multi=100,\
                  collision_with_head=True, simple_observation=False,periodic_boundaries_mode=True):
        "Given Variables"
        self.snake_size=snake_size                        
        self.win_width=win_width        #only multiples of snake_size
        self.win_height=win_height
        self.special_food_time=special_food_time
        self.special_food_mode=special_food_mode
        self.special_food_frequency=special_food_frequency        #only for single player
        self.simple_observation=simple_observation
        self.multiplayer=multiplayer
        self.tron_mode=tron_mode
        self.special_food_frequency_multi=special_food_frequency_multi #[frames], must be >special_food_time
        self.collision_with_head=collision_with_head
        self.periodic_boundaries_mode=periodic_boundaries_mode
        "Info"
        self.num_action=4
        
    def setup(self):
        "-------Create Snakes and food------"         
        self.snake=self.snakeparts(self.snake_size*4,self.snake_size*4,self.snake_size,self.snake_size,self.snake_size,0) #pos_x,pos_y,witdh,height,v_x,v_y
        self.snake.num_parts=0
        self.parts=list()
        self.food=self.foods(self.snake_size,self.snake_size,self.win_width,\
                                 self.win_height,self.snake_size,self.snake.num_parts,\
                                 self.snake.x,self.snake.y,self.parts)  
        if self.multiplayer:
            self.snake2=self.snakeparts(self.snake_size*5,self.snake_size*12,self.snake_size,self.snake_size,self.snake_size,0)
        self.parts2=list()
        "-----Set useful variables-----"      #T_inicial
        self.add_part=False
        self.add_part2=False
        self.run = True
        self.special_food_active=False
        self.add_food=False
        self.wait_for_part=False
        self.special_count=0
        if self.special_food_mode:
            self.specialfood=self.specialfoods(self.snake_size,self.snake_size,self.win_width,self.win_height,\
                                               self.snake_size,self.snake.num_parts,self.parts,self.food)
        
        "AI"
        "output"
        self.observation=np.zeros(int(2*self.win_height*self.win_width/self.snake_size**2))#*2 because 2 position arguments
        self.s_ob=np.zeros(4)
        self.reward=0
        self.done=False
        if self.multiplayer:
            self.reward2=0

    "-------Run one frame of game-----------"
    def step(self,action, action2=0):
        self.action=action
        self.action2=action2
        "-----Input------" 
        self.snake=self.bot_input(self.snake,self.action)
        if self.multiplayer:
            self.snake2=self.bot_input(self.snake2,self.action2)
            self.reward2=0
            self.reward2_col=0
        self.reward=0
        self.reward_col=0

        "---Computation---"
        if self.add_food:
            self.add_food=False
            self.food=self.foods(self.snake_size,self.snake_size,self.win_width,\
                                 self.win_height,self.snake_size,self.snake.num_parts,\
                                 self.snake.x,self.snake.y,self.parts)     #creates new food if food has been eaten       
        if self.special_food_mode and not self.multiplayer:
            self.special_food_fun(self.snake,self.parts)       
        [self.snake,self.parts]=self.update_snake(self.snake,self.parts)
        [self.snake,self.food,self.reward]=self.food_eaten_check(self.snake,self.food) 
        if self.periodic_boundaries_mode:
            [self.snake,self.parts]=self.periodic_boundaries(self.snake,self.parts)
        else:
            [self.snake,self.parts,self.reward_col]=self.no_periodic_boundaries(self.snake,self.parts)               
            self.reward+=self.reward_col 
        [self.snake,self.parts,self.reward_col]=self.collision_check(self.snake,self.parts)                #Check if snake collides with itself
        self.reward+=self.reward_col #Sum up rewards from food with collisions
        if self.multiplayer:
            if self.special_food_mode:
                self.special_food_multi(self.snake,self.parts,self.snake2,self.parts2)
            [self.snake2,self.parts2]=self.update_snake(self.snake2,self.parts2)
            [self.snake2,self.food,self.reward2]=self.food_eaten_check(self.snake2,self.food)   
            if self.periodic_boundaries_mode:
                [self.snake2,self.parts2]=self.periodic_boundaries(self.snake2,self.parts2)
            else:
                [self.snake2,self.parts2,self.reward2_col]=self.no_periodic_boundaries(self.snake2,self.parts2)                        
                self.reward2+=self.reward2_col #Sum up rewards from food with collisions
            [self.snake2,self.parts2,self.reward2_col]=self.collision_check(self.snake2,self.parts2)
            self.reward2+=self.reward2_col #Sum up rewards from food with collisions            
            if self.tron_mode:
                [self.snake,self.parts2,self.reward_col]=self.collision_check(self.snake,self.parts2)
                [self.snake2,self.parts,self.reward2_col]=self.collision_check(self.snake2,self.parts)
                self.reward2+=self.reward2_col
                self.reward+=self.reward_col
                if self.collision_with_head:
                    if self.snake.x==self.snake2.x and self.snake.y==self.snake2.y:
                        self.snake.lost=True
                        self.snake2.lost=True
                        self.run=False
                        self.done=True
            "ZeroSumGame"            
            reward2_temp=self.reward2
            self.reward2-=self.reward
            self.reward-=reward2_temp
        
        "--Observation--and return"
        if self.multiplayer == False:
            self.observation[0:7]=[self.snake.x,self.snake.y,self.snake.v_x,self.snake.v_y,\
                            self.food.x,self.food.y, len(self.parts)]
            for i in range(0,len(self.parts)): 
                self.observation[7+i]=self.parts[i].x
                self.observation[-i-1]=self.parts[i].y #write at end because doesn't matter where in list
            if self.simple_observation:
                self.s_ob[0:4]=[self.snake.x,self.snake.y,self.food.x,self.food.y]
                return self.s_ob, self.reward, self.done
            else:
                return self.observation, self.reward, self.done
        else:
            self.observation[0:12]=[self.snake.x,self.snake.y,self.snake.v_x,self.snake.v_y,\
                            self.food.x,self.food.y, len(self.parts),self.snake2.x,self.snake2.y,\
                            self.snake2.v_x,self.snake2.v_y, len(self.parts2)]  
            for i in range(0,len(self.parts)): 
                self.observation[12+2*i]=self.parts[i].x
                self.observation[-2*i-1]=self.parts[i].y
            for i in range(0,len(self.parts2)):
                self.observation[13+2*i]=self.parts2[i].x
                self.observation[-2*i-2]=self.parts2[i].y
        
            return self.observation, self.reward, self.reward2, self.done
            
       
    def bot_input(self,snake,action):
        if action==0 and snake.v_x==0: #0,1,2,3 <-> l,u,r,d
            snake.v_x=-self.snake_size
            snake.v_y=0
        elif action==1 and snake.v_y==0:
            snake.v_y=-self.snake_size
            snake.v_x=0
        elif action==2 and snake.v_x==0:
            snake.v_x=self.snake_size
            snake.v_y=0
        elif action==3 and snake.v_y==0:
            snake.v_y=self.snake_size
            snake.v_x=0
        return snake
    
    def update_snake(self,s,p):
            if s.add_part:
                s.num_parts+=1
                s.add_part=False          
                [s,p]=self.add_part_to_snake(s,p)                   
            s.x+=s.v_x
            s.y+=s.v_y              
            s.mov_x.append(s.v_x)
            s.mov_y.append(s.v_y)
            for i in range(s.num_parts):
                    p[i].v_x=s.mov_x[-i-2]
                    p[i].v_y=s.mov_y[-i-2]                
            for i in range(s.num_parts):
                p[i].x+=p[i].v_x 
                p[i].y+=p[i].v_y        
            del s.mov_x[0]
            del s.mov_y[0]
            return s, p
    
    def add_part_to_snake(self,sn,pa): 
        if sn.num_parts==1:
            sn.mov_x.append(sn.v_x)
            sn.mov_y.append(sn.v_y) 
            pa.append(self.snakeparts(sn.x-sn.v_x,sn.y-sn.v_y,sn.width,sn.height,sn.v_x,sn.v_y))
        if sn.num_parts>1:                
            sn.mov_x.insert(0,sn.mov_x[0])
            sn.mov_y.insert(0,sn.mov_y[0]) 
            pa.append(self.snakeparts(pa[-1].x-pa[-1].v_x,pa[-1].y-pa[-1].v_y,sn.width,sn.height,pa[-1].v_x,pa[-1].v_y))
        return sn, pa
    def periodic_boundaries(self,snake,parts):
        if snake.x<0:
            snake.x=self.win_width-snake.width
        if snake.x>self.win_width-snake.width:
            snake.x=0
        if snake.y<0:
            snake.y=self.win_height-snake.height
        if snake.y>self.win_height-snake.height:
            snake.y=0   
        for k in range(snake.num_parts):
            if parts[k].x<0:
                parts[k].x=self.win_width-snake.width
            if parts[k].x>self.win_width-snake.width:
                parts[k].x=0
            if parts[k].y<0:
                parts[k].y=self.win_height-snake.height
            if parts[k].y>self.win_height-snake.height:
                parts[k].y=0    
        return snake, parts
        
    def no_periodic_boundaries(self,snake,parts):
        reward=0
        if snake.x<0 or snake.x>self.win_width-snake.width or snake.y<0 or snake.y>self.win_height-snake.height:
            snake.lost=True
            self.run=False
            reward=-5
            self.done=True
        return snake, parts, reward
    
    def food_eaten_check(self,snake,food,special=False):
        self.special=special
        reward=0

        if special:
            if snake.x==food.x and snake.y==food.y and self.special_food_active:                 
                snake.add_part=True
                self.special_food_active=False
                snake.special_point+=1
                reward=1
        else:
            if snake.x==food.x and snake.y==food.y:  
                self.add_food=True
                snake.add_part=True
                reward=1

        return snake,food,reward
    def collision_check(self,snake,parts):
        reward=0
        for i in range(len(parts)):
            if snake.x==parts[i].x and snake.y==parts[i].y:
                snake.lost=True
                self.run=False
                reward=-5
                self.done=True
        return snake, parts, reward
                
    def special_food_fun(self,snake,parts):
        self.food_eaten_check(snake,self.specialfood,special=True)   
        if len(parts)!=0 and np.mod(len(parts),self.special_food_frequency)==0 and self.special_food_block==False: 
            self.specialfood=self.specialfoods(self.snake_size,self.snake_size,self.win_width,self.win_height,\
                                               self.snake_size,snake.num_parts,parts,self.food)
            self.special_food_active=True
            self.special_food_block=True
            self.special_count=0
        if np.mod(len(parts),self.special_food_frequency)==self.special_food_frequency-1:
            self.special_food_block=False           
        if self.special_count==self.special_food_time:
            self.special_food_active=False      
        self.special_count+=1
        
                
    def special_food_multi(self,snake,parts,snake2,parts2):
        self.food_eaten_check(snake,self.specialfood,special=True)   
        self.food_eaten_check(snake2,self.specialfood,special=True)
        if self.special_count==self.special_food_frequency_multi:
            self.specialfood=self.specialfoods(self.snake_size,self.snake_size,self.win_width,self.win_height,\
                                               self.snake_size,snake.num_parts,parts,self.food)
            self.special_food_active=True
            self.special_count=0       
        if self.special_count==self.special_food_time:
            self.special_food_active=False      
        self.special_count+=1
        
    class specialfoods:
        def __init__(self,width,height,win_width,win_height,snake_size,num_parts,parts,food):
            self.width=width
            self.height=height
            new_special_food=True
            while new_special_food:
                x=np.random.randint(0,(win_width-snake_size)//snake_size)*snake_size
                y=np.random.randint(0,(win_height-snake_size)//snake_size)*snake_size
                new_special_food=False
                for i in range(num_parts):
                        if x==parts[i].x and y==parts[i].y \
                        or x==food.x and y==food.y:
                            new_special_food=True
            self.x=x
            self.y=y
        def draw(self,win):    
            pygame.draw.rect(win,(0,255,255),(self.x,self.y,self.width,self.height))
        
    class snakeparts:
        def __init__(self,x,y,width,height,v_x,v_y):
            self.x=x
            self.y=y
            self.width=width
            self.height=height
            self.v_x=v_x
            self.v_y=v_y        
            self.mov_x=[]#remember moves
            self.mov_y=[]
            self.mov_x.append(self.v_x)
            self.mov_y.append(self.v_y)
            self.num_parts=0
            self.add_part=False
            self.special_point=0
            self.lost=False
        def draw(self,win,reshape_factor=1,r=255,g=0,b=0):
            pygame.draw.rect(win,(r,g,b),(self.x,self.y,self.width*reshape_factor,self.height*reshape_factor))


    class foods:
        def __init__(self,width,height,win_width,win_height,snake_size,num_parts,x,y,parts):
            self.width=width
            self.height=height

            new_food=True
            while new_food:
                x=np.random.randint(0,(win_width-snake_size)//snake_size)*snake_size
                y=np.random.randint(0,(win_height-snake_size)//snake_size)*snake_size
                new_food=False
                for i in range(num_parts):
                        if x==parts[i].x and y==parts[i].y:
                            new_food=True
            self.x=x
            self.y=y
        def draw(self,win):    
            pygame.draw.rect(win,(0,150,0),(self.x,self.y,self.width,self.height))


