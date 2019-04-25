# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:29 2019

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
            -Reward
            -done=True if game ended
    """
    def __init__ (self,snake_size=20,win_width=20*30,win_height=20*20,\
                  special_food_mode=True,special_food_time=40,special_food_frequency=5,\
                  multiplayer=False,tron_mode=True,special_food_frequency_multi=100,\
                  collision_with_head=True):
        "Given Variables"
        self.snake_size=snake_size                        
        self.win_width=win_width        #only multiples of snake_size
        self.win_height=win_height
        self.special_food_time=special_food_time
        self.special_food_mode=special_food_mode
        self.special_food_frequency=special_food_frequency        #only for single player

        self.multiplayer=multiplayer
        self.tron_mode=tron_mode
        self.special_food_frequency_multi=special_food_frequency_multi #[frames], must be >special_food_time
        self.collision_with_head=collision_with_head
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
        self.reward=0

        "---Computation---"
        if self.add_food:
            self.add_food=False
            self.food=self.foods(self.snake_size,self.snake_size,self.win_width,\
                                 self.win_height,self.snake_size,self.snake.num_parts,\
                                 self.snake.x,self.snake.y,self.parts)     #creates new food if food has been eaten       
        if self.special_food_mode and not self.multiplayer:
            self.special_food_fun(self.snake,self.parts)       
        [self.snake,self.parts]=self.update_snake(self.snake,self.parts)
        [self.snake,self.parts]=self.periodic_boundaries(self.snake,self.parts)
        [self.snake,self.food,self.reward]=self.food_eaten_check(self.snake,self.food)         
        [self.snake,self.parts,self.reward]=self.collision_check(self.snake,self.parts)                #Check if snake collides with itself
        if self.multiplayer:
            if self.special_food_mode:
                self.special_food_multi(self.snake,self.parts,self.snake2,self.parts2)
            [self.snake2,self.parts2]=self.update_snake(self.snake2,self.parts2)
            [self.snake2,self.parts2]=self.periodic_boundaries(self.snake2,self.parts2)
            [self.snake2,self.food,self.reward2]=self.food_eaten_check(self.snake2,self.food)         
            [self.snake2,self.parts2,self.reward2]=self.collision_check(self.snake2,self.parts2)
            if self.tron_mode:
                [self.snake,self.parts2,self.reward2]=self.collision_check(self.snake,self.parts2)
                [self.snake2,self.parts,self.reward2]=self.collision_check(self.snake2,self.parts)
                if self.collision_with_head:
                    if self.snake.x==self.snake2.x and self.snake.y==self.snake2.y:
                        self.snake.lost=True
                        self.snake2.lost=True
                        self.run=False
                        self.done=True
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
        
    
    def food_eaten_check(self,snake,food,special=False):
        self.special=special
        reward=0
        if snake.x==food.x and snake.y==food.y:     
            snake.add_part=True
            reward=1
            if special:
                snake.add_part=True
                self.special_food_active=False
                snake.special_point+=1
            else:
                self.add_food=True
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



"------Example Random Bot Singleplayer-------"
#rewards=0
#dones=0
#env=snake_env(multiplayer=False)
#env.setup()
#for i in range(50000):
#    [observation,reward,done]=env.step(np.random.randint(0,3))
#    rewards+=reward
#    if done:
#        env.setup()
#        dones+=1

           
"-----Example Random Bot Multiplayer--------"
#rewards=0
#rewards2=0
#dones=0
#env=snake_env(multiplayer=True,tron_mode=False)
#env.setup()
#for i in range(20000):
#    [observation,reward,reward2,done]=env.step(np.random.randint(0,3),np.random.randint(0,3))
#    rewards+=reward
#    rewards2+=reward2
#    if done:
#        env.setup()
#        dones+=1


"----------Rendered Game-------------------"

def in_player(keys):
    #Action=0, 1, 2 or 3 corresponding to left, up, right, down
    global action
    if keys[pygame.K_LEFT]:
        action=0
    elif keys[pygame.K_UP]:
        action=1
    elif keys[pygame.K_RIGHT]:
        action=2
    elif keys[pygame.K_DOWN]:
        action=3
    return action

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

"------------Main-------------"
opponent_is_Bot=True
player1_is_Bot=False
env=snake_env(multiplayer=True)
vel=5
delay_penultimate_frame=1000
delay_last_frame=1000
time_delay=120 

pygame.init() 
pygame.display.set_caption("Snaketron")
win=pygame.display.set_mode((env.win_width,env.win_height)) #define window
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
        action=reflexAgent(observation,1)
    else:
        in_player(keys)
    if env.multiplayer and not opponent_is_Bot:
        in_player_WASD(keys)
    elif env.multiplayer and opponent_is_Bot:
        action2=reflexAgent(observation,2)
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
    time_temp=0
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
 
