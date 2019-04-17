# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:29 2019

@author: Gerry
"""

import pygame
import numpy as np

"------Easily Adjustables------"
vel=5                           #velocity of snake from 0-10
snake_size=20                          
win_width=snake_size*30         #only multiples of snake_size
win_height=snake_size*20
special_food_time=40
special_food_mode=True
special_food_frequency=5         #only for single player

"Multiplayer"
multiplayer=False
tron_mode=True
special_food_frequency_multi=100 #[frames], must be >special_food_time
collision_with_head=True

"Visuals"
delay_penultimate_frame=1000
delay_last_frame=1000


"-----------Classes-----------"
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
        self.v_x_temp=v_x
        self.v_y_temp=v_y
        self.special_point=0
        self.lost=False
    def draw(self,win,reshape_factor=1,r=255,g=0,b=0):
        pygame.draw.rect(win,(r,g,b),(self.x,self.y,self.width*reshape_factor,self.height*reshape_factor))

class foods:
    def __init__(self,width,height):        
        self.width=width
        self.height=height
        
        new_food=True
        while new_food:
            x=np.random.randint(0,(win_width-snake_size)//snake_size)*snake_size
            y=np.random.randint(0,(win_height-snake_size)//snake_size)*snake_size
            new_food=False
            for i in range(snake.num_parts):
                    if x==parts[i].x and y==parts[i].y:
                        new_food=True
        self.x=x
        self.y=y
        
    def draw(self):    
        pygame.draw.rect(win,(0,150,0),(self.x,self.y,self.width,self.height))


class specialfoods:
    def __init__(self,width,height):
        self.width=width
        self.height=height
        new_special_food=True
        while new_special_food:
            x=np.random.randint(0,(win_width-snake_size)//snake_size)*snake_size
            y=np.random.randint(0,(win_height-snake_size)//snake_size)*snake_size
            new_special_food=False
            for i in range(snake.num_parts):
                    if x==parts[i].x and y==parts[i].y or x==food.x and y==food.y:
                        new_special_food=True
        self.x=x
        self.y=y
        
    def draw(self):    
        pygame.draw.rect(win,(0,255,255),(self.x,self.y,self.width,self.height))
        
"------------Functions----------------"

def check_keys_for_snake(snake,keys):
    if keys[pygame.K_LEFT] and snake.v_x==0:
        snake.v_x_temp=-snake_size
        snake.v_y_temp=0
    elif keys[pygame.K_UP] and snake.v_y==0:
        snake.v_y_temp=-snake_size
        snake.v_x_temp=0
    elif keys[pygame.K_RIGHT] and snake.v_x==0:
        snake.v_x_temp=snake_size
        snake.v_y_temp=0
    elif keys[pygame.K_DOWN] and snake.v_y==0:
        snake.v_y_temp=snake_size
        snake.v_x_temp=0

def update_snake(snake,parts):
        if snake.add_part:
            snake.num_parts+=1
            snake.add_part=False          
            add_part_to_snake(snake,parts)    
        snake.v_x=snake.v_x_temp
        snake.v_y=snake.v_y_temp                
        snake.x+=snake.v_x
        snake.y+=snake.v_y              
        snake.mov_x.append(snake.v_x)
        snake.mov_y.append(snake.v_y)
        for i in range(snake.num_parts):
                parts[i].v_x=snake.mov_x[-i-2]
                parts[i].v_y=snake.mov_y[-i-2]
            
        for i in range(snake.num_parts):
            parts[i].x+=parts[i].v_x #fix: example: 2 parts: mov_x=[2,1,snake]
            parts[i].y+=parts[i].v_y        
        del snake.mov_x[0]
        del snake.mov_y[0]

def add_part_to_snake(snake,parts): #requieres two objects
    if snake.num_parts==1:
        snake.mov_x.append(snake.v_x)
        snake.mov_y.append(snake.v_y) 
        parts.append(snakeparts(snake.x-snake.v_x,snake.y-snake.v_y,snake.width,snake.height,snake.v_x,snake.v_y))
    if snake.num_parts>1:                
        snake.mov_x.insert(0,snake.mov_x[0]) #fix: insert original value that gets deleted afterwords - process is therefor not interupted
        snake.mov_y.insert(0,snake.mov_y[0]) 
        parts.append(snakeparts(parts[-1].x-parts[-1].v_x,parts[-1].y-parts[-1].v_y,snake.width,snake.height,parts[-1].v_x,parts[-1].v_y))

def periodic_boundaries(snake,parts):
    if snake.x<0:
        snake.x=win_width-snake.width
    if snake.x>win_width-snake.width:
        snake.x=0
    if snake.y<0:
        snake.y=win_height-snake.height
    if snake.y>win_height-snake.height:
        snake.y=0       
    for k in range(snake.num_parts):
        if parts[k].x<0:
            parts[k].x=win_width-snake.width
        if parts[k].x>win_width-snake.width:
            parts[k].x=0
        if parts[k].y<0:
            parts[k].y=win_height-snake.height
        if parts[k].y>win_height-snake.height:
            parts[k].y=0            

def food_eaten_check(snake,food,special=False):
    global add_food, special_food_active
    if snake.x==food.x and snake.y==food.y:     
        snake.add_part=True
        if special:
            snake.add_part=True
            special_food_active=False
            snake.special_point+=1
        else:
            add_food=True
            
def special_food_fun(snake,parts):
    global special_count, special_food_active, specialfood, special_food_block
    food_eaten_check(snake,specialfood,special=True)   
    if len(parts)!=0 and np.mod(len(parts),special_food_frequency)==0 and special_food_block==False: 
        specialfood=specialfoods(snake_size,snake_size)
        special_food_active=True
        special_food_block=True
        special_count=0
    if np.mod(len(parts),special_food_frequency)==special_food_frequency-1:
        special_food_block=False           
    if special_count==special_food_time:
        special_food_active=False      
    special_count+=1
    
            
def special_food_multi(snake,parts,snake2,parts2):
    global special_count, special_food_active, specialfood, special_food_block
    food_eaten_check(snake,specialfood,special=True)   
    food_eaten_check(snake2,specialfood,special=True)
    if special_count==special_food_frequency_multi:
        specialfood=specialfoods(snake_size,snake_size)
        special_food_active=True
        special_count=0       
    if special_count==special_food_time:
        special_food_active=False      
    special_count+=1
    
def collision_check(snake,parts):
    global run
    for i in range(len(parts)):
        if snake.x==parts[i].x and snake.y==parts[i].y:
            snake.lost=True
            pygame.time.delay(delay_penultimate_frame)
            run=False

def check_keys_for_snake2(snake,keys):
    if keys[pygame.K_a] and snake2.v_x==0:
        snake2.v_x_temp=-snake_size
        snake2.v_y_temp=0
    elif keys[pygame.K_w] and snake2.v_y==0:
        snake2.v_y_temp=-snake_size
        snake2.v_x_temp=0
    elif keys[pygame.K_d] and snake2.v_x==0:
        snake2.v_x_temp=snake_size
        snake2.v_y_temp=0
    elif keys[pygame.K_s] and snake2.v_y==0:
        snake2.v_y_temp=snake_size
        snake2.v_x_temp=0
        
"-------------------------------Code--------------------------------------------"

pygame.init() #always inicialize
pygame.display.set_caption("My Snake")
win=pygame.display.set_mode((win_width,win_height)) #define window
pygame.time.delay(500)
mylist=pygame.display.list_modes

"-------Create Snakes and food------"         
snake=snakeparts(snake_size*4,snake_size*4,snake_size,snake_size,snake_size,0) #pos_x,pos_y,witdh,height,v_x,v_y
parts=list()
food=foods(snake_size,snake_size)
if multiplayer:
    snake2=snakeparts(snake_size*8,snake_size*8,snake_size,snake_size,snake_size,0)
parts2=list()
"-----Set useful variables-----"
add_part=False
time_delay=120                     #in ms
vel=vel*10
time0=pygame.time.get_ticks()       #T_inicial
run = True
special_food_active=False
add_food=False
wait_for_part=False
special_count=0
specialfood=specialfoods(snake_size,snake_size)
"-------Run game-----------"
while run:    
    pygame.time.delay(time_delay-vel)
    "-----Input------" 
    for event in pygame.event.get():            
        if event.type ==pygame.QUIT:                #Close window if X is pressed
            run=False
    keys =pygame.key.get_pressed()                  #Input
    check_keys_for_snake(snake,keys)
    
    if multiplayer:
        check_keys_for_snake2(snake2,keys)   
    if keys[pygame.K_SPACE]:                        #cheat to try out stuff
        snake.add_part=True        
    "---Computation---"
    if add_food:
        add_food=False
        food=foods(snake_size,snake_size)       #create new food if food has been eaten       
    if special_food_mode and not multiplayer:
        special_food_fun(snake,parts)       
    update_snake(snake,parts)
    periodic_boundaries(snake,parts)
    food_eaten_check(snake,food)         
    collision_check(snake,parts)                #Check if snake collides with itself
    if multiplayer:
        if special_food_mode:
            special_food_multi(snake,parts,snake2,parts2)
        update_snake(snake2,parts2)
        periodic_boundaries(snake2,parts2)
        food_eaten_check(snake2,food)         
        collision_check(snake2,parts2)
        if tron_mode:
            collision_check(snake,parts2)
            collision_check(snake2,parts)
            if collision_with_head:
                if snake.x==snake2.x and snake.y==snake2.y:
                    snake.lost=True
                    snake2.lost=True
                    pygame.time.delay(delay_penultimate_frame)
                    run=False
    "----Updating the screen-----"
    pygame.draw.rect(win,(0,0,0),(0,0,win_width,win_height))
    if special_food_active:
        specialfood.draw()
    food.draw()
    snake.draw(win,r=180,g=70,b=0)
    for i in range(snake.num_parts):
        parts[i].draw(win,r=200,g=100,b=0)
    if multiplayer:
        snake2.draw(win,r=0,g=0,b=180)
        for i in range(snake2.num_parts):
            parts2[i].draw(win,r=0,g=0,b=200)                   
    pygame.display.update()
    time_temp=0

time=(pygame.time.get_ticks()-time0)/1000 #time of programm in seconds          
if multiplayer:
    if snake.lost and snake2.lost:
        print("\nYou both are loosers!\n")    
    elif snake.lost:
        print("\nPlayer WASD won!\n")
    elif snake2.lost:
        print("\nPlayer ARROWS won!\n")
    print("Player ARROWS:")
    print("Snakelength: ", len(parts))
    if special_food_mode:
        print("Special Food Eaten: ", snake.special_point)
    print("Player WASD:")
    print("Snakelength: ", len(parts2))
    if special_food_mode:
        print("Special Food Eaten: ", snake2.special_point)   
else:
    print("Snakelength: ", len(parts))
    if special_food_mode:
        print("Special Food Eaten: ", snake.special_point)
        print("Your Score: ", (len(parts)*(snake.special_point+1)-int(time/2))*vel/4)
    else:
        print("Your Score: ", (len(parts)*3-int(time/2))*vel/4)            
pygame.time.delay(delay_last_frame)
pygame.quit()

















