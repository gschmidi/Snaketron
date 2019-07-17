# -*- coding: utf-8 -*-
"""
Created on Thu May 23 23:27:57 2019

@author: Gerry
"""
#import time
import numpy as np
import pygame
from my_module import snake_env

def reflex_survive(pov,distance=1):
    """obs=(distances to snakeparts: right,down,up,left,\
                                     distances to end of screen: right,down,up,left\
                                     diagonal distances to snakeparts:rightup,rightdown,leftup,leftdown\
                                     distance to food: right,left,down,up\
                                     distance to special food:right,left,down,up)
    Action=0, 1, 2 or 3 corresponding to left, up, right, down"""
    action=2
#    action=np.random.randint(3)
    #"Corners"
    if pov[6]==distance and pov[4]==distance:
        action=0
    elif pov[7]==distance and pov[6]==distance:
        action=3
    elif pov[7]==distance and pov[5]==distance:
        action=2  
    elif pov[4]==distance and pov[5]==distance:
        action=1
    #"Tails"
    #surviving
    elif pov[0]==distance and pov[2]==distance and pov[3]==distance:
        action=3
    elif pov[0]==distance and pov[1]==distance and pov[3]==distance:
        action=1
    elif pov[0]==distance and pov[1]==distance and pov[2]==distance:
        action=0
    elif pov[1]==distance and pov[2]==distance and pov[3]==distance:
        action=2
    #"obstacle
    elif pov[0]==distance and pov[1]==distance:
        action=1
    elif pov[0]==distance and pov[2]==distance:
        action=0
    elif pov[0]==distance and pov[3]==distance:
        action=1
    elif pov[1]==distance and pov[2]==distance:
        action=0
    elif pov[1]==distance and pov[3]==distance:
        action=1
    elif pov[2]==distance and pov[3]==distance:
        action=2
    #"Mid_field"
    elif pov[7]==distance and pov[5]==0.25:
        action=2 
    elif pov[6]==0.25 and pov[4]==distance:
        action=0        
    #"Screen_ends"
    elif pov[4]==distance:
        action=1
    elif pov[5]==distance:
        action=2
    elif pov[6]==distance:
        action=0        
    elif pov[7]==distance:
        action=3
    #go forward
    elif pov[0]==distance:
        action=0
    elif pov[1]==distance:
        action=1
    elif pov[2]==distance:
        action=3
    elif pov[3]==distance:
        action=2     
    return action

def prob_action(q):
    action=2
    r=np.random.rand()
    if r < q[0]:
        action=0
    elif r < q[0]+q[1]:
        action=1
    elif r < q[0]+q[1]+q[2]:
        action=2
    elif r < q[0]+q[1]+q[2]+q[3]:
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

def format_simple_obs(obs):
    if np.ndim(obs)==1:
        obs[:4]=obs[:4]/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
        obs[4:6]=obs[4:6]/snake_size/2+0.5       
    elif np.ndim(obs)==2:
        obs[:,:4]=obs[:,:4]/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
        obs[:,4:6]=obs[:,4:6]/snake_size/2+0.5
    else:
        print("Wrong Dimesion of Observation: Cannot format")
    return obs

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def create_population(n):
    w1=(np.random.rand(n,input_nodes,hidden_nodes)-0.5)*2
    b1=(np.random.rand(n,hidden_nodes)-0.5)*2
    w2=(np.random.rand(n,hidden_nodes,output_nodes)-0.5)*2
    b2=(np.random.rand(n,output_nodes)-0.5)*2
    return w1,w2,b1,b2

def predict(w1,w2,b1,b2,x):  
    hidden_out=np.maximum(0,x@w1+b1) #maximum = Relu
    y=softmax(hidden_out@w2+b2)
    return y

def max_action(q):
    for i in range(4):
        if q[i]>0.5:
            action=i
            return action
    for i in range(4):
        if q[i]>=q[0] and q[i]>=q[1] and q[i]>=q[2] and q[i]>=q[3]:
            action=i
            return action
    print("max_action Error")

def selection(rews):
    #q=softmax(rews)
    q=rews**3
    for k in range(len(q)-1):
        q[k+1]=q[k+1]+q[k]
    selected=np.zeros(len(rews))
    for k in range(len(rews)):
        r=np.random.rand()*q[-1]
        for i in range(len(q)):
            if r<q[i]:
                selected[k]=i
                break
    return selected

def clone(w1,w2,b1,b2,selected):
    w1_new=np.zeros_like(w1)
    w2_new=np.zeros_like(w2)
    b1_new=np.zeros_like(b1)
    b2_new=np.zeros_like(b2)
    for i in range(len(selected)):
        w1_new[i]=w1[int(selected[i])]
        w2_new[i]=w2[int(selected[i])]
        b1_new[i]=b1[int(selected[i])]
        b2_new[i]=b2[int(selected[i])]
    return w1_new,w2_new,b1_new,b2_new

def crossover(w1,w2,b1,b2):
    n=len(w1)
    w1_new=np.copy(w1)
    w2_new=np.copy(w2)
    b1_new=np.copy(b1)
    b2_new=np.copy(b2)
    for i in range(n):
        T=np.random.rand(w1.shape[1],w1.shape[2])<0.5
        w1_new[i,T]=w1[np.random.randint(n),T]
        T=np.random.rand(w2.shape[1],w2.shape[2])<0.5
        w2_new[i,T]=w2[np.random.randint(n),T]
        T=np.random.rand(b1.shape[1])<0.5
        b1_new[i,T]=b1[np.random.randint(n),T]
        T=np.random.rand(b2.shape[1])<0.5
        b2_new[i,T]=b2[np.random.randint(n),T]
    return w1_new,w2_new,b1_new,b2_new

def mutate(w1,w2,b1,b2,n,mutation_strength,mutation_rate,tweak_only=True):
    #Overwrite one Node completely    
    if not tweak_only:
        T=np.random.rand(w1.shape[0],w1.shape[1],w1.shape[2])<mutation_rate #random # to choose
        R=(np.random.rand(w1.shape[0],w1.shape[1],w1.shape[2])-0.5)*2 #random numbers to replace weights
        w1[T]=R[T]
        T=np.random.rand(w2.shape[0],w2.shape[1],w2.shape[2])<mutation_rate #random # to choose
        R=(np.random.rand(w2.shape[0],w2.shape[1],w2.shape[2])-0.5)*2 #random numbers to replace weights
        w2[T]=R[T]
        T=np.random.rand(b1.shape[0],b1.shape[1])<mutation_rate #random # to choose
        R=(np.random.rand(b1.shape[0],b1.shape[1])-0.5)*2 #random numbers to replace weights
        b1[T]=R[T]
        T=np.random.rand(b2.shape[0],b2.shape[1])<mutation_rate #random # to choose
        R=(np.random.rand(b2.shape[0],b2.shape[1])-0.5)*2 #random numbers to replace weights
        b2[T]=R[T]
        return w1,w2,b1,b2
    else:
        T=np.random.rand(w1.shape[0],w1.shape[1],w1.shape[2])<mutation_rate #random # to choose
        R=np.random.rand(w1.shape[0],w1.shape[1],w1.shape[2]) #random numbers to replace weights
        w1[T]=w1[T]+(R[T]-0.5)*2*mutation_strength
        T=np.random.rand(w2.shape[0],w2.shape[1],w2.shape[2])<mutation_rate #random # to choose
        R=np.random.rand(w2.shape[0],w2.shape[1],w2.shape[2]) #random numbers to replace weights
        w2[T]=w2[T]+(R[T]-0.5)*2*mutation_strength
        T=np.random.rand(b1.shape[0],b1.shape[1])<mutation_rate #random # to choose
        R=np.random.rand(b1.shape[0],b1.shape[1]) #random numbers to replace weights
        b1[T]=b1[T]+(R[T]-0.5)*2*mutation_strength    
        T=np.random.rand(b2.shape[0],b2.shape[1])<mutation_rate #random # to choose
        R=np.random.rand(b2.shape[0],b2.shape[1]) #random numbers to replace weights
        b2[T]=b2[T]+(R[T]-0.5)*2*mutation_strength    
    return w1,w2,b1,b2

def save_best(w1,w2,b1,b2,rews):
    idx=np.argmax(rews)
    w1_b=w1[idx]
    w2_b=w2[idx]
    b1_b=b1[idx]
    b2_b=b2[idx]
    return w1_b, w2_b, b1_b, b2_b

def keep_best_alive(w1,w2,b1,b2,w1_best,w2_best,b1_best,b2_best):
    w1[0]=w1_best
    w2[0]=w2_best
    b1[0]=b1_best
    b2[0]=b2_best
    return w1,w2,b1,b2

def add_n_dimensions(w1,w2,b1,b2,n):
    w1_new=np.zeros([n,input_nodes,hidden_nodes])
    b1_new=np.zeros([n,hidden_nodes])
    w2_new=np.zeros([n,hidden_nodes,output_nodes])
    b2_new=np.zeros([n,output_nodes])
    for i in range(n):
        w1_new[i]=w1
        w2_new[i]=w2
        b1_new[i]=b1
        b2_new[i]=b2
    return w1_new, w2_new, b1_new,b2_new

def generate(use_played):
    if use_played:
        w1=np.load("played_19nodes_5epochs_w1.npy")
        w2=np.load("played_19nodes_5epochs_w2.npy")
        b1=np.load("played_19nodes_5epochs_b1.npy")
        b2=np.load("played_19nodes_5epochs_b2.npy")
        w1,w2,b1,b2=add_n_dimensions(w1,w2,b1,b2,n)
    else:
        w1,w2,b1,b2=create_population(n)
    return w1,w2,b1,b2

def check_selected(selected,rews):
    rews_sel=np.copy(rews)
    for i in range(len(rews)):
        rews_sel[i]=rews[int(selected[i])]
    return rews_sel

def mix_in_genes(w1,w2,b1,b2,w1_in,w2_in,b1_in,b2_in):
    for i in range(len(w1)):
        T=np.random.rand(w1.shape[1],w1.shape[2])<0.5
        w1[i,T]=w1_in[np.random.randint(n),T]
        T=np.random.rand(w2.shape[1],w2.shape[2])<0.5
        w2[i,T]=w2_in[np.random.randint(n),T]
        T=np.random.rand(b1.shape[1])<0.5
        b1[i,T]=b1_in[np.random.randint(n),T]
        T=np.random.rand(b2.shape[1])<0.5
        b2[i,T]=b2_in[np.random.randint(n),T]
    return w1,w2,b1,b2

def playgame(env,w1_1,w2_1,b1_1,b2_1):
    delay_last_frame=1000
    time_delay=120 
    rews=0
    rews2=0    
    pygame.init() 
    pygame.display.set_caption("Snaketron")
    win=pygame.display.set_mode((env.win_width,env.win_height))
    pygame.time.delay(500)   
    observation=env.setup()
    done=False
    confused=False
    action=2
    action2=2
    while not done:    
        pygame.time.delay(time_delay-vel*10)
        "-----Input------" 
        keys =pygame.key.get_pressed()                
        if player1_is_Bot:
            if simple:
                observation=format_simple_obs(observation)
            q = predict(w1_1,w2_1,b1_1,b2_1,observation[:20]) #normalize obs
            if np.max(q)<0.6:
                confused=True
            action=np.argmax(q)
        else:
            action=in_player(keys,action)
        if env.multiplayer and not opponent_is_Bot:
            action2=in_player_WASD(keys,action2)
        elif env.multiplayer and opponent_is_Bot:
            q = predict(w1_2,w2_2,b1_2,b2_2,observation[20:40]) #normalize obs
#            if np.max(q)<0.6:
#                confused2=True
            action2=np.argmax(q)
#            action2=reflex_survive(observation[20:40])            
            print(observation[20:24])
        if not env.multiplayer:   
            [observation,reward,done]=env.step(action)
        else:
            [observation,reward,reward2,done]=env.step(action,action2)
        pygame.draw.rect(win,(0,0,0),(0,0,env.win_width,env.win_height))
        if env.special_food_active:
            env.specialfood.draw(win)
        env.food.draw(win)   
        if not confused:
            env.snake.draw(win,r=180,g=70,b=0)
            for i in range(env.snake.num_parts):
                env.parts[i].draw(win,r=200,g=100,b=0)
        else:
            confused=False
            env.snake.draw(win,r=180,g=0,b=20)
            for i in range(env.snake.num_parts):
                env.parts[i].draw(win,r=200,g=0,b=50)
        if env.multiplayer:
#            if not confused2:
#                env.snake2.draw(win,r=0,g=30,b=150)
#                for i in range(env.snake.num_parts):
#                    env.parts2[i].draw(win,r=200,g=100,b=0)
#            else:
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

def life(n,max_frames=10**4,multi=False):
    count=0
    won=0
    draw=0
    rews=np.zeros(n)+6#sp it starts at 1
    for i in range(n):
        frame=0
        done=False
        obs=env.setup()
        while not done:
            frame+=1
            if simple:
                obs=format_simple_obs(obs)
            q=predict(w1[i],w2[i],b1[i],b2[i],obs[:20])
#            action=max_action(q)
            action=np.argmax(q)
#            action=prob_action(q)
            if not multi:
                obs,rew,done=env.step(action)
            elif multi:
#                q=predict(w1_2,w2_2,b1_2,b2_2,obs[20:40])
#                action2=np.argmax(q)
                action2=reflex_survive(obs[20:40])
                obs,rew,_,done=env.step(action,action2)
            rews[i]+=rew
            if survive_reward:
                rews[i]+=0.01
            if frame==max_frames:
                done=True
                count+=1
        if multi:
            
            if not env.snake.lost and env.snake2.lost:
                won+=1
            elif env.snake.lost and env.snake2.lost:
                draw+=1
#    print("Maximum Frame reached: {}, Won:{}, Draw:{}".format(count,won,draw))   
    print("Average reward in this Generation: {:.2f}".format(sum(rews)/n))
    return rews

def video():
    for i in range(len(gen_video)):
        print("Generation ", gen_video[i]+1)
        playgame(env,w1_video[i],w2_video[i],b1_video[i],b2_video[i])
#video()

"Game Options"
width=12
height=10
vel=0
snake_size=20
simple=False
multi=False
opponent_is_Bot=True
player1_is_Bot=False
food_pos=np.random.randint(0,(width*snake_size-snake_size)//snake_size,size=(2,200))*snake_size
env=snake_env(multiplayer=multi,simple_observation=simple,\
          win_width=snake_size*width,win_height=snake_size*height,special_food_mode=False,\
          periodic_boundaries_mode=False,no_tail_mode=False,POV_mode=True,starving_mode=True,\
          starving_limit=100,starving_only_1=True,zerosum=False,tron_mode=True,\
          collision_with_head=True,give_food_pos=False,food_pos=food_pos,\
          starting_tail=2,starting_tail2=2,allow_180=False)
playgame(env,w1_1,w2_1,b1_1,b2_1)
"Neuro Evolution Options"
n=5000#population
num_gen=8
max_frames=2000
use_played=False

hidden_nodes=19
input_nodes=20
output_nodes=4
mutation_strength=0.2 #if you tweak
mutation_rate=0.1 #to replace
survive_reward=False
create_video=False

"Loading existing player"
#w1_2=np.load("surv100,best34,gen500,w1_best.npy")
#w2_2=np.load("surv100,best34,gen500,w2_best.npy")
#b1_2=np.load("surv100,best34,gen500,b1_best.npy")
#b2_2=np.load("surv100,best34,gen500,b2_best.npy")
"Loading existing population"
w1=np.load("gen150,nocross,01,02,180allow,w1.npy")
w2=np.load("gen150,w2.npy")
b1=np.load("gen150,b1.npy")
b2=np.load("gen150,b2.npy")
#playgame(env,w1_1,w2_1,b1_1,b2_1)
"Creating new population"
#w1,w2,b1,b2=generate(use_played)

if create_video:
    w1_video=[]
    w2_video=[]
    b1_video=[]
    b2_video=[]
    gen_video=[]
w1_best,w2_best,b1_best,b2_best=create_population(num_gen) 
max_rew_old=0
for k in range(num_gen):
    food_pos=np.random.randint(0,(width*snake_size-snake_size)//snake_size,size=(2,200))*snake_size        
    env=snake_env(multiplayer=multi,simple_observation=simple,\
          win_width=snake_size*width,win_height=snake_size*height,special_food_mode=False,\
          periodic_boundaries_mode=False,no_tail_mode=False,POV_mode=True,starving_mode=True,\
          starving_limit=100,starving_only_1=True,zerosum=False,tron_mode=True,\
          collision_with_head=True,give_food_pos=True,food_pos=food_pos,\
          starting_tail=10,starting_tail2=5,allow_180=True)
    rews=life(n,max_frames,multi=multi)
    w1_best[k],w2_best[k],b1_best[k],b2_best[k]=save_best(w1,w2,b1,b2,rews)
    selected= selection(rews)
    w1,w2,b1,b2=clone(w1,w2,b1,b2,selected)
    #w1,w2,b1,b2=crossover(w1,w2,b1,b2)
    w1,w2,b1,b2=mutate(w1,w2,b1,b2,n,mutation_strength,mutation_rate,tweak_only=True)
    w1,w2,b1,b2=keep_best_alive(w1,w2,b1,b2,w1_best[k],w2_best[k],b1_best[k],b2_best[k])
    print("Generation {}, Best {:.2f}".format(1+k,max(rews)))
    w1_1=w1_best[k]
    w2_1=w2_best[k]
    b1_1=b1_best[k]
    b2_1=b2_best[k]
    if create_video and max(rews)>max_rew_old: #for video of evolution
        w1_video.append(w1_1)
        w2_video.append(w2_1)
        b1_video.append(b1_1)
        b2_video.append(b2_1)
        gen_video.append(k)
    max_rew_old=max(rews)
    w1_2=w1_best[k]
    w2_2=w2_best[k]
    b1_2=b1_best[k]
    b2_2=b2_best[k]

"Snippets"
#playgame(env,w1_1,w2_1,b1_1,b2_1)

#np.save("w1_video3_5000,02,01,nocross,70gen",w1_video)
#np.save("w2_video3",w2_video)
#np.save("b1_video3",b1_video)
#np.save("b2_video3",b2_video)
#np.save("gen_video3",gen_video)
#np.save("food_video3",food_pos)

#rews_sel=check_selected(selected,rews)

#w1_1=w1_best[-1]
#w2_1=w2_best[-1]
#b1_1=b1_best[-1]
#b2_1=b2_best[-1]

#w1=np.load("surviving,n=5000,w1_1.npy")
#w2=np.load("surviving,n=5000,w2_1.npy")
#b1=np.load("surviving,n=5000,b1_1.npy")
#b2=np.load("surviving,n=5000,b2_1.npy")

#np.save("gen150,nocross,01,02,180allow,w1",w1)
#np.save("gen150,w2",w2)
#np.save("gen150,b1",b1)
#np.save("gen150,b2",b2)

#string_w1="gen500,n2000,sel3,str0.1,rate0.01,starving100,rews"+str(max(rews))+",w1"
#string_w2="gen500,n2000,sel3,str0.1,rate0.01,starving100,rews"+str(max(rews))+",w2"
#string_b1="gen500,n2000,sel3,str0.1,rate0.01,starving100,rews"+str(max(rews))+",b1"
#string_b2="gen500,n2000,sel3,str0.1,rate0.01,starving100,rews"+str(max(rews))+",b2"
#np.save(string_w1,w1)
#np.save(string_w2,w2)
#np.save(string_b1,b1)
#np.save(string_b2,b2)

#playgame(env)
   
#np.save("surv100,best34,gen500,w1_best",w1_best[0])
#np.save("surv100,best34,gen500,w2_best",w2_best[0])
#np.save("surv100,best34,gen500,b1_best",b1_best[0])
#np.save("surv100,best34,gen500,b2_best",b2_best[0])
#
#w1=np.load("gen350,n=5000,rew=13,strength=0.2,rate=0.02,w1_1.npy")
#w2=np.load("gen350,w2_1.npy")
#b1=np.load("gen350,b1_1.npy")
#b2=np.load("gen350,b2_1.npy")
#w1_2=w1[0]
#w2_2=w2[0]
#b1_2=b1[0]
#b2_2=b2[0]


#np.save("multi1_w1",w1)
#np.save("multi1_w2",w2)
#np.save("multi1_b1",b1)
#np.save("multi1_b2",b2)

#b1=np.load("gen500,n2000,sel3,str0.1,rate0.01,starving100,rews35.0,b1.npy")
#b2=np.load("gen500,n2000,sel3,str0.1,rate0.01,starving100,rews35.0,b2.npy")
#w1=np.load("gen500,n2000,sel3,str0.1,rate0.01,starving100,rews35.0,w1.npy")
#w2=np.load("gen500,n2000,sel3,str0.1,rate0.01,starving100,rews35.0,w2.npy")

#w2=np.load("gen140,n=5000,rew=7,w2_2.npy")
#b1=np.load("gen140,n=5000,rew=7,b1_2.npy")
#b2=np.load("gen140,n=5000,rew=7,b2_2.npy")
