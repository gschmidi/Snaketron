# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:55:17 2019

@author: Gerald Schmidhofer
"""
import statistics
import numpy as np
import tensorflow as tf
from tensorflow import keras
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


def reflexAgent(obs,player,obs_simple=False):#input observation
    #obs[0]=snake.x,obs[1]=snake.y, obs[4]=food.x, obs[5]=food.y
    if obs_simple:
        action=2
        if obs[0]==obs[2]:
            action=1
    else:
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

def playgame(env):
    opponent_is_Bot=False
    player1_is_Bot=True
    delay_last_frame=1000
    time_delay=120 
    
    pygame.init() 
    pygame.display.set_caption("Snaketron")
    win=pygame.display.set_mode((env.win_width,env.win_height))
    pygame.time.delay(500)
    
    observation=env.setup()
    observation=format_obs(observation,simple=simple)
    done=False
    confused=False
    action=np.random.randint(2)+1# 0, 1, 2 or 3 corresponding to left, up, right, down
    action2=np.random.randint(2)+1# 0, 1, 2 or 3 corresponding to left, up, right, down
    while not done:    
        pygame.time.delay(time_delay-vel*10)
        "-----Input------" 
        keys =pygame.key.get_pressed()                
        if player1_is_Bot:
            #action=np.random.randint(0,4)
#            action=reflexAgent(observation,1,obs_simple=True)
            q = model.predict( np.array([observation,])) #normalize obs
            if np.max(q)<0.6:
                confused=True
            r=np.random.rand()
            if r < q[0,0]:
                action=0
            elif r < q[0,0]+q[0,1]:
                action=1
            elif r < q[0,0]+q[0,1]+q[0,2]:
                action=2
            elif r < q[0,0]+q[0,1]+q[0,2]+q[0,3]:
                action=3
        else:
            action=in_player(keys,action)
        if env.multiplayer and not opponent_is_Bot:
            action2=in_player_WASD(keys,action2)
        elif env.multiplayer and opponent_is_Bot:
            action2=reflexAgent(observation,2)
        if not env.multiplayer:   
            [observation,reward,done]=env.step(action)
            observation=format_obs(observation,simple=simple)
        else:
            [observation,reward,reward2,done]=env.step(action,action2)
            observation=format_obs(observation,simple=simple)
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


def random_games(env, model=False):
    "Inicialization"
    memory_obs=[]
    memory_ac=[]
    observations=[]
    actions=[]
    done=False
    reward=0
    action=np.random.randint(2)+1# 0, 1, 2 or 3 corresponding to left, up, right, down
    total_rewards=[]
    for i in range(initial_games):
        observation=env.setup()
        observation=format_obs(observation,simple=simple)
        frame=0
        while not done:
            frame+=1
            observations.append(np.copy(observation)) #save old obs
            [observation, reward_temp, done]=env.step(action)
            observation=format_obs(observation,simple=simple)
            actions.append(action)#save old action
            reward+=reward_temp#+0.1 #0.1 for surviving
            if model==False:
                action=reflexAgent(observation,1,obs_simple=True)
#                action=np.random.randint(0,4)
            else:            
                q = model.predict(np.array([observation,]))
##                action=np.argmax(q)
                r=np.random.rand()
                if r < q[0,0]:
                    action=0
                elif r < q[0,0]+q[0,1]:
                    action=1
                elif r < q[0,0]+q[0,1]+q[0,2]:
                    action=2
                elif r < q[0,0]+q[0,1]+q[0,2]+q[0,3]:
                    action=3
            if frame==max_frames:
                break
        if done:
            del actions[-1]
            del observations[-1]
            reward+=5
        print("Reward:{}, Died in frame {}, Iteration {}, Games Saved {}".format(reward,frame,i,len(memory_ac)))
        if reward >= threshold:
            memory_obs.append(observations)
            memory_ac.append(actions)
        total_rewards.append(reward)
        observations=[]
        actions=[]
        reward=0
        done=False
        if len(memory_ac)>=memory_limit:
            print("Limit of {} recorded games reached".format(memory_limit))
            break
        
    print("Average Reward ",statistics.mean(total_rewards))
    print("{} games above threshold".format(len(memory_ac)))
    return memory_obs,memory_ac,statistics.mean(total_rewards)

def playgame_save(env):
    "Space to eliminate last 10 moves"
    opponent_is_Bot=False
    delay_last_frame=1000
    time_delay=120 
    
    memory_obs=[]
    memory_ac=[]
       
    pygame.init() 
    pygame.display.set_caption("Snaketron")
    win=pygame.display.set_mode((env.win_width,env.win_height))
    pygame.time.delay(500)
      
    observation=env.setup()
    
    done=False
    action=np.random.randint(2)+1# 0, 1, 2 or 3 corresponding to left, up, right, down
    action2=np.random.randint(3) #therefore never goes left in the beginning
    k=10 #how many times input check per frame
    game_done=False
    while not game_done:    
        for l in range(k):
            pygame.time.delay(int((time_delay-vel*10)/k))
            "-----Input------" 
            keys =pygame.key.get_pressed()     
            action=in_player(keys,action)
            if env.multiplayer and not opponent_is_Bot:
                action2=in_player_WASD(keys,action2)
            elif env.multiplayer and opponent_is_Bot:
                action2=reflexAgent(observation,2)
        if not env.multiplayer:
            memory_obs.append(np.copy(observation))
            [observation,reward,done]=env.step(action)
            memory_ac.append(action)
        else:
            memory_obs.append(observation)
            [observation,reward,reward2,done]=env.step(action,action2)
            memory_ac.append(action)
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
                game_done=True
        if keys[pygame.K_SPACE]:
            print("Last 20 Moves Deleted")
            del memory_ac[-20:]
            del memory_obs[-20:]
        if done:
            action=np.random.randint(3)
            observation=env.setup()
            
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
    
    return format_obs(np.array(memory_obs),simple),np.array(memory_ac)

"-----Supervised Learning----"
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(200,)),
#        keras.layers.Dense(16*6, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
#        keras.layers.Dense(64*16, activation=tf.nn.relu),
#        keras.layers.Dense(64*12, activation=tf.nn.relu),
#        keras.layers.Dense(16*6, activation=tf.nn.relu),
        keras.layers.Dense(4, activation=tf.nn.softmax)
        ])

    #sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9) #decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer="adam", 
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model

def get_training_data(model=False):
    observations,actions,av_rew=random_games(env,model=model)
    if observations==[]:
        print("Threshold never archieved")
        return observations, actions, av_rew
    else:
        x=np.array(observations[0])
        for i in range(len(observations)-1):
            x=np.append(x,np.array(observations[i+1]),axis=0)
        y=np.hstack(actions)
        #x=format_obs(x) #obs never zero and up to 1, already formated!
    return x,y,av_rew       

def format_obs(obs, simple=True):
    if simple:
        if np.ndim(obs)==1:
            obs[:4]=obs[:4]/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
            obs[4:6]=obs[4:6]/snake_size/2+0.5       
        elif np.ndim(obs)==2:
            obs[:,:4]=obs[:,:4]/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
            obs[:,4:6]=obs[:,4:6]/snake_size/2+0.5
        else:
            print("Wrong Dimesion of Observation: Cannot format")
    else:
        if np.ndim(obs)==1:
            temp=obs[2:4]/snake_size/2+0.5 #velocities
            temp2=obs[6]/(3*snake_size) #snake_lenght
            obs=obs/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
            obs[2:4]=temp
            obs[6]=temp2
        elif np.ndim(obs)==2:
            temp=obs[:,2:4]/snake_size/2+0.5 #velocities
            temp2=obs[:,6]/(3*snake_size) #snake_lenght
            obs=obs/(snake_size*max(width,height))+1/(max(width,height)) #position normalize
            obs[:,2:4]=temp
            obs[:,6]=temp2
    return obs

"Adjustables"
width=10
height=10
vel=7

snake_size=20
initial_games= 40000
max_frames=30
threshold=4
epochs=3
batch_size=100
memory_limit=10000
simple=False
env=snake_env(multiplayer=False,simple_observation=simple,\
              win_width=snake_size*width,win_height=snake_size*height,\
              periodic_boundaries_mode=True,no_tail_mode=False,starting_tail=1)


"Load Played Game"
obs=np.load("played10x10_Singleplayer_obs_2.npy")
ac=np.load("played10x10_Singleplaye_ac_2.npy")


"Train"
#obs,ac=playgame_save(env)
model=create_model()
model.fit(obs,ac,epochs=12)

playgame(env)

"Snippets"

#model.summary()
#obs1=format_obs(obs,simple=False)
#obs1,ac1,av_rew1=get_training_data(model=model)

"Combine 2 played games"
#obs_mix=np.concatenate((obs,obs2))
#ac_mix=np.concatenate((ac,ac2))

#for i in range(1):
#    [x,y,av_rew]=get_training_data()
#    print("len(x)",len(x))
#    model.fit(x,y, epochs=epochs, batch_size=batch_size)
#

"Get Training Data"
#initial_games=100
#threshold=-5
#[x2,y2,av_rew2]=get_training_data(model)

#print("Gain1",av_rew2-av_rew)

#obs_new=np.delete(obs,-1,0)
#ac_new=np.delete(ac,0)
#obs=np.load("10x10_100max_10th_obs_3rd.npy")
##obs=obs/180
#ac=np.load("10x10_100max_10th_ac_3rd.npy")

#np.save("frames30_th3_Vel_obs_1",obs1)
#np.save("frames30_th3_Vel_ac_1",ac1)
#
#np.save("played10x10_Singleplayer_obs_2.npy",obs)
#np.save("played10x10_Singleplaye_ac_2.npy",ac)
#x=x/200


#model.summary()
#test_loss, test_acc = model.evaluate(x, y2)
#print('Test accuracy:', test_acc)
#input_x = tf.placeholder(tf.float32, [None, 10], name='input_x') 
#first_layer_weights = model.layers[0].get_weights()[0]
#first_layer_biases  = model.layers[0].get_weights()[1]
#second_layer_weights = model.layers[1].get_weights()[0]
#second_layer_biases  = model.layers[1].get_weights()[1]
