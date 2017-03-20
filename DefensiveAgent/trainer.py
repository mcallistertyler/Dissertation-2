#!/usr/bin/env python

#DQN Algorithm
#1. Initialise replay memory experienceReplay
#2. Initialise action value function Q
#3. Observe initial state s
#4. Repeat:
#5.        Select an action a
#6.        With probability epsilon select a random action
#7.        Otherwise select a = argmaxa'Q(s,a')
#8. Carry out action a
#9. Observe reward r and new state s'
#10.Store experience <s, a , r ,s'> in replay memory experienceReplay
#11.Sample random transitions from replay memory experienceReplay
#12.Calculate target for each minibatchSample transition
#13.       If ss' is terminal state then tt=rr
#14.       Otherwise tt=rr + GAMMA * maxa'Q(ss', aa')    
#15.Train the Q network using (tt = Q(ss, aa))^2
#16.s = s'
import tensorflow as tf
import numpy as np
import initNetwork
import sys
import subprocess
import pong as pong 
import random
from collections import deque
import cv2
import pickle

actions = 3 #Number of actions to choose from
futureReward = 0.99 # decay rate of past observations
observe = 5000. # timesteps to observe before training
explore = 5000. # frames over which to anneal epsilon
alpha = 0.0025 #Learning rate
X_AXIS,Y_AXIS = (80,80) #Initial downsample size
WRITER = True
totalReward = []
totalQmax = []

def averagePlot(time, step, reward, qmax):
    global totalReward, totalQmax
    #print("Reward.." + str(reward) + " \n")
    totalReward.append(reward)
    totalQmax.append(qmax)
    if time % step == 0:
        plot_file = open("plotfile.txt", "a")
        print("Writing to file")
        plot_file.write("Timesteps " + str(time) + "\n")
        plot_file.write("Reward " + str(np.mean(totalReward)) + "\n")
        plot_file.write("QMax " + str(np.mean(totalQmax)) + "\n")
        plot_file.write("\n")
        plot_file.close()
        totalReward = []
        totalQmax = []

def saveState(time, step, saver, sess, reward, qmax):
    #Save network weights to file
    if time % step == 0:
        saver.save(sess, 'savedNetworks/convnets', global_step = time)
        
def loadState(saver, sess):
    #global INITIAL_EPSILON
    #Load network weights loaded from file
    checkpoint = tf.train.get_checkpoint_state("savedNetworks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)

        #Don't initiate greedy policy if we have loaded a network
        #INITIAL_EPSILON = 0.1
        print("Loaded network:", checkpoint.model_checkpoint_path)
    else:
        print("No network found at location")

def trainNetwork(inputLayer, output, h_fc4, sess):
    # define the cost function
    global WRITER
    action = tf.placeholder("float", [None, actions]) #action
    target = tf.placeholder("float", [None])          #target

    #reduce_sum sums all of the elements of a vector
    output_action = tf.reduce_sum(tf.mul(output, action), reduction_indices = 1)
    #Cost is loss(?)
    cost = tf.reduce_mean(tf.square(target - output_action))
    #Train_step minimizes the cost function
    #It computes partial derivatives of the loss function
    #relatively to all of the weights and biases.
    #The adam optimizer is used to update the weights and biases.
    train_step = tf.train.RMSPropOptimizer(alpha).minimize(cost)
    #Create a pong state
    game_state = pong.GameState()
    #Algorithm step 1 - Initialize replay memory - experienceReplay.
    #A double ended queue which allows you to append
    #and delete elements from either side of the list
    experienceReplay = deque()
    #Algorithm step 2 - Initialise action value function Q with random weights
    #Get the first state by doing nothing and preprocess the image to 80x80x4
    #We perform no action in the first step and capture the initial game image
    do_nothing = np.zeros(actions)
    do_nothing[0] = 1
    image_data, noReward, terminal = game_state.frame_step(do_nothing)
    #Resize the image to 80x80 pixels and convert it to greyscale
    #Set the image threshold to 255 in case there are any hidden or unclear items on screen
    image_data = cv2.cvtColor(cv2.resize(image_data, (X_AXIS, Y_AXIS)), cv2.COLOR_BGR2GRAY)
    #Remove parts of the image that contain the player/basic/perfect paddle
    for x in range(0, 79):
        image_data[79,x] = 0
    for x in range(0, 79):
        image_data[78, x] = 0
    for x in range(0, 79):
        image_data[77, x] = 0
    z, image_data = cv2.threshold(image_data,1,255,cv2.THRESH_BINARY)
    #Stack 4 frames together to feed into the convolutional neural network
    #Algorithm step 3 - Observe initial state s
    state_t = np.stack((image_data, image_data, image_data, image_data), axis = 2) #Sequences of actions and observations

    #Initialise saving of networks
    saver = tf.train.Saver()
    #Run tensorflow and initialize all variables
    sess.run(tf.global_variables_initializer())
    #Load a saved state, if one exists
    loadState(saver, sess)
    #Set epsilon, used for greedy policy
    epsilon = 1.0

    timesteps = 0
    #Algorithm step 4
    while True:
        #Greedy Policy
        #qVals contains the list of actions produced
        #when the network is evaluated.
        #feed_dict = {previous states : [current states]}
        qVals = output.eval(feed_dict = {inputLayer : [state_t]})[0]
        currentAction = np.zeros([actions])
        keyPress = 0
        #If random integer between 0 and 1 is less than epsilon
        #or the number of timesteps is less than or equal to the observation
        #strategy.
        #Algorithm step 6 - With probability epsilon select a random action
        if random.random() <= epsilon or timesteps <= observe:
            #Choose an action at random
            keyPress = random.randrange(actions)
            #Perform the action
            currentAction[keyPress] = 1
        else:
            #Algorithm step 7 - Select the action with max q value
            #over all other actions.
            keyPress = np.argmax(qVals)
            #Algorithm step 8 - Perform the action
            currentAction[keyPress] = 1

        #Scale down epsilon to reduce the amount of times we
        #choose a random action as timesteps improve and we reach
        #a specified maximum value.
        if epsilon > 0.1 and timesteps > observe:
            epsilon -= (1.0 - 0.1) / explore

        #Select an action on every 4th frame
        for i in range(0, 4):
            #Algorithm step 9 - Run the selected action and observe next state and reward
            image_data, reward, terminal = game_state.frame_step(currentAction)
            image_data = cv2.cvtColor(cv2.resize(image_data, (X_AXIS, Y_AXIS)), cv2.COLOR_BGR2GRAY)
            for x in range(0, 79):
                image_data[79,x] = 0
            for x in range(0, 79):
                image_data[78, x] = 0
            for x in range(0, 79):
                image_data[77, x] = 0
            z, image_data = cv2.threshold(image_data,1,255,cv2.THRESH_BINARY)
            image_data = np.reshape(image_data, (X_AXIS, Y_AXIS, 1))##
            state_t1 = np.append(image_data, state_t[:,:,0:3], axis = 2)##


            #Algorithm step 10 - Store the current transition in memory
            #If the length of stored transitions is greater than our
            #memory then pop the leftmost value - the oldest transition.
            #print("What is reward ", reward)
            experienceReplay.append((state_t, currentAction, reward, state_t1, terminal))
            if len(experienceReplay) > 270000:
                experienceReplay.popleft()
        #Once the number of iterations is more than observe
        #Begin learning
        if timesteps > observe:
            #Algorithm Step 11 - Get 32 random minibatches to train on
            minibatchSample = random.sample(experienceReplay, 32)

            # get the batch variables
            previousStates = [d[0] for d in minibatchSample]   #State_t
            actionSet = [d[1] for d in minibatchSample]     #Action_t
            rewardSet = [d[2] for d in minibatchSample]     #Reward_t
            currentStates = [d[3] for d in minibatchSample]  #State_t1

            expectedReward = []
            rewardPerAction = output.eval(feed_dict = {inputLayer : currentStates})
            for i in range(0, len(minibatchSample)):
                #minibatchSample[i][4] = terminal
                #If the game has ended
                if minibatchSample[i][4]:
                    #Algorithm Step 13 - If ss' is terminal state then tt = rr 
                    #Append future reward
                    expectedReward.append(rewardSet[i])
                else:
                    #Algorithm Step 14 - Otherwise tt = rr + gamma * maxQ(s,a)
                    #Append discounted future reward
                    expectedReward.append(rewardSet[i] + GAMMA * np.max(rewardPerAction[i]))

            #Algorithm Step 15 - Train the network
            train_step.run(feed_dict = {
                target : expectedReward,
                action : actionSet,
                inputLayer : previousStates})

        #Algorithm Step 16 - Old state = new state
        state_t = state_t1
        timesteps += 1

        #Save progress every 10000 iterations
        saveState(timesteps, 10000, saver, sess, reward, np.max(qVals))
        averagePlot(timesteps, 10000, reward, np.max(qVals))
        if keyPress == 1:
            movement = "Up"
        elif keyPress == 2:
            movement = "Down"
        else:
            movement = "No movement"
        print ("Frame Total", timesteps, "/ Current Reward", reward, "/ Q Max %e" % np.max(qVals))
        #cv2.imwrite("logs_pong/frame" + str(timesteps) + ".png", image_data)
def main():
    x = initNetwork.convNet()
    sess = tf.InteractiveSession()
    inputLayer, output, h_fc4 = x.createNetwork(actions, X_AXIS, Y_AXIS)
    trainNetwork(inputLayer, output, h_fc4, sess)

if __name__ == "__main__":
    main()
