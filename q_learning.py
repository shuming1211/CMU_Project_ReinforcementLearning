from environment import MountainCar
import sys
import numpy as np
import random


mode = sys.argv[1]
weight_out = sys.argv[2]
returns_out = sys.argv[3]
episodes = int(sys.argv[4])
max_iterations = int(sys.argv[5])
epsilon = float(sys.argv[6])
gamma = float(sys.argv[7])
learning_rate = float(sys.argv[8])

env = MountainCar(mode=mode)

def FindAction(state, w, b):
    actionList = []    
    for i in range(len(w[0])):
        sum = 0
        for element in state:
            temp = state[element] * w[element][i]
            sum += temp
        q = sum + b
        actionList.append(q)
    return actionList.index(max(actionList))

def FindMaxQ(state, w, b):
    actionList = []    
    for i in range(len(w[0])):
        sum = 0
        for element in state:
            temp = state[element] * w[element][i]
            sum += temp
        q = sum + b
        actionList.append(q)    
    return max(actionList)
    
def epsilonGreedy(bestaction):
    if random.uniform(0,1) <= 1-epsilon:
        return bestaction
    else:
        return np.random.choice([0, 1, 2])
    
def TrainMountainCar(env):
    wr_return = open(returns_out,'w')
     
    if mode == 'raw':
        w = np.zeros((2, 3))
    else:
        w = np.zeros((2048, 3))
    b = 0
    
    for i in range(episodes):
        totalreward = 0
        state  = env.reset()
        for iteration in range(max_iterations):            
            bestaction = FindAction(state, w, b)
            bestaction = epsilonGreedy(bestaction)
            
            sum = 0
            for element in state:
                temp = state[element] * w[element][bestaction]
                sum += temp
            q = sum + b

            next_state, reward, done = env.step(bestaction)#best action?
            totalreward +=  reward
            maxQ = FindMaxQ(next_state, w, b)

            if mode == 'raw':
                gradient = np.zeros((2, 3)) 
            else:
                gradient = np.zeros((2048, 3)) 
            
            for element in next_state:
                gradient[element][bestaction] = next_state[element]

            w = w - learning_rate * (q - (reward + gamma * maxQ)) * gradient
            b = b - learning_rate * (q - (reward + gamma * maxQ))
            
            state = next_state
            if done:
                break 
            
        wr_return.write(str(totalreward))
        wr_return.write('\n')
    return w , b 
   
def writeWeight(w,b):
    wr_weight = open(weight_out,'w')
    wr_weight.write(str(b))
    wr_weight.write('\n')
    for row in w:
        for element in row:
            wr_weight.write(str(element))
            wr_weight.write('\n')
    return 
     
def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv)
    w, b = TrainMountainCar(env)
    writeWeight(w,b)
