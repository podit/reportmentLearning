# Import numpy for array managment and timeit to time execution
import math
import numpy as np
from timeit import default_timer as timer

import multiprocessing as mp

# Import plotting functions
import plotKew as plt
# Import single and double Q-Learning classes
from sinKew import SinKew
from dblKew import DblKew

# Incorporated control script for parallell execution
# Function to run the various elements of training and testing of a Q-table
def do(q, runs, episodes, bins, resolution, dataPoints, profileFlag, eDecayFlag,
        gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
        eDecayRate, eDecayExp, aDecayFlag, gDecayFlag, penalty, exponent,
        length, renderFlag, e, queue):

    # Create aggregate arrays to store values for run length
    aggr_rewards = np.zeros(runs)
    aggr_stds = np.zeros(runs)
    aggr_ts_r = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_min = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_max = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_uq = np.zeros((runs, int(dataPoints)))
    aggr_ts_r_lq = np.zeros((runs, int(dataPoints)))

    # Calculate decay esponent -TODO:change division to variable
    if eDecayFlag and eDecayExp: exp = 1 / (episodes / 5)
    
    # Iterate through each run
    for r in range(runs):
        # Create array to store rewards for bin length and iterate bins
        avg_rwd = np.zeros(bins)
        
        # Start split timer for each run
        start_split = timer()
            
        for b in range(bins):
            # Reset datapoints iterator for each run
            dp = 0

            # Create arrays to store timestep values for profiling training
            timestep_reward = np.zeros(int(dataPoints))
            timestep_reward_min = np.zeros(int(dataPoints))
            timestep_reward_max = np.zeros(int(dataPoints))
            timestep_reward_up_q = np.zeros(int(dataPoints))
            timestep_reward_low_q = np.zeros(int(dataPoints))

            # Reset environment and Q-tables
            q.init_env(resolution)
            
            # Reset decaying epsilon to starting value
            if eDecayFlag and not eDecayExp: epsilon = epsilonDecay

            # Iterate through each episode in a run
            for episode in range(episodes):
                #episode += 1
                
                # Check for epsilon decay flag
                if eDecayFlag:
                    # Check for linear epsilon decay
                    if not eDecayExp:
                        # Decay epsilon values during epsilon decay range
                        if eDecayEnd >= episode >= eDecayStart:
                            epsilon -= eDecayRate
                            # Prevent epsilon from going negative
                            if epsilon < 0:
                                epsilon = 0
                    # Check for exponential decay and calculate from episode
                    elif eDecayExp: epsilon = 0.5 * math.exp(-exp * episode)

                # Check alpha decay flag and set alpha according episode
                if aDecayFlag:
                    alpha = math.exp(-exp * episode)
                # Also for gamma -TODO:replace hardcoded intersect values
                if gDecayFlag:
                    gamma = 1 + -math.exp(-exp * episode)
                
                # Perform learning for each episode
                q.lrn(epsilon, episode, penalty, exponent, length, alpha, gamma)

                # Record descriptive statistics at each resolution step
                if episode % resolution == 0:
                    timestep_reward[dp] = np.average(
                            q.timestep_reward_res)
                    timestep_reward_min[dp] = np.min(
                            q.timestep_reward_res)
                    timestep_reward_max[dp] = np.max(
                            q.timestep_reward_res)
                    timestep_reward_up_q[dp] = np.percentile(
                            q.timestep_reward_res, 75)
                    timestep_reward_low_q[dp] = np.percentile(
                            q.timestep_reward_res, 25)
                    dp += 1
            
            # Check if testing is to be rendered and if so wait for user input
            if renderFlag: input('Start testing (rendered)')
            # Perform testing on trained Q table after episodes are completed
            avg_rwd[b], std_rwd = q.test_qtable() 
     
        # Record aggregate values over total run length
        aggr_rewards[r] = np.mean(avg_rwd)
        aggr_stds[r] = std_rwd
        aggr_ts_r[r] = timestep_reward
        aggr_ts_r_min[r] = timestep_reward_min
        aggr_ts_r_max[r] = timestep_reward_max
        aggr_ts_r_uq[r] = timestep_reward_up_q
        aggr_ts_r_lq[r] = timestep_reward_low_q
        
        # Check is profiling flag is set
        if profileFlag:
            # Calculate split (total runs) time and report profiling values
            end_split = timer()
            segment = end_split - start_split
            print('Run:', r)
            print(f'Average reward:{avg_rwd}, std:{std_rwd}')
            print('Split time:', segment)
            print('#--------========--------#')

    # Record rewards into multithreaded queue
    queue[e].put(aggr_rewards)

    return True

# Set initialisation policy for Q-table
initialisation = 'uniform'      # uniform, ones, zeros

# Set on-policy (sarsa) or off-policy (q_lrn) control method for training
policy = 'sarsa'                # q_lrn, sarsa

# Control flags for double Q-learning, epsilon decay and expontntial penalties
doubleFlag = True
# Epsilon decay linearly
eDecayFlag = True
# Exponential epsilon decay flag, dependent on epsilon decay flag
eDecayExp = False

# Alpha decay flag
aDecayFlag = False

# Set gamma decay flag for episodic OR decay for number of encounters with each
#   state action pair
gDecayFlag = False
gDecayEncounter = False
if gDecayFlag and gDecayEncounter: print('Episodic decay will not be followed')

# Flags to report each run and each resolution step
# Report run number and average reward from test for each run
profileFlag = False
# Report episode and epsilon value as well as reward for each resolution step
verboseFlag = False

# Render flags for testing and training
renderTest = False
# Render training at each resolution step
renderTrain = False

# Set openai gym environment (CartPole and MountainCar have been tested)
environment = 'CartPole-v1'     # CartPole-v1, MountainCar-v0
#environment = 'MountainCar-v0'

# Flags for continuous observation and action spaces
contOS = True
contAS = False
# Discretisation factor to set number of bins for continuous
#   observation/action spaces depending on flags
discretisation = 6

# Set resolution for bins to record performance every <resolution> epsiodes
resolution = 5

# Set max steps in the environment
maxSteps = 500
# Set number of tests to be run (average is reported)
nTests = 100

# Set penalty to be applied at episode completion (positive acts as reward)
penalty = 0

# Used when logFlag is enabled
# Set exponent for exponential penalty and length of applied steps
exponent = -0.75
length = 5

# Set number of episodes and runs to be completed by the agent
episodes = 100
# Episodes constitute run length before testing
runs = 100

bins = 10

# Set hyper-parameters for use in bellman equation for updating Q table
# Discount factor
gamma = 0.995
# Learning rate
alpha = 0.5

# Set epsilon value for constant e-greedy method
epsilon = 0.1

# Used whrn eDecayFlag is enabled
# Set decay coefficient
decay = 1.5
# Set epsilon start value
epsilonDecay = 0.25

# Start experiment timer
start = timer()

# Number of experimental parameters
experiments = 4

ind = [1, 2, 3, 4]

# List of experimental parameters to be tested
val1 = ['q_lrn', 'sarsa', 'q_lrn', 'sarsa']
val2 = [False, False, True, True]
#val3 = [8, 8, 9, 9]
# List of values to be revorded and compared in boxplot
rwds = [None] * experiments
avgs = [None] * experiments

threads = []

queue = [mp.Queue()] * experiments

# Iterate through each experimental value and run Q-learning
for e in range(experiments):
    # Chenge value to the correponding hyper-parameter
    policy = val1[e]
    doubleFlag = val2[e]
    #discretisation = val3[e]

    # Calculate the decay period
    eDecayStart = 1
    eDecayEnd = episodes // decay
    # Calculate decay rate
    eDecayRate = epsilonDecay / eDecayEnd

    # Create number of individual data points for run length
    dataPoints = episodes / resolution

    # Initialise double or single QL class with the doubleFlag value provided
    if doubleFlag: q = DblKew(initialisation, policy, environment, contOS,
                contAS, discretisation, maxSteps, nTests, gDecayEncounter,
                verboseFlag, renderTest, renderTrain)
    else: q = SinKew(initialisation, policy, environment, contOS, contAS,
                discretisation, maxSteps, nTests, gDecayEncounter, verboseFlag,
                renderTest, renderTrain)

    # Run experiment passing relevent variables to do script to run QL,
    #   as a multithreading process
    process = mp.Process(target=do, args=(q, runs, episodes, bins, resolution,
            dataPoints, profileFlag, eDecayFlag, gamma, alpha, epsilon, decay,
            epsilonDecay, eDecayStart, eDecayEnd, eDecayRate, eDecayExp,
            aDecayFlag, gDecayFlag, penalty, exponent, length, renderTest, e,
            queue,))

    process.start()
    threads.append(process)
    
e = 0
for process in threads:
    rwds[e] = queue[e].get()
    process.join()
    e += 1

for e in range(experiments):
    avgs[e] = np.average(rwds[e])

# End timer and print time
end = timer()
print('Time:', end-start)
print('Environment:', environment)
# Wait for input to show plot
input('Show plots')

print(val1)
print(val2)
#print(val3)
print(avgs, ind) 

plt.boxPlot(rwds, avgs, ind)

