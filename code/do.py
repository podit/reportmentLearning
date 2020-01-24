import math
import numpy as np
from timeit import default_timer as timer

# Function to run the various elements of training and testing of a Q-table
def do(q, runs, episodes, resolution, dataPoints, profileFlag, eDecayFlag,
        gamma, alpha, epsilon, decay, epsilonDecay, eDecayStart, eDecayEnd,
        eDecayRate, eDecayExp, aDecayFlag, gDecayFlag, penalty, exponent, length,
        renderFlag):

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
        
        # Start split timer for each run
        start_split = timer()

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
                # Check for exponential epsilon decay and calculate from episode
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
        avg_rwd, std_rwd = q.test_qtable() 
     
        # Record aggregate values over total run length
        aggr_rewards[r] = avg_rwd
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

    # Return aggregate statistics over total length of runs
    return aggr_rewards, aggr_stds, aggr_ts_r, aggr_ts_r_min, aggr_ts_r_max,\
            aggr_ts_r_uq, aggr_ts_r_lq


