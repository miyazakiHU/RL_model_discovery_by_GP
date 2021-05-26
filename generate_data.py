import numpy as np
import csv

def q_learn_generate_data(params, Trial_num, r_prob):
    """
    This is a data generating function based on Standard Q-learning Model.
    Please pass two parameters as list.
    """
    
    alpha = params[0]
    beta = params[1]

    if not(0 <= alpha <= 1):
        raise Exception("alpha must be bigger than 0, and smaller than 1.")
    elif (beta < 0):
        raise Exception("beta must be positive.")
    elif (Trial_num < 0):
        raise Exception("Trial_num must be integer.")
    elif (r_prob < 0):
        raise Exception("r_prob must be bigger than 0, and smaller than 1.")
    
    act = [0]*Trial_num
    reward= [0]*Trial_num

    pA = np.zeros(Trial_num)
    Q = np.zeros((2,Trial_num))

    for t in range(Trial_num):
        pA[t] = 1/(1+np.exp(-beta * (Q[0,t]-Q[1,t])))

        # Case of you select one
        if (np.random.rand() < pA[t]):
            act[t] = 0
            if (np.random.rand() < r_prob):
                reward[t] = 1
            else:
                reward[t] = 0

        # Case of you select the other
        else:
            act[t] = 1
            if (np.random.rand() < (1-r_prob)):
                reward[t] = 1
            else:
                reward[t] = 0

        # update Q value
        if (t < (Trial_num-1)):
            Q[act[t],t+1] = Q[act[t],t] + alpha * (reward[t] - Q[act[t],t])

            # for unchosen option
            Q[1-act[t],t+1] = Q[1-act[t],t]

    act_str = [str(c)+"\n" for c in act]
    reward_str = [str(c)+"\n" for c in reward]

    # Output results
    with open("output\\act_q.txt", mode='w') as f:
        f.writelines(act_str)

    with open("output\\reward_q.txt", mode='w') as f:
        f.writelines(reward_str)

    true_parameters_list = [alpha, beta, Trial_num, r_prob]
    true_parameters = [str(c)+"\n" for c in true_parameters_list]
    with open("output\\true_parmeters.txt", mode='w') as f:
        f.writelines(true_parameters)


# Define parameters, Trial number, and reward rate
true_params = [0.5,0.3]
T = 5000
r_p = 0.7  # reward rate of good choice

q_learn_generate_data(params=true_params, Trial_num=T, r_prob=r_p)
