import numpy as np
 
def q_learn(params, act, reward):
    """
    This is a Standard Q-learning Function.
    Please pass params, act, and reward as list.
    """
    if (len(act) != len(reward)):
        raise Exception("Error1: act and reward length are mismatch.")
    alpha = params[0]
    beta = params[1]
    Trial_num = len(act)

    pA = np.zeros(Trial_num)
    Q = np.zeros((2,Trial_num))
    loglik = 0

    for t in range(Trial_num-1):
        pA[t] = 1/(1+np.exp(-beta * (Q[0,t]-Q[1,t])))
        pA[t] = max(min(pA[t], 0.9999), 0.0001)

        loglik += (act[t]==0) * np.log(pA[t]) + (act[t]==1) * np.log(1-pA[t])

        # update Q value
        if (t < Trial_num):
            Q[act[t],t+1] = Q[act[t],t] + alpha * (reward[t] - Q[act[t],t] ) 
      
            # for unchosen option
            Q[1-act[t],t+1] = Q[1-act[t],t]

    return -loglik



def eval_func(function, beta, act, reward):
    """
    This function returns AIC according to behavioral data(act, reward).
    Please pass formula f(Q_t, t).
    """
    if (len(act) != len(reward)):
        raise Exception("Error1: act and reward length are mismatch.")
    Trial_num = len(act)

    pA = np.zeros(Trial_num)
    Q = np.zeros((2,Trial_num))
    loglik = 0

    for t in range(Trial_num-1):
        pA[t] = 1/(1+np.exp(-beta * (Q[0,t]-Q[1,t])))
        pA[t] = max(min(pA[t], 0.9999), 0.0001)

        loglik += (act[t]==0) * np.log(pA[t]) + (act[t]==1) * np.log(1-pA[t])

        # update Q value
        if (t < Trial_num):
            # if (Q[act[t],t]+reward[t] == 0):
            #     print("t:", t)
            Q[act[t],t+1] = function(Q[act[t],t], reward[t])
            # Q[act[t],t+1] = Q[act[t],t] + alpha * ( reward[t] - Q[act[t],t] ) 
      
            # for unchosen option
            """
            実際はこちらは別の更新式を指定する必要があるが，遺伝子が2つ必要になるため，とりあえず更新しないようにする
            """
            Q[1-act[t],t+1] = Q[1-act[t],t]

    return -loglik
