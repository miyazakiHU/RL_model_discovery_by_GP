def get_act_reward():
    with open("output\\act_q.txt", mode='r') as f:
        act_str = f.readlines()

    with open("output\\reward_q.txt", mode='r') as f:
        reward_str = f.readlines()

    act = [int(c.replace('\n','')) for c in act_str]
    reward = [int(c.replace('\n','')) for c in reward_str]
    
    # print("act:", act)
    # print("reward:", reward)

    return act, reward

# get_act_reward()