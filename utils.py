import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn


def fancy_visual(value_func,policy_int):
    """
    Credits: Desik Rengarajan and Srinivas Shakkottai, Texas A&M University
    """    
    grid = 4    
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    reshaped = np.reshape(value_func,(grid,grid))
    seaborn.heatmap(reshaped, cmap="icefire",vmax=1.1, robust = True,
                square=True, xticklabels=grid+1, yticklabels=grid+1,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt="f")
    counter = 0
    for j in range(0, 4):
        for i in range(0, 4):
            if int(policy_int[counter]) == 1:
                plt.text(i+0.5, j+0.7, u'\u2193', fontsize=12)
            elif int(policy_int[counter]) == 3:
                plt.text(i+0.5, j+0.7, u'\u2191', fontsize=12)
            elif int(policy_int[counter]) == 0:
                plt.text(i+0.5, j+0.7, u'\u2190', fontsize=12)
            else:
                plt.text(i+0.5, j+0.7, u'\u2192', fontsize=12)
            counter=counter+1

    plt.title('Heatmap of policy iteration with value function values and directions')
    print('Value Function',value_func)
    print('Policy',policy_int)
    plt.show()


def find_opt_pol(env, gamma, val_fun):
    """
    Function to compute optimal policy from value fun
    Input: env - gym environment, gamma - constant between 0 - 1, val_fun - value fun, pol_fun - policy fun
    Returns: pol_fun - optimal policy    
    """
    pol_fun = np.zeros(env.observation_space.n) # initialize optimal policy

    for s in range(env.observation_space.n): # for each state do
        v_new = np.zeros(env.action_space.n)
        for a in range(env.action_space.n): # for each action
            for P in env.P[s][a]:
                R = P[2] # reward after playing action 'a' from state 's'
                s_n = P[1] # next state after playing action 'a' from state 's'
                v_new[a] += P[0] * ( R + gamma * val_fun[s_n]) # expectation of value
        p_star = np.argmax(v_new)
        pol_fun[s] = p_star

    return pol_fun


def find_opt_q(env, gamma, val_fun):
    """
    Function to compute optimal Q table from value fun
    Input: env - gym environment, gamma - constant between 0 - 1, val_fun - value fun
    Returns: Q_tab - Q table    
    """
    Q_tab = np.zeros((env.observation_space.n, env.action_space.n)) # initalize Q table

    for s in range(env.observation_space.n): # for each state do
        for a in range(env.action_space.n): # for each action do
            q_val = 0
            for P in env.P[s][a]:
                R = P[2] # reward after playing action 'a' from state 's'
                s_n = P[1] # next state after playing action 'a' from state 's'
                q_val += P[0] * (R + gamma * val_fun[s_n]) # expected next state value
            Q_tab[s][a] = q_val # update q function

    return Q_tab


def value_iter(env, gamma, theta, val_fun = None):
    """
    Function to perform value iteration using the Bellman operator
    Input: env - gym environment, gamma - constant between 0 - 1, theta - convergence error, val_fun - initial value fun
    Returns: val_fun - optimal value function, val_hist - 2-norm of value error after each iteration    
    """
    if val_fun is None:
        val_fun = np.zeros(env.observation_space.n) # initialize value function with zeros
    val_hist = [] # store error trend

    while True: # iterate while error is larger than threshold
        max_error = 0 # cumulative max error over all states
        for s in range(env.observation_space.n): # for each state do
            v_k = val_fun[s]
            v_new = np.zeros(env.action_space.n)
            for a in range(env.action_space.n): # for each action
                for P in env.P[s][a]:
                    R = P[2] # reward after playing action 'a' from state 's'
                    s_n = P[1] # next state after playing action 'a' from state 's'
                    v_new[a] += P[0] * ( R + gamma * val_fun[s_n]) # expectation of value
            v_k2 = np.max(v_new) # output of Bellman operator
            val_fun[s] = v_k2 # update new value in value_func

            error = np.linalg.norm(v_k - v_k2) # error in value iteration
            if error > max_error:
                max_error = error # update maximum error over all states
        
        val_hist.append(max_error) # record error after iteration
        if max_error < theta:
            break # convergence reached

    return val_fun, val_hist


def policy_eval(env, gamma, theta, pol_fun, val_fun):
    """
    Function to perform value iteration under a given policy for policy iteration procedure
    Input: env - gym environment, gamma - constant between 0 - 1, theta - convergence error, pol_fun - policy function, val_fun - initial value fun
    Returns: val_fun - optimal value function 
    """
    while True: # iterate while error is larger than threshold
        #count += 1
        max_error = 0 # cumulative max error over all states
        for s in range(env.observation_space.n): # for each state do
            v_k = val_fun[s]
            v_k2 = 0
            a = pol_fun[s] # action under policy for state 's'
            for P in env.P[s][a]:
                R = P[2] # reward after playing action 'a' from state 's'
                s_n = P[1] # next state after playing action 'a' from state 's'
                v_k2 += P[0] * ( R + gamma * val_fun[s_n]) # expectation of value
            val_fun[s] = v_k2 # update new value in value_func
            error = np.linalg.norm(v_k - v_k2) # error in value iteration
            if error > max_error:
                max_error = error # update maximum error over all states
    
        if max_error < theta:
            break # convergence reached

    return val_fun


def pol_iter(env, gamma, theta, val_fun = None, pol_fun = None):
    """
    Function to perform policy iteration
    Input: env - gym environment, gamma - constant between 0 - 1, theta - convergence error, val_fun - initial value fun
    Returns: pol_fun - optimal policy function, val_fun - value fun, val_hist - 2-norm of value error after each policy update
    """
    if val_fun is None:
        val_fun = np.random.rand(env.observation_space.n)
    if pol_fun is None:
        pol_fun = np.random.randint(0, env.action_space.n, size=env.observation_space.n)
    val_hist = [] # store value error trend

    while True:
        # 1. policy evaluation
        v_k =  np.copy(val_fun)
        v_k2 = policy_eval(env, gamma, theta, pol_fun, val_fun) # iterative update of value function
        val_fun =  np.copy(v_k2)
        error = np.linalg.norm(v_k - v_k2) # error in value iteration
        val_hist.append(error)

        # 2. policy update
        p_k = np.copy(pol_fun)
        p_k2 = find_opt_pol(env, gamma, val_fun) # policy update function
        pol_fun =  np.copy(p_k2)
        if np.array_equal(p_k2, p_k): # optimal policy reached
            break

    return pol_fun, val_fun, val_hist


def q_learning(env, gamma, max_ep, max_step, eps_max, eps_min, eps_decay, alpha_max, alpha_min, alpha_decay, Q_star, Q_tab = None):
    """
    Function to perform Q learning
    Input: env - gym environment, gamma - constant between 0 - 1, max_ep - max no of episodes, max_step - max no of steps, 
        eps_max - max epsilon,  eps_min - min epsilon, eps_decay - epsilon decay rate,
        alpha_max - maximum learning rate, alpha_min - minimum learning rate, alpha_decay - alpha decay rate
        Q_star - optimal Q table for trends, Q_tab - initial Q table
    Returns: Q_tab - updated Q table, hist_q - error in Q history, hist_qstar - error in Q - Qstar history, hist_ret - return history
    """
    if Q_tab is None:
        Q_tab = np.zeros((env.observation_space.n, env.action_space.n)) # initialize Q table
    eps = eps_max # initila eps
    alpha = alpha_max # initial alpha
    hist_q = []
    hist_qstar = []
    hist_ret = []

    for i in range(max_ep): # iterate over maximum number of episodes
        Q_tab_ti = np.copy(Q_tab) # value of Q_tab at trajectory beginning
        s_k = env.reset() #Reset the env
        ret = 0 # initialize return
        for step in range(max_step):   

            # Epsilon - greedy
            if np.random.rand() < eps:
                a_k = env.action_space.sample() # random action
            else:
                a_k = np.argmax(Q_tab[s_k]) # argmax action
            s_k2, R, terminal, _ = env.step(a_k) # Take a step 
            
            # Q updation
            q_k = Q_tab[s_k][a_k]
            a_k2 = np.argmax(Q_tab[s_k2]) # maximizing future action
            q_k2 = Q_tab[s_k2][a_k2]
            q_new = q_k + alpha * (R + gamma * q_k2 - q_k) # updating Q
            Q_tab[s_k][a_k] = q_new # update new Q in table
            ret += math.pow(gamma, step) * R # compute return

            if terminal: # terminal state reached
                break
            s_k = s_k2 # update next step
        
        eps = eps_min + (eps_max - eps_min) * np.exp(-1 * eps_decay * i) # exponentially decay epsilon
        alpha = alpha_min + (alpha_max - alpha_min) * np.exp(-1 * alpha_decay * i) # exponentially decay epsilon

        # Metrics
        Q_tab_ti2 = np.copy(Q_tab) # value of Q_tab at trajectory end
        hist_q.append(np.linalg.norm(Q_tab_ti - Q_tab_ti2, 'fro')) # record q - qstar difference
        hist_qstar.append(np.linalg.norm(Q_star - Q_tab_ti2, 'fro')) # record qk2 - qk1 difference
        hist_ret.append(ret) # record return
    
    return Q_tab, hist_q, hist_qstar, hist_ret


def find_opt_vq(env, Q_tab):
    """
    Function to compute optimal value and optimal policy from Q table
    Input: env - gym environment, Q_tab - Q table
    Returns: val_fun - optimal value fun, pol_fun - optimal policy func
    """
    val_fun = np.zeros(env.observation_space.n) # initialize value function
    pol_fun = np.zeros(env.observation_space.n) # initalize policy function

    for s in range(env.observation_space.n): # for each state do
        val_fun[s] = np.max(Q_tab[s]) # maximum Q over all actions
        pol_fun[s] = np.argmax(Q_tab[s]) # action that leads to maximum Q value
    return val_fun, pol_fun


def sliding_avg(ip_list, win_len):
    """
    Function to compute sliding average over a list
    Input: ip_list - list to be averaged, win_len - averaging window length
    Returns: avg_list - averaged list
    """   
    avg_list = []
    for i in range(len(ip_list) - win_len):
        sum = 0
        for j in range(i, win_len+i): # iterate over input list
            sum += ip_list[j]
        sum /= win_len # compute average
        avg_list.append(sum)
    return avg_list