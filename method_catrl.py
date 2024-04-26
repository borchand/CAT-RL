import os
import numpy as np
import random 
import time
import numpy as np

from src.log import Log_experiments
from src.abstraction import Abstraction
from src.results import Results 
import src.hyper_param as hyper_param
from src.envs.taxiworld import *
from src.envs.officeworld import *
from src.envs.waterworld import *
from src.envs.wumpusworld import *
from src.envs.mountaincar import *
from src.learning import *

def evaluate(
        i, 
        agent,
        test_abs_every,
        step_max,
        env,
        abstract):
    if (i % test_abs_every == 0 and i>0):
        total_reward_list = []
        for j in range (10):
            state = env.reset()
            state_abs = abstract.state(state)
            agent.update_qtable (state_abs)
            new_action = agent.policy (state_abs)
            done = False
            reward = 0
            epoch = 0
            while (not done) and (epoch < step_max):
                action = new_action
                new_state_abs = state_abs
                r = 0
                while new_state_abs == state_abs:
                    new_state, temp_r, done, success = env.step (action)
                    new_state_abs = abstract.state (new_state)
                    agent.update_qtable (new_state_abs)
                    new_action = agent.policy_fixed(new_state_abs)
                    r += temp_r
                    epoch += 1
                    if state == new_state or done:
                        break
                    state = new_state
                agent.update_qtable (new_state_abs)
                new_action = agent.policy(new_state_abs)
                state = new_state
                state_abs = new_state_abs
                reward += r         
            total_reward_list.append(reward)
        agent._acc_reward_data["Num_episodes"].append(i)
        agent._acc_reward_data["Cumulative_rewards"].append(total_reward_list)
    return agent


def train_CAT_RL(
        map_name: str,
        step_max: int,
        episodes: int,
        env: object,
        boot: str,
        gamma: float,
        alpha: float,
        epsilon_min: float,
        decay: float,
        k_cap: int,
        epsilon: float = 1,
        seed: int = None,
        approach_name: str = "adrl",
):
    """
    Train the agent using the CAT RL algorithm
    :param map_name: the name of the map
    :param step_max: the maximum number of steps
    :param episodes: the number of episodes
    :param env: the environment
    :param boot: the bootstrapping method (from_concrete, from_ancestor or from_init)
    :param gamma: the discount factor
    :param alpha: the learning rate
    :param epsilon_min: the minimum value of epsilon
    :param decay: the decay rate of epsilon
    :param k_cap: the maximum number of abstractions
    :param epsilon: the initial value of epsilon
    :param seed: the seed for the random number generator. If None, the number of seeds is randomly generated.
    :param approach_name: the name of the approach

    """
    basepath = os.getcwd()
    heatmaps_directory = basepath + "/heatmaps/"
    if not os.path.exists(heatmaps_directory):
        os.makedirs(heatmaps_directory)
    env_alg_to_time = dict()
    abstraction_colors = dict()
    best_actions = dict()

    log_data = {
        "episode": [],
        "reward": [],
        "epochs": [],
        "epsilon": [],
        "abs_size": [],
        "rate": [],
        "success": []
    }

    log_info = {
        "seed": [],
        "time": [],
        "agent": [],
        "env": [],
        "episodes": [],
        "epochs": [],
        "success_rate": [],
        "abs_size": []
    }

    if seed is None:
        seed = random.sample(range(1, 1000), 1)[0]

    start_time = time.time()

    for trial in range(1):
        # ____________ main Parameters ___________________________
        random.seed(seed)
        approach_name = 'adrl'
        file_name = map_name + "_" + approach_name + "_" + str(trial)

        # env.seed(seed)
        #_________________________________________________________
        start_time = time.time()
        succ = []
        agent_abs_q = qlearning_abs(
            epsilon=epsilon,
            gamma=gamma,
            alpha=alpha,
            action_size = env._action_size,
            eps_min=epsilon_min,
            decay=decay)

        agent = agent_abs_q
        agent_con = qlearning (
            env, 
            epsilon=epsilon,
            gamma=gamma,
            alpha=alpha,
            state_size = env._state_size, 
            action_size = env._action_size,
            eps_min=epsilon_min,
            decay=decay)
        # agent_con = None
        abstract = Abstraction(env = env, n_state_variables = env._n_state_variables, 
                            state_variable_ranges = env._state_ranges, n_action_size = env._action_size, 
                            agent_con = agent_con, agent = agent, k_cap=k_cap, boot_type = boot)

        agent._abstract = abstract

        test_abs_every = 10
        eval_episodes = 100
        do_abs_every = 100
        log = Log_experiments(lp =do_abs_every, ep = eval_episodes)
        agent._acc_reward_data["Num_episodes"] = list()
        agent._acc_reward_data["Cumulative_rewards"] = list()
        i = 0

        while (i  <= episodes):
            i += 1
            state = env.reset()
            state_abs = abstract.state(state)
            agent.update_qtable (state_abs)
            done = False
            reward = 0
            epoch = 0
            while (not done) and (epoch < step_max):
                action = agent.policy (state_abs)
                new_state_abs = state_abs
                r = 0
                while new_state_abs == state_abs:
                    new_state, temp_r, done, success = env.step (action)
                    new_state_abs = abstract.state (new_state)
                    agent.update_qtable (new_state_abs)
                    if boot == 'from_concrete': new_action = agent_con.policy(new_state)
                    if boot == 'from_concrete': agent_con.train (state, new_state, action, new_action, temp_r)
                    r += temp_r
                    epoch += 1
                    if state == new_state or done:
                        break
                    state = new_state
                    
                new_action = agent.policy(new_state_abs)
                agent.update_qtable (new_state_abs)
                agent.train (state_abs, new_state_abs, action, new_action, r)
                state_abs = new_state_abs
                reward += r

            agent.decay()
            log.log_episode(reward, success, epoch)
            recent_success = log.recent_success_rate(do_abs_every)
            succ.append(recent_success)
            print ("_______________________________")
            print ("episode: " + str(i) + '\t' + "reward: " + str (reward) + '\t' + "epochs: " + str(epoch) 
                    + '\t' + "epsilon: " + str(round(agent._epsilon,3)) + '\t' +   "abs size: " + str(abstract._n_abstract_states) 
                    + '\t' + "rate: " + str (round (recent_success,2)) + '\t' + "success: " + str(success))

            log_data["episode"].append(i)
            log_data["reward"].append(reward)
            log_data["epochs"].append(epoch)
            log_data["epsilon"].append(agent._epsilon)
            log_data["abs_size"].append(abstract._n_abstract_states)
            log_data["rate"].append(recent_success)
            log_data["success"].append(success)
    #_______________________________________________________________________________________
            agent = evaluate(i, agent, test_abs_every, step_max, env, abstract)
    #______________________________________________________________________________________
            if i % do_abs_every == 0  and recent_success < 0.8:
                agent.intialize_eval()
                batch_con = []
                batch_abs = []
                for j in range (eval_episodes):
                    i +=1
                    state = env.reset()
                    state_abs = abstract.state(state)
                    agent.update_qtable (state_abs)
                    agent.update_eval (state_abs)
                    new_action = agent.policy (state_abs)
                    done = False
                    reward = 0
                    epoch = 0
                    while (not done) and (epoch < step_max):
                        action = new_action
                        new_state_abs = state_abs
                        r = 0
                        rr = 0
                        while new_state_abs == state_abs:
                            new_state, temp_r, done, success = env.step (action) 
                            new_state_abs = abstract.state (new_state)
                            agent.update_qtable (new_state_abs)
                            if boot == 'from_concrete': new_action = agent_con.policy(new_state)
                            if boot == 'from_concrete': batch_con.append([state, new_state, action, new_action, temp_r])
                            new_action = agent.policy(new_state_abs)
                            r += temp_r
                            rr += temp_r * agent._gamma
                            epoch += 1
                            if state == new_state or done:
                                break
                            state = new_state
                        agent.update_qtable (new_state_abs)
                        new_action = agent.policy(new_state_abs)
                        batch_abs.append([state_abs, new_state_abs, action, new_action, r])
                        agent.update_eval (new_state_abs)
                        agent.log_values (state_abs, new_state_abs, action, r)
                        state = new_state
                        state_abs = new_state_abs
                        reward += r
                                        
                    log.log_episode(reward, success, epoch)
                    agent = evaluate(i, agent, test_abs_every, step_max, env, abstract)


                agent.batch_train(batch_abs)
                if boot == 'from_concrete': agent_con.batch_train(batch_con)
                abstract.update_abstraction (agent._eval)

        end_time = time.time()

        env_alg_to_time[file_name] = float(round((time.time() - start_time),2))
        print("Time taken for "+str(file_name)+"(s):"+str(env_alg_to_time[file_name]))

        print("Time mean:",np.mean(list(env_alg_to_time.values())))
        print("Time std:",np.std(list(env_alg_to_time.values())))


        log_info["seed"].append(seed)
        log_info["time"].append(end_time - start_time)
        log_info["agent"].append("CAT-RL")
        log_info["env"].append(map_name)
        log_info["episodes"].append(episodes)
        log_info["epochs"].append(sum(log_data["epochs"]))
        log_info["success_rate"].append(sum(log_data["success"]) / len(log_data["success"]))
        log_info["abs_size"].append(log_data["abs_size"][-1])

        return agent, abstract, log_data, log_info

def main():
    train_CAT_RL(
        hyper_param.map_name,
        hyper_param.step_max,
        hyper_param.episode_max,
        hyper_param.env,
        hyper_param.bootstrap,
        hyper_param.gamma,
        hyper_param.alpha,
        hyper_param.epsilon_min,
        hyper_param.decay
    )

if __name__ == "__main__":
    main()