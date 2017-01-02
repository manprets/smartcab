import random
import os
import numpy as np
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Qdf = self.init_Q()
        self.past_state = 0
        self.past_reward = 0
        self.past_action = 'None'
        self.epsilon_iter = 1
        self.state_visit_hist=np.zeros(10,dtype=np.int)

    def init_Q(self):
        # Sr #  Car to my Left  Car Oncoming    Traffic Light   Next Waypoint   state num
        # 1     otherwise       otherwise       Red             F               s0
        # 2     otherwise       otherwise       Red             L               s1
        # 3     otherwise       otherwise       Red             R               s2
        # 4     otherwise       otherwise       Green           F               s3
        # 5     otherwise       otherwise       Green           L               s4
        # 6     otherwise       otherwise       Green           R               s5
        # 7     forward         otherwise       Red             F               s6
        # 8     forward         otherwise       Red             L               s7
        # 9     forward         otherwise       Red             R               s8
        # 10    forward         otherwise       Green           F               s9
        # 11    forward         otherwise       Green           L               s10
        # 12    forward         otherwise       Green           R               s11
        # 13    otherwise       forward         Red             F               s12
        # 14    otherwise       forward         Red             L               s13
        # 15    otherwise       forward         Red             R               s14
        # 16    otherwise       forward         Green           F               s15
        # 17    otherwise       forward         Green           L               s16
        # 18    otherwise       forward         Green           R               s17
        # 19    forward         forward         Red             F               s18
        # 20    forward         forward         Red             L               s19
        # 21    forward         forward         Red             R               s20
        # 22    forward         forward         Green           F               s21
        # 23    forward         forward         Green           L               s22
        # 24    forward         forward         Green           R               s23
        # 25    otherwise       right           Red             F               s24
        # 26    otherwise       right           Red             L               s25
        # 27    otherwise       right           Red             R               s26
        # 28    otherwise       right           Green           F               s27
        # 29    otherwise       right           Green           L               s28
        # 30    otherwise       right           Green           R               s29
        # 31    forward         right           Red             F               s30
        # 32    forward         right           Red             L               s31
        # 33    forward         right           Red             R               s32
        # 34    forward         right           Green           F               s33
        # 35    forward         right           Green           L               s34
        # 36    forward         right           Green           R               s35
        
        valid_left = ['forward', 'otherwise']
        valid_oncoming = ['forward', 'right', 'otherwise']
        valid_lights = ['red', 'green']
        valid_next_waypoint = ['forward', 'left', 'right']
        
        actions = ['forward', 'left', 'right', 'None']
        states = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9']
        
        num_of_states = len(states)#len(valid_left) * len(valid_oncoming) * len(valid_lights) * len(valid_next_waypoint)
        num_of_actions = len(actions)
        
        Q = np.empty((num_of_states, num_of_actions))
        Q[:] = 0#float("-inf")#np.NAN
        # Q = np.random.randn(num_of_states, num_of_actions)
        
        Qdf = pd.DataFrame(Q, columns=actions)
        
        return Qdf
        
    def get_curr_state(self, inputs, next_waypoint):
        # state = None
        light = inputs['light']
        oncoming = inputs['oncoming']
        left = inputs['left']
        
        if left == 'forward':
            valid_left_ = left
        else:
            valid_left_ = 'otherwise'
        if oncoming == 'forward' or oncoming == 'right':
            valid_oncoming_ = oncoming
        else:
            valid_oncoming_ = 'otherwise'
        
        # create keys to state dict
        keys_to_state_dict={}
        # keys_to_state_dict[('otherwise','otherwise','red','forward')] = 0
        # keys_to_state_dict[('otherwise','otherwise','red','left')] = 1
        # keys_to_state_dict[('otherwise','otherwise','red','right')] = 2
        # keys_to_state_dict[('otherwise','otherwise','green','forward')] = 3
        # keys_to_state_dict[('otherwise','otherwise','green','left')] = 4
        # keys_to_state_dict[('otherwise','otherwise','green','right')] = 5
        # keys_to_state_dict[('forward','otherwise','red','forward')] = 6
        # keys_to_state_dict[('forward','otherwise','red','left')] = 7
        # keys_to_state_dict[('forward','otherwise','red','right')] = 8
        # keys_to_state_dict[('forward','otherwise','green','forward')] = 9
        # keys_to_state_dict[('forward','otherwise','green','left')] = 10
        # keys_to_state_dict[('forward','otherwise','green','right')] = 11
        # keys_to_state_dict[('otherwise','forward','red','forward')] = 12
        # keys_to_state_dict[('otherwise','forward','red','left')] = 13
        # keys_to_state_dict[('otherwise','forward','red','right')] = 14
        # keys_to_state_dict[('otherwise','forward','green','forward')] = 15
        # keys_to_state_dict[('otherwise','forward','green','left')] = 16
        # keys_to_state_dict[('otherwise','forward','green','right')] = 17
        # keys_to_state_dict[('forward','forward','red','forward')] = 18
        # keys_to_state_dict[('forward','forward','red','left')] = 19
        # keys_to_state_dict[('forward','forward','red','right')] = 20
        # keys_to_state_dict[('forward','forward','green','forward')] = 21
        # keys_to_state_dict[('forward','forward','green','left')] = 22
        # keys_to_state_dict[('forward','forward','green','right')] = 23
        # keys_to_state_dict[('otherwise','right','red','forward')] = 24
        # keys_to_state_dict[('otherwise','right','red','left')] = 25
        # keys_to_state_dict[('otherwise','right','red','right')] = 26
        # keys_to_state_dict[('otherwise','right','green','forward')] = 27
        # keys_to_state_dict[('otherwise','right','green','left')] = 28
        # keys_to_state_dict[('otherwise','right','green','right')] = 29
        # keys_to_state_dict[('forward','right','red','forward')] = 30
        # keys_to_state_dict[('forward','right','red','left')] = 31
        # keys_to_state_dict[('forward','right','red','right')] = 32
        # keys_to_state_dict[('forward','right','green','forward')] = 33
        # keys_to_state_dict[('forward','right','green','left')] = 34
        # keys_to_state_dict[('forward','right','green','right')] = 35

        # Sr #	Yes or No	Direction	Yes or No	Direction	Traffic Light	Next Waypoint   State
        # 1	        No		                No		                Red	            F           s0
        # 2	        No		                No		                Red	            L           s1
        # 3	        No		                No		                Red	            R           s2
        # 4	        No		                No		                Green	        F           s3
        # 5	        No		                No		                Green	        L           s4
        # 6	        No		                No		                Green	        R           s5
        # 13	    Yes	        F	        No		                Red	            R           s6
        # 22	    Yes	        F	        No		                Green	        R           s7
        # 37	    No		                Yes	        F	        Green	        L           s8
        # 39	    No		                Yes	        R	        Green	        L           s9
        # keys_to_state_dict[('otherwise','otherwise','red','forward')] = 0
        # keys_to_state_dict[('otherwise','otherwise','red','left')] = 1
        # keys_to_state_dict[('otherwise','otherwise','red','right')] = 2
        # keys_to_state_dict[('otherwise','otherwise','green','forward')] = 3
        # keys_to_state_dict[('otherwise','otherwise','green','left')] = 4
        # keys_to_state_dict[('otherwise','otherwise','green','right')] = 5
        # keys_to_state_dict[('forward','otherwise','red','right')] = 6
        # keys_to_state_dict[('forward','otherwise','green','right')] = 7
        # keys_to_state_dict[('otherwise','forward','green','left')] = 8
        # keys_to_state_dict[('otherwise','right','green','left')] = 9
        
        # state = keys_to_state_dict[(valid_left_, valid_oncoming_, light, next_waypoint)]
        if next_waypoint=='forward':
            if light=='red':
                state = 0#'s0'
            elif light=='green':
                state = 3#'s3'
        elif next_waypoint=='left':
            if light == 'green':
                if oncoming == 'forward':
                    state = 8#'s8'
                elif oncoming == 'right':
                    state = 9#'s9'
                else:
                    state = 4#'s4'
            elif light == 'red':
                state = 1#'s1'
        elif next_waypoint=='right':
            if light == 'green':
                if left == 'forward':
                    state = 7#'s7'
                else:
                    state = 5#'s5'
            elif light == 'red':
                if left == 'forward':
                    state = 6#'s6'
                else:
                    state = 2#'s2'
        return state

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        self.state = self.get_curr_state(inputs, self.next_waypoint)
        print "state:", self.state
        #update state_visit_hist
        self.state_visit_hist[self.state] = self.state_visit_hist[self.state]+1
        
        # TODO: Learn policy based on state, action, reward
        # Q_hat(current state, next action) <--alpha-- (reward for reaching future state) + gamma*Q_hat(future state, action that would give max value for that future state))
        # Replacing above equation with one time step before
        # Q_hat(past state, past action) <--alpha-- past reward + gamma*Q_hat(present state, action that maximizes Q val for present state)
        # some action was taken in the past that brought agent in the present state
        # past details: past_state, past_reward, past_action
        qval_in_past_state = self.Qdf.at[self.past_state, self.past_action]
        max_qval_in_curr_state = np.max(self.Qdf.loc[self.state, :].values)
        
        # learning rate: alpha
        alpha = 0.9
        # gamma: discount to be applied on future value in present time
        gamma = 0.5
        
        # update qval in past state
        self.Qdf.at[self.past_state, self.past_action] = (1-alpha)*qval_in_past_state + alpha*(self.past_reward + gamma*max_qval_in_curr_state)
        
        # TODO: Select action according to your policy
        action_arr = ['forward', 'left', 'right', None]
        # action = random.sample(action_arr,1)[0]
        # action = self.next_waypoint
        # action should be such that qval is maximized.
        qidx = np.argmax(self.Qdf.loc[self.state, :].values)
        action = action_arr[qidx]
        
        # the agent must explore and exploit at the same time
        # the agent should choose a random action sometimes but how ???
        # most of the values unknown in Q are of states that are not visited
        # agent should explore those states that it has not visited
        # how can the agent make itself visit those states ???
        # qval = self.Qdf.at[self.state, qaction]
        # Using simulated annealing like approach, (greedy exploration)
        # the agent takes the action learnt from policy with probability 1-epsilon
        # and a random action with probability epsilon
        # The agent with this approach will be able to visit all the states
        # However, once the agent has explored all the states, it does not make sense that it keeps 
        # taking random choices. Hence, a decaying epsilon makes more sense
        epsilon = 0.25# starts with choosing random action 25% of times
        decayed_epsilon = epsilon/self.epsilon_iter
        #pick a random number
        rdm = random.uniform(0, 1)
        if self.state_visit_hist[self.state]>=50:#(rdm < 1 - decayed_epsilon):
            action = action
        else:
            action = random.sample(action_arr,1)[0]
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        #save state, action and reward to be used in future time step
        # if action == None:
            # action = "None"
        if action == None:
            qaction = "None"
        else:
            qaction = action
        self.past_action = qaction
        self.past_reward = reward
        self.past_state = self.state
        # self.epsilon_iter = self.epsilon_iter + 1
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "\nQdf:\n", self.Qdf
        print "\nstate_visit_hist:\n", self.state_visit_hist

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
