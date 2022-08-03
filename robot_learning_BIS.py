# Imports
import arcade
import numpy as np
import random
import settings
import environment
import robot
import torch
from matplotlib import pyplot as plt
from tqdm import *


dataset_moves=[]

# The Agent class, which represents the robot's "brain"
class Agent:

    def __init__(self):

        # Create a robot, which represents the physical robot in the environment (the agent is just the "brain")
        self.robot = robot.Robot()


    # Function to take a physical action in the physical environment
    def take_action(self, environment):

        # Get the next action
        # Currently, this is just a random action
        next_action = np.random.uniform(self.robot.min_action, self.robot.max_action, 2)
        # Execute this action and hence update the state of the robot
        self.robot.take_action(next_action, environment)


class Demonstrator:

    def __init__(self,agent):
        self.policy_demonstrator=None

        self.robot=agent.robot

        #self.waypoints=[np.array([0.2,0.55]),np.array([0.23,0.75]),np.array([0.45,0.82]),np.array([0.85,0.8]),np.array([0.75,0.65]),np.array([0.65, 0.5]),np.array([0.65, 0.15])]

        #self.waypoints=[np.array([0.2,0.55]),np.array([0.23,0.75]),np.array([0.45,0.82]),np.array([0.85,0.8]),np.array([0.75,0.65]),np.array([0.65, 0.5]),np.array([0.65, 0.15])]

        #self.waypoints=[np.array([0.2,0.55]),np.array([0.45,0.82]),np.array([0.85,0.8]),np.array([0.75,0.65]),np.array([0.65, 0.5]),np.array([0.65, 0.15])]

        self.waypoints=[np.array([0.2,0.55]),np.array([0.45,0.82]),np.array([0.85,0.8]),np.array([0.65, 0.5]),np.array([0.65, 0.15])]



    def compute_actions(self,initial_state,num_sequences, num_actions_per_sequence, environment):

        nb_iterations=50  #100
        num_top_k_percent=10  #5
        print("initial_state")
        print(initial_state)
        for waypoint in self.waypoints:

            print("waypoint")
            print(waypoint)

            if (waypoint!=self.waypoints[0]).all():

                self.robot.state=initial_state

                last_path=self.simulate_action_sequence(num_actions_per_sequence,initial_state, mean_action_sequence,environment)

                initial_state=last_path[-1]

                print("initial_state")
                print(initial_state)


                self.robot.state=initial_state


            for iter_num in tqdm(range(nb_iterations)):

                if iter_num == 0:
                    action_sequences = np.random.uniform(self.robot.min_action, self.robot.max_action, [num_sequences,num_actions_per_sequence, 2])
                else:

                    action_sequences_flat = np.random.multivariate_normal(mean_action_sequence_flat, covar_action_sequence_flat, num_sequences)
                    action_sequences = np.reshape(action_sequences_flat, [num_sequences,num_actions_per_sequence, 2])

                for action_sequence in action_sequences:
                    for action_num in range(num_actions_per_sequence):
                        action = action_sequence[action_num]
                        action[0] = np.clip(action[0], self.robot.min_action, self.robot.max_action)
                        action[1] = np.clip(action[1], self.robot.min_action, self.robot.max_action)
                        action_sequence[action_num]=action

                scores = np.zeros(num_sequences, dtype=np.float32)

                for sequence_num in range(num_sequences):

                    action_sequence = action_sequences[sequence_num]

                    path = self.simulate_action_sequence(num_actions_per_sequence,initial_state, action_sequence,environment)


                    score = self.evaluate_path(path,waypoint)


                    scores[sequence_num] = score

                num_top = int(num_top_k_percent * 0.01 * num_sequences)
                action_sequences_flat = np.reshape(action_sequences, [num_sequences, 2 * num_actions_per_sequence])
                top_indices = (-scores).argsort()[:num_top]
                top_action_sequences_flat = action_sequences_flat[top_indices]
                mean_action_sequence_flat = np.mean(top_action_sequences_flat, 0)
                mean_action_sequence = np.reshape(mean_action_sequence_flat, [num_actions_per_sequence, 2])
                covar_action_sequence_flat = np.cov(top_action_sequences_flat.transpose())


            #print("mean_action_sequence per step")
            #print(mean_action_sequence)

            if (waypoint==self.waypoints[0]).all():
                final_means=mean_action_sequence
            else:
                final_means=np.concatenate((final_means,mean_action_sequence))


        print(final_means)
        return final_means



    def simulate_action_sequence(self,num_actions_per_sequence, init_state, action_sequence,environment):
        current_state = init_state
        path = np.zeros([num_actions_per_sequence, 2], dtype=np.float32)

        for action_num in range(num_actions_per_sequence):
            action = action_sequence[action_num]
            current_state = environment.dynamics(current_state, action)
            path[action_num] = current_state
        return path

    def evaluate_path(self, path,waypoint):

        final_state = path[-1]
        distance_to_goal = np.linalg.norm(final_state - waypoint)
        score = -distance_to_goal

        return score







# The main Program class
class MainProgram(arcade.Window):

    # Initialisation function to create a new program
    def __init__(self):
        super().__init__(width=settings.SCREEN_SIZE, height=settings.SCREEN_SIZE, title=settings.SCREEN_TITLE, update_rate=1.0/settings.UPDATE_RATE)

        # Create the environment
        self.environment = environment.Environment()



        # Create the agent
        self.agent = Agent()

        # Set the environment's background colour
        arcade.set_background_color(settings.BACKGROUND_COLOR)

        #Create the demonstrator

        self.demonstrator=Demonstrator(self.agent)


        self.sequences=[]

        self.list_initial_points=[np.array([0.1,0.1]),np.array([0.1,0.3]),np.array([0.3,0.1]),np.array([0.3,0.3])]


        for initial_state in self.list_initial_points:

            self.initial_state=initial_state

            self.agent.robot.state=self.initial_state

            best_seq_actions=self.demonstrator.compute_actions(initial_state=self.initial_state,num_sequences=100, num_actions_per_sequence=15,
                                                         environment=self.environment)


            self.sequences.append(best_seq_actions)


        #print("self.best_seq_actions")
        #print(self.best_seq_actions)
        #self.agent.train_neural_net()


    '''
    # on_update is called once per loop and is used to update the robot / environment
    def on_update(self, delta_time):

        # On each timestep, the agent will execute an action
        self.agent.take_action(self.environment)

    '''


    # on_draw is called once per loop and is used to draw the environment
    def on_draw(self):

        # Clear the screen
        arcade.start_render()

        # Draw the environment
        self.environment.draw()

        # Draw the robot
        #self.agent.robot.draw()

        # You may want to add code here to draw the policy, the sampled paths, or any other visualisations

        nb_seq=len(self.sequences)


        lists_seqs_positions=[]

        for i in range(nb_seq):
            initial_state=self.list_initial_points[i]
            seq=self.sequences[i]

            nb_actions = len(seq)

            self.agent.robot.state = initial_state

            positions = [self.agent.robot.state]

            for j in range(nb_actions):

                action_j=seq[j]
                #self.agent.robot.take_action(action_i, self.environment)

                position=positions[-1]

                position = self.environment.dynamics(position, action_j)

                positions.append(position)

            #print(positions)
            scaled_positions=[]
            for ii in range(nb_actions):

                pos=[settings.SCREEN_SIZE * positions[ii][0],settings.SCREEN_SIZE * positions[ii][1]]
                scaled_positions.append(pos)

                if ii==0:
                    arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

                elif ii==nb_actions-1:
                    arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

                #else:
                    #arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[220,220,220])

            lists_seqs_positions.append(scaled_positions)


        #print(lists_seqs_positions[0])
        arcade.draw_line_strip(point_list=lists_seqs_positions[0],color=[178, 190, 181],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[1],color=[128, 128, 128],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[2],color=[229, 228, 226],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[3],color=[112, 128, 144],line_width=3)

        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.2,settings.SCREEN_SIZE * 0.55 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.45,settings.SCREEN_SIZE * 0.82 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.75,settings.SCREEN_SIZE * 0.65 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.65,settings.SCREEN_SIZE * 0.5 , radius=5, color=[121, 28, 248])



# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()
