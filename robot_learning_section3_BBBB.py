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




    def train_neural_net(self,batch_size):

        random.shuffle(dataset_moves)

        nb_data = len(dataset_moves)

        train_set = np.array(dataset_moves[:int(0.8 * nb_data)])

        test_set = np.array(dataset_moves[int(0.8 * nb_data):])

        '''
        print("heya")
        print(len(train_set))
        print(train_set)
        print(train_set[0][1])
        '''

        train_input_data = np.array([np.array([elt[0][0], elt[0][1]]) for elt in train_set])
        train_label_data = np.array([np.array([elt[1][0], elt[1][1]]) for elt in train_set])

        test_input_data = np.array([np.array([elt[0][0], elt[0][1]]) for elt in test_set])
        test_label_data = np.array([np.array([elt[1][0], elt[1][1]]) for elt in test_set])

        '''
        print(len(train_label_data))
        print(train_label_data)

        print(len(test_label_data))
        print(test_label_data)
        '''

        test_input_tensor = torch.tensor(test_input_data).float()
        test_label_tensor = torch.tensor(test_label_data).float()
        nb_tests = len(test_input_data)

        # Create the neural network
        network = Network(input_dimension=2, output_dimension=2)
        # Create the optimiser
        optimiser = torch.optim.Adam(network.parameters(), lr=0.001) #lr=0.0005

        # Create lists to store the losses and epochs
        losses = []
        iterations = []
        test_losses = []

        # Create a graph which will show the loss as a function of the number of training iterations
        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

        # Loop over training iterations
        for training_iteration in range(500): # avant 500 !! derner 750
            # Set all the gradients stored in the optimiser to zero.
            optimiser.zero_grad()
            # Sample a mini-batch of size 5 from the training data

            #print(int(len(train_input_data)/2)) ==== size du batch



            ##REMETTRE CA SI CA MARCHE PAS
            minibatch_indices = np.random.choice(range(len(train_input_data)), int(len(train_input_data) /5))
            #minibatch_indices = np.random.choice(range(len(train_input_data)), batch_size)


            minibatch_inputs = train_input_data[minibatch_indices]
            minibatch_labels = train_label_data[minibatch_indices]

            # Convert the NumPy array into a Torch tensor
            minibatch_input_tensor = torch.tensor(minibatch_inputs).float()
            minibatch_labels_tensor = torch.tensor(minibatch_labels).float()
            # Do a forward pass of the network using the inputs batch

            #print("minibatch_input_tensor")
            #print(minibatch_input_tensor)

            network_prediction = network.forward(minibatch_input_tensor)
            # Compute the loss based on the label's batch

            #print("prediction")
            #print(network_prediction)
            #print("labels")
            #print(minibatch_labels_tensor)


            loss = torch.nn.MSELoss()(network_prediction, minibatch_labels_tensor)
            # Compute the gradients based on this loss,
            # i.e. the gradients of the loss with respect to the network parameters.
            loss.backward()
            # Take one gradient step to update the network
            optimiser.step()
            # Get the loss as a scalar value
            loss_value = loss.item()
            # Print out this loss
            print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss_value))
            # Store this loss in the list
            losses.append(loss_value)
            # Update the list of iterations
            iterations.append(training_iteration)
            # Plot and save the loss vs iterations graph

            test_loss = 0

            for test_i in range(nb_tests):
                test_network_prediction = network.forward(test_input_tensor[test_i])

                test_loss += torch.nn.MSELoss()(test_network_prediction, test_label_tensor[test_i])

            test_loss = test_loss / nb_tests

            print(test_loss)

            test_losses.append(test_loss.item())

        ax.plot(iterations, losses, color='blue')
        ax.plot(iterations, test_losses, color='red')
        plt.yscale('log')
        plt.show()
        fig.savefig("loss_vs_iterations.png")

        return network



class Demonstrator:

    def __init__(self,agent):
        self.policy_demonstrator=None

        self.robot=agent.robot

        self.waypoints = [np.array([0.2, 0.55]), np.array([0.45, 0.82]), np.array([0.85, 0.8]), np.array([0.65, 0.5]),np.array([0.65, 0.15])]

    def compute_actions(self, initial_state_origin, num_sequences, num_actions_per_sequence, environment):

        initial_state = initial_state_origin

        nb_iterations = 50  # 100
        num_top_k_percent = 10  # 5


        for waypoint in self.waypoints:

            print("waypoint")
            print(waypoint)
            print("initial_state")
            print(initial_state)


            if (waypoint != self.waypoints[0]).all():
                self.robot.state = initial_state

                last_path = self.simulate_action_sequence(num_actions_per_sequence, initial_state, mean_action_sequence,
                                                          environment)

                initial_state = last_path[-1]



                self.robot.state = initial_state

            for iter_num in range(nb_iterations):

                if iter_num == 0:
                    action_sequences = np.random.uniform(self.robot.min_action, self.robot.max_action,
                                                         [num_sequences, num_actions_per_sequence, 2])
                else:

                    action_sequences_flat = np.random.multivariate_normal(mean_action_sequence_flat,
                                                                          covar_action_sequence_flat, num_sequences)
                    action_sequences = np.reshape(action_sequences_flat, [num_sequences, num_actions_per_sequence, 2])

                for action_sequence in action_sequences:
                    for action_num in range(num_actions_per_sequence):
                        action = action_sequence[action_num]
                        action[0] = np.clip(action[0], self.robot.min_action, self.robot.max_action)
                        action[1] = np.clip(action[1], self.robot.min_action, self.robot.max_action)
                        action_sequence[action_num] = action

                scores = np.zeros(num_sequences, dtype=np.float32)

                for sequence_num in range(num_sequences):
                    action_sequence = action_sequences[sequence_num]

                    path = self.simulate_action_sequence(num_actions_per_sequence, initial_state, action_sequence,
                                                         environment)

                    score = self.evaluate_path(path, waypoint)

                    scores[sequence_num] = score

                num_top = int(num_top_k_percent * 0.01 * num_sequences)
                action_sequences_flat = np.reshape(action_sequences, [num_sequences, 2 * num_actions_per_sequence])
                top_indices = (-scores).argsort()[:num_top]
                top_action_sequences_flat = action_sequences_flat[top_indices]
                mean_action_sequence_flat = np.mean(top_action_sequences_flat, 0)
                mean_action_sequence = np.reshape(mean_action_sequence_flat, [num_actions_per_sequence, 2])
                covar_action_sequence_flat = np.cov(top_action_sequences_flat.transpose())

            # print("mean_action_sequence per step")
            # print(mean_action_sequence)

            if (waypoint == self.waypoints[0]).all():
                final_means = mean_action_sequence
            else:
                final_means = np.concatenate((final_means, mean_action_sequence))

        #print(final_means)

        nb_actions = len(final_means)

        self.robot.state = initial_state_origin

        for i in range(nb_actions):
            action_i = final_means[i]

            old_position = self.robot.state

            self.robot.take_action(action_i, environment)

            dataset_moves.append([old_position, action_i])

        return final_means

    def simulate_action_sequence(self, num_actions_per_sequence, init_state, action_sequence, environment):
        current_state = init_state
        path = np.zeros([num_actions_per_sequence, 2], dtype=np.float32)

        for action_num in range(num_actions_per_sequence):
            action = action_sequence[action_num]
            current_state = environment.dynamics(current_state, action)
            path[action_num] = current_state
        return path

    def evaluate_path(self, path, waypoint):

        final_state = path[-1]
        distance_to_goal = np.linalg.norm(final_state - waypoint)
        score = -distance_to_goal

        return score


# Turn on interactive mode for PyPlot, to prevent the displayed graph from blocking the program flow
plt.ion()


# Create a Network class, which inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input.
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 10 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


# The main Program class
class MainProgram(arcade.Window):

    # Initialisation function to create a new program
    def __init__(self):
        super().__init__(width=settings.SCREEN_SIZE, height=settings.SCREEN_SIZE, title=settings.SCREEN_TITLE, update_rate=1.0/settings.UPDATE_RATE)

        # Create the environment
        self.environment = environment.Environment()

        self.num_sequences = 100
        self.num_actions_per_sequence = 15


        # Create the agent
        self.agent = Agent()

        # Set the environment's background colour
        arcade.set_background_color(settings.BACKGROUND_COLOR)

        #Create the demonstrator

        self.demonstrator=Demonstrator(self.agent)


        nb_demonstrations=[1,2,5,20,100]

        list_batch_size=[8,8,16,32,64]
        indice=0

        self.list_network_nb_demo=[]



        for i in tqdm(range(nb_demonstrations[-1])):

            self.initial_state=np.random.randint(0,4,2)/10

            self.agent.robot.state=self.initial_state

            best_seq_actions=self.demonstrator.compute_actions(initial_state_origin=self.initial_state,num_sequences=self.num_sequences, num_actions_per_sequence=self.num_actions_per_sequence,
                                                                 environment=self.environment)
            self.nb_actions = len(best_seq_actions)

            if i+1 in nb_demonstrations:
                batch_size=list_batch_size[indice]
                self.list_network_nb_demo.append(self.agent.train_neural_net(batch_size))
                indice=+1


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


        self.agent.robot.state = np.array([0.2,0.2])

        list_seqs_positions=[]



        for network in self.list_network_nb_demo:

            self.agent.robot.state = np.array([0.2, 0.2])

            positions = [self.agent.robot.state]

            for i in range(self.nb_actions):

                position = torch.tensor(self.agent.robot.state).float()

                action_i=network.forward(position).detach().numpy()
                #print("action")
                #print(action_i)
                self.agent.robot.take_action(action_i, self.environment)
                positions.append(self.agent.robot.state)

            list_seqs_positions.append(positions)

        #all appart from the first and last

        #print(positions)

        print("list_seqs_positions")
        print(list_seqs_positions)


        lists_seqs_positions=[]

        for positions in list_seqs_positions:

            scaled_positions=[]

            for i in range(self.nb_actions):

                pos=[settings.SCREEN_SIZE * positions[i][0],settings.SCREEN_SIZE * positions[i][1]]
                scaled_positions.append(pos)
                if i==0:
                    arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

                elif i==self.nb_actions-1:
                    arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[0,255,0])

                #else:
                    #arcade.draw_circle_filled(pos[0],pos[1],radius=5, color=[220,220,220])

            lists_seqs_positions.append(scaled_positions)

        arcade.draw_line_strip(point_list=lists_seqs_positions[0],color=[255,128,0],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[1],color=[255,0,255],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[2],color=[0,255,255],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[3],color=[255,255,0],line_width=3)
        arcade.draw_line_strip(point_list=lists_seqs_positions[4],color=[220,220,220],line_width=3)


        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.2,settings.SCREEN_SIZE * 0.55 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.42,settings.SCREEN_SIZE * 0.82 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.85,settings.SCREEN_SIZE * 0.8 , radius=5, color=[121, 28, 248])
        arcade.draw_circle_filled(settings.SCREEN_SIZE * 0.65,settings.SCREEN_SIZE * 0.6 , radius=5, color=[121, 28, 248])



# The main entry point
if __name__ == "__main__":

    # Create a new program, which will also do the robot's initial planning
    MainProgram()

    # Run the main Arcade loop forever
    # This will repeatedly call the MainProgram.on_update() and MainProgram.on_draw() functions.
    arcade.run()
