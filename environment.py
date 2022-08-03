### DO NOT EDIT THIS FILE ###

import arcade
import numpy as np

import settings


# The Environment class, which defines the dynamics of the world
class Environment:

    # Initialisation function to create a new environment
    def __init__(self):

        # Set the free space area.
        # Each free space block is defined by [bottom_left_x, bottom_left_y, top_right_x, top_right_y].
        self.free_spaces = np.array([
                                    [0, 0, 0.4, 0.4],
                                    [0.2, 0.4, 0.4, 0.6],
                                    [0.1, 0.5, 0.2, 0.7],
                                    [0.15, 0.7, 0.4, 0.9],
                                    [0.4, 0.8, 0.9, 0.85],
                                    [0.8, 0.6, 0.9, 0.8],
                                    [0.625, 0.6, 0.8, 0.7],
                                    [0.625, 0.45, 0.675, 0.6],
                                    [0.5, 0.45, 0.8, 0.5],
                                    [0.5, 0.2, 0.6, 0.45],
                                    [0.7, 0.2, 0.8, 0.45],
                                    [0.55, 0.1, 0.75, 0.2]
                                    ])
        self.num_free_spaces = len(self.free_spaces)
        # Set the goal's state
        self.goal_state = np.array([0.65, 0.15])

    # Function to draw the environment onto the screen
    def draw(self):

        # Draw an obstacle across the entire screen
        arcade.draw_rectangle_filled(settings.SCREEN_SIZE * 0.5,
                                     settings.SCREEN_SIZE * 0.5,
                                     settings.SCREEN_SIZE * 1,
                                     settings.SCREEN_SIZE * 1,
                                     color=settings.OBSTACLE_COLOUR)

        # Draw the free spaces onto the screen
        for free_space_num in range(self.num_free_spaces):
            free_space = self.free_spaces[free_space_num]
            centre_x = 0.5 * (free_space[0] + free_space[2])
            centre_y = 0.5 * (free_space[1] + free_space[3])
            width = free_space[2] - free_space[0]
            height = free_space[3] - free_space[1]
            if free_space_num == 0:
                colour = settings.INIT_REGION_COLOUR
            else:
                colour = settings.FREE_SPACE_COLOUR
            arcade.draw_rectangle_filled(settings.SCREEN_SIZE * centre_x,
                                     settings.SCREEN_SIZE * centre_y,
                                     settings.SCREEN_SIZE * width,
                                     settings.SCREEN_SIZE * height,
                                         color=colour)

        # Draw the goal onto the screen
        arcade.draw_circle_filled(settings.SCREEN_SIZE * self.goal_state[0],
                                  settings.SCREEN_SIZE * self.goal_state[1], settings.SCREEN_SIZE * settings.GOAL_SIZE,
                                  settings.GOAL_COLOUR)

    # The dynamics function, which returns the robot's next state given its current state and current action
    def dynamics(self, robot_state, robot_action):

        # First, set the robot's next state by assuming that the movement is in free space
        robot_next_state = robot_state + robot_action
        # Check if the robot's next state is inside any of the free spaces
        is_in_free_space = False
        for free_space_num in range(self.num_free_spaces):
            free_space = self.free_spaces[free_space_num]
            if robot_next_state[0] > free_space[0]:
                if robot_next_state[0] < free_space[2]:
                    if robot_next_state[1] > free_space[1]:
                        if robot_next_state[1] < free_space[3]:
                            is_in_free_space = True

        # The dynamics is dependent on whether the robot is attempting to move through free space or not
        if is_in_free_space:
            robot_next_state = robot_state + robot_action
        else:
            robot_next_state = robot_state
        return robot_next_state