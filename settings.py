# Constants for the main program loop
SCREEN_SIZE = 600  # Width and height of the displayed on the screen (can be adjusted if you like)
SCREEN_TITLE = "Robot Learning Tutorial 3"
UPDATE_RATE = 10  # Number of times per second that the main loop is running at
BACKGROUND_COLOR = (0, 0, 0)

# Constants for visualising the robot
ROBOT_SIZE = 0.01  # Relative to the world size, which ranges from 0 to 1
ROBOT_COLOUR = (50, 100, 255)

# Constants for visualising the goal
GOAL_SIZE = 0.01  # Relative to the world size, which ranges from 0 to 1
GOAL_COLOUR = (255, 50, 100)

# Constants for visualising the free space and the obstacles
INIT_REGION_COLOUR = (30, 30, 60)
FREE_SPACE_COLOUR = (0, 0, 0)
OBSTACLE_COLOUR = (150, 150, 150)


NUM_ACTIONS_PER_SEQUENCE=60
NUM_ACTION_SEQUENCES=1000
NUM_ITERATIONS=10
NUM_TOP_K_PERCENT=5