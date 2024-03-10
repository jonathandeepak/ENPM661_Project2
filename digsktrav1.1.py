import cv2
import numpy as np
import time
from collections import deque
import heapq


def move_up(node):
    """
    Returns the new node state after moving up (0, 1)
    """
    return (node[0], node[1] + 1)


def move_down(node):
    """
    Returns the new node state after moving down (0, -1)
    """
    return (node[0], node[1] - 1)


def move_left(node):
    """
    Returns the new node state after moving left (-1, 0)
    """
    return (node[0] - 1, node[1])


def move_right(node):
    """
    Returns the new node state after moving right (1, 0)
    """
    return (node[0] + 1, node[1])


def move_up_left(node):
    """
    Returns the new node state after moving up-left (-1, 1)
    """
    return (node[0] - 1, node[1] + 1)


def move_up_right(node):
    """
    Returns the new node state after moving up-right (1, 1)
    """
    return (node[0] + 1, node[1] + 1)


def move_down_left(node):
    """
    Returns the new node state after moving down-left (-1, -1)
    """
    return (node[0] - 1, node[1] - 1)


def move_down_right(node):
    """
    Returns the new node state after moving down-right (1, -1)
    """
    return (node[0] + 1, node[1] - 1)


# Action set
ACTION_SET = [move_up, move_down, move_left, move_right,
              move_up_left, move_up_right, move_down_left, move_down_right]

ACTION_COSTS = {
    move_up: 1.0,
    move_down: 1.0,
    move_left: 1.0,
    move_right: 1.0,
    move_up_left: 1.4,
    move_up_right: 1.4,
    move_down_left: 1.4,
    move_down_right: 1.4
}


def dijkstra(start, goal, obstacle_map, clearance=5):
    """
    Implements the Dijkstra algorithm to find the shortest path
    between the start and goal nodes.
    """
    open_list = []
    closed_list = deque()
    start_node = (start[0], start[1], 0, None)  # (x, y, cost, parent)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        x, y, cost, parent = current_node

        # Check if the current node is the goal
        if (x, y) == goal:
            return closed_list, current_node

        # Add the current node to the closed list
        closed_list.append(current_node)

        # Generate valid successors
        for action in ACTION_SET:
            new_x, new_y = action((x, y))

            # Check if the new node is within the map boundaries
            if (0 <= new_x < obstacle_map.shape[1] and
                    0 <= new_y < obstacle_map.shape[0]):

                # Check if the new node is in the obstacle space (check pixel color)
                if not np.any(obstacle_map[new_y, new_x] != 0):  # Assuming 0 represents free space
                    new_cost = cost + ACTION_COSTS[action]
                    new_node = (new_x, new_y, new_cost, current_node)

                    # Check if the new node is not in the closed list
                    if new_node not in closed_list:
                        # Update the cost if a cheaper path is found
                        node_in_open_list = [node for node in open_list if node[:2] == (new_x, new_y)]
                        if node_in_open_list and node_in_open_list[0][2] > new_cost:
                            open_list.remove(node_in_open_list[0])
                            heapq.heappush(open_list, new_node)
                        elif not node_in_open_list:
                            heapq.heappush(open_list, new_node)

    # If the goal is not reachable, return the closed list
    return closed_list, None


def backtrack(goal_node):
    """
    Backtracks from the goal node to the start node
    to find the optimal path.
    """
    path = []
    current_node = goal_node

    while current_node is not None:
        path.append(current_node[:2])  # Append (x, y) coordinates
        current_node = current_node[3]  # Move to parent node

    path.reverse()  # Reverse the path to start from the start node
    return path


def visualize_path(obstacle_map, closed_list, optimal_path, start, goal):
    """
    Visualize the exploration and optimal path on the map.
    """
    # Create a copy of the obstacle map for visualization
    visualization_map = obstacle_map.copy()

    # Color the start and goal nodes
    visualization_map = cv2.circle(visualization_map, start, 5, (0, 255, 0), -1)  # Start node (green)
    visualization_map = cv2.circle(visualization_map, goal, 5, (0, 0, 255), -1)  # Goal node (red)

    # Color the explored nodes (closed list)
    for node in closed_list:
        x, y, _, _ = node
        visualization_map = cv2.circle(visualization_map, (x, y), 2, (255, 0, 0), -1)  # Explored nodes (blue)

    # Color the optimal path
    for i in range(len(optimal_path) - 1):
        start_point = optimal_path[i]
        end_point = optimal_path[i + 1]
        visualization_map = cv2.line(visualization_map, start_point, end_point, (0, 255, 255),
                                     2)  # Optimal path (yellow)

    # Display the visualization
    cv2.imshow("Exploration and Optimal Path", visualization_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



red = (0, 0, 255)

# Create a canvas (map) and draw obstacles

canvas = np.ones((550, 1250, 3), dtype="uint8") * 255
cv2.rectangle(canvas, (0, 0), (1200, 500), red, 1)
# Draw obstacles on the canvas
cv2.rectangle(canvas, (100, 0), (175, 400), red, -1)
cv2.rectangle(canvas, (275, 500), (350, 100), red, -1)
vertices = [[650, 100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]]
cv2.fillPoly(canvas, [np.array(vertices, dtype=np.int32)], red)
cv2.rectangle(canvas, (1020, 50), (1100, 450), red, -1)
cv2.rectangle(canvas, (900, 375), (1020, 450), red, -1)
cv2.rectangle(canvas, (900, 50), (1020, 125), red, -1)


# Convert the canvas to a grayscale image
gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray_canvas)
# cv2.waitKey(0)
print((type(gray_canvas)))

# Create a numpy array to store the map
map_data = np.zeros_like(gray_canvas)

# Threshold the grayscale image to separate obstacles from free spaces
_, threshed = cv2.threshold(gray_canvas, 254, 1, cv2.THRESH_BINARY_INV)

# Set obstacle positions to 1 in the map array
map_data[threshed == 1] = 1

# You can now use map_data with the Dijkstra algorithm
# Get start and goal coordinates from the user
# start = tuple(map(int, input("Enter start coordinates (x, y): ").split(',')))
# goal = tuple(map(int, input("Enter goal coordinates (x, y): ").split(',')))
start = (10, 10)
goal = (1175, 475)

# Check if start and goal are valid
if np.any(gray_canvas[start[1], start[0]] != 0) or np.any(gray_canvas[goal[1], goal[0]] != 0):
    print("Start or goal is in obstacle space. Please try again.")

# Run the Dijkstra algorithm
closed_list, goal_node = dijkstra(start, goal, gray_canvas)

if goal_node is not None:
    optimal_path = backtrack(goal_node)
    print("Optimal path:", optimal_path)
    visualize_path(gray_canvas, closed_list, optimal_path, start, goal)
else:
    print("Goal is not reachable.")

