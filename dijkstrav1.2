import heapq
import cv2
import numpy as np
def dijkstra(start, goal, obstacle_map):
    """
    Implements the Dijkstra algorithm to find the shortest path
    between the start and goal nodes, avoiding obstacles.
    """
    open_list = []
    closed_set = set()
    open_dict = {}

    start_node = (start[0], start[1], 0, None)  # (x, y, cost, parent)
    heapq.heappush(open_list, start_node)
    open_dict[start_node] = 0

    while open_list:
        current_node = heapq.heappop(open_list)
        x, y, cost, parent = current_node

        # Check if the current node is the goal
        if (x, y) == goal:
            return backtrack(current_node)

        # Add the current node to the closed set
        closed_set.add(current_node)

        # Generate valid successors
        for action in ACTION_SET:
            new_x, new_y = action((x, y))

            # Check if the new node is within the map boundaries
            if (0 <= new_x < obstacle_map.shape[1] and
                    0 <= new_y < obstacle_map.shape[0]):

                # Check if the new node is not an obstacle
                if not np.any(obstacle_map[new_y, new_x] != 255):  # Assuming obstacles are black (0, 0, 0)
                    new_cost = cost + ACTION_COSTS[action]
                    new_node = (new_x, new_y, new_cost, current_node)

                    if new_node not in closed_set:
                        if new_node in open_dict:
                            if open_dict[new_node] > new_cost:
                                open_dict[new_node] = new_cost
                                heapq.heappush(open_list, new_node)
                        else:
                            open_dict[new_node] = new_cost
                            heapq.heappush(open_list, new_node)

    # If the goal is not reachable, return None
    return None

def backtrack(goal_node):
    """
    Backtracks from the goal node to the start node
    to find the optimal path.
    """
    if goal_node is None:
        return []

    path = []
    current_node = goal_node

    while current_node is not None:
        path.append(current_node[:2])
        current_node = current_node[3]

    path.reverse()
    return path
# Get start and goal coordinates from the user
start = tuple(map(int, input("Enter start coordinates (x, y): ").split(',')))
goal = tuple(map(int, input("Enter goal coordinates (x, y): ").split(',')))

# Check if start and goal are valid
if not (0 <= start[0] < canvas.shape[1] and 0 <= start[1] < canvas.shape[0]) or \
   not (0 <= goal[0] < canvas.shape[1] and 0 <= goal[1] < canvas.shape[0]):
    print("Start or goal coordinates are out of bounds. Please try again.")
    exit()


canvas = np.ones((550, 1250, 3), dtype = "uint8")*255
black = (0, 0, 0)
red = (0, 0, 255)
cv2.rectangle(canvas, (0, 0), (1200, 500), black,1)
# draw a red 50x50 pixel square, starting at 10x10 and ending at 60x60
cv2.rectangle(canvas, (100, 0), (175, 400), red,-1)
# cv2.imshow("Canvas", canvas)

# draw a red 50x50 pixel square, starting at 10x10 and ending at 60x60
cv2.rectangle(canvas, (275, 500), (350, 100), red,-1)
# cv2.imshow("Canvas", canvas)

# Define vertices of the hexagon
vertices = [[650,100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]]

# Draw the filled hexagon
cv2.fillPoly(canvas, [np.array(vertices, dtype=np.int32)], red)

cv2.rectangle(canvas, (1020, 50), (1100, 450), red,-1)
cv2.rectangle(canvas, (900, 375), (1020, 450), red,-1)
cv2.rectangle(canvas, (900, 50), (1020, 125), red,-1)


# Display the result
cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Action set and action costs (assuming 8-connected grid)
ACTION_SET = [lambda x: (x[0], x[1] + 1),  # Move up
              lambda x: (x[0], x[1] - 1),  # Move down
              lambda x: (x[0] - 1, x[1]),  # Move left
              lambda x: (x[0] + 1, x[1]),  # Move right
              lambda x: (x[0] - 1, x[1] + 1),  # Move up-left
              lambda x: (x[0] + 1, x[1] + 1),  # Move up-right
              lambda x: (x[0] - 1, x[1] - 1),  # Move down-left
              lambda x: (x[0] + 1, x[1] - 1)]  # Move down-right

ACTION_COSTS = {
    ACTION_SET[0]: 1.0,  # Move up
    ACTION_SET[1]: 1.0,  # Move down
    ACTION_SET[2]: 1.0,  # Move left
    ACTION_SET[3]: 1.0,  # Move right
    ACTION_SET[4]: 1.4,  # Move up-left
    ACTION_SET[5]: 1.4,  # Move up-right
    ACTION_SET[6]: 1.4,  # Move down-left
    ACTION_SET[7]: 1.4   # Move down-right
}



# Run the Dijkstra algorithm
obstacle_map = canvas  # Assuming obstacles are black (0, 0, 0) and free spaces are white (255, 255, 255)
optimal_path = dijkstra(start, goal, obstacle_map)

if optimal_path is not None:
    print("Optimal path:", optimal_path)
    # You can add code here to visualize the optimal path on the canvas
else:
    print("Goal is not reachable.")
