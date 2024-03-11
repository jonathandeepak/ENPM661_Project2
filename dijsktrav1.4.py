import cv2
import numpy as np
import heapq
import time
import math

# Define colors
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# Define actions
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
COSTS = [1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4]

# Step 1: Define actions
def move_up(node):
    return (node[0], node[1] - 1)

def move_down(node):
    return (node[0], node[1] + 1)

def move_left(node):
    return (node[0] - 1, node[1])

def move_right(node):
    return (node[0] + 1, node[1])

def move_up_left(node):
    return (node[0] - 1, node[1] - 1)

def move_up_right(node):
    return (node[0] + 1, node[1] - 1)

def move_down_left(node):
    return (node[0] - 1, node[1] + 1)

def move_down_right(node):
    return (node[0] + 1, node[1] + 1)

# Step 2: Generate map with clearance
def generate_map(clearance=5):
    canvas = np.ones((550, 1250, 3), dtype="uint8") * 255  # White background

    # Draw wall
    cv2.rectangle(canvas, (0, 0), (1200, 500), BLACK, 1)

    # Draw obstacles
    cv2.rectangle(canvas, (100 - clearance, 0 - clearance), (175 + clearance, 400 + clearance), RED, -1)
    cv2.rectangle(canvas, (275 - clearance, 500 - clearance), (350 + clearance, 100 + clearance), RED, -1)
    vertices = np.array([[650 - clearance, 100 - clearance], [800 + clearance, 175 - clearance],
                         [800 + clearance, 325 + clearance], [650 + clearance, 400 + clearance],
                         [500 - clearance, 325 + clearance], [500 - clearance, 175 - clearance]], dtype=np.int32)
    cv2.fillPoly(canvas, [vertices], RED)
    cv2.rectangle(canvas, (1020 - clearance, 50 - clearance), (1100 + clearance, 450 + clearance), RED , -1)
    cv2.rectangle(canvas, (900 - clearance, 375 - clearance), (1020 + clearance, 450 + clearance), RED, -1)
    cv2.rectangle(canvas, (900 - clearance, 50 - clearance), (1020 + clearance, 125 + clearance), RED, -1)
    # cv2.imshow('canvas', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return canvas

def dijkstra(start, goal, canvas, clearance=5):
    open_list = []
    closed_list = set()
    parent = {}
    cost_to_come = {}

    heapq.heappush(open_list, (0, start))
    cost_to_come[start] = 0
    parent[start] = None

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == goal:
            backtrack_path = []
            node = goal
            while node is not None:
                backtrack_path.append(node)
                node = parent[node]
            backtrack_path.reverse()
            return backtrack_path

        closed_list.add(current_node)

        for action, cost in zip(ACTIONS, COSTS):
            next_node = (current_node[0] + action[0], current_node[1] + action[1])

            if (
                    next_node not in closed_list
                    and not is_obstacle(next_node, clearance)
            ):
                new_cost = current_cost + cost
                if next_node not in [node[1] for node in open_list]:
                    cost_to_come[next_node] = new_cost
                    parent[next_node] = current_node
                    heapq.heappush(open_list, (new_cost, next_node))
                elif new_cost < cost_to_come[next_node]:
                    cost_to_come[next_node] = new_cost
                    parent[next_node] = current_node
                    open_list = [
                        (cost, node) if node != next_node else (new_cost, next_node)
                        for cost, node in open_list
                    ]
                    heapq.heapify(open_list)

    return []

def is_obstacle(node, clearance=25):
    x, y = node

    # Wall
    if y <= clearance or y >= 500 - clearance:
        return True

    # Rectangle 1
    if (100 - clearance <= x <= 175 + clearance) and (0 - clearance <= y <= 400 + clearance):
        return True

    # Rectangle 2
    if (275 - clearance <= x <= 350 + clearance) and (100 - clearance <= y <= 500 + clearance):
        return True

    # Polygon
    hexagon_vertices = np.array([[650 - clearance, 100 - clearance], [800 + clearance, 175 - clearance],
                              [800 + clearance, 325 + clearance], [650 + clearance, 400 + clearance],
                              [500 - clearance, 325 + clearance], [500 - clearance, 175 - clearance]])
    n = len(hexagon_vertices)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        v1 = hexagon_vertices[i]
        v2 = hexagon_vertices[j]
        if (v2[1] > y) != (v1[1] > y) and x < (v1[0] - v2[0]) * (y - v1[1]) / (v1[1] - v2[1]) + v1[0]:
            inside = not inside
    if inside:
        return True

    # Rectangle 3
    if (1020 - clearance <= x <= 1100 + clearance) and (50 - clearance <= y <= 450 + clearance):
        return True

    # Rectangle 4
    if (900 - clearance <= x <= 1020 + clearance) and (375 - clearance <= y <= 450 + clearance):
        return True

    # Rectangle 5
    if (900 - clearance <= x <= 1020 + clearance) and (50 - clearance <= y <= 125 + clearance):
        return True

    return False
# Step 4: Optimal path (backtracking)
# Implemented in the dijkstra function

# Step 5: Represent the optimal path
def visualize_path(canvas, path):
    height, width, _ = canvas.shape

    for i in range(len(path) - 1):
        cv2.line(canvas, path[i], path[i + 1], BLUE, 2)

    cv2.namedWindow("Optimal Path", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optimal Path", width // 2, height // 2)
    cv2.imshow("Optimal Path", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save animation as a video
    out = cv2.VideoWriter("optimal_path.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for node in path:
        canvas_copy = canvas.copy()
        cv2.circle(canvas_copy, node, 2, BLUE, -1)
        out.write(canvas_copy)
    out.release()



# Main function
def main():
    start_x = 50 #int(input("Enter starting x coordinate: "))
    start_y = 50#int(input("Enter starting y coordinate: "))
    start = (start_x, start_y)


    goal_x = 1150#int(input("Enter goal x coordinate: "))
    goal_y = 200#int(input("Enter goal y coordinate: "))
    goal = (goal_x, goal_y)

    canvas = generate_map()

    if (
        canvas[start_y, start_x, :].tolist() == RED
        or canvas[goal_y, goal_x, :].tolist() == RED
    ):
        print("Invalid start or goal coordinates. Please try again.")
        return

    start_time = time.time()
    path = dijkstra(start, goal, canvas)
    end_time = time.time()

    if not path:
        print("No path found between start and goal.")
    else:
        print(f"Path found in {end_time - start_time:.2f} seconds.")
        # print(f"Cost of the path: {cost}")
        visualize_path(canvas, path)

if __name__ == "__main__":
    main()
