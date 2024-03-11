import numpy as np
import cv2
import heapq

def point_in_free_space(point, canvas_size):
    """
    Checks if a given point lies in free space (not inside any obstacles).
    """
    # Check if the point is inside the canvas
    if 0 <= point[0] < canvas_size[1] and 0 <= point[1] < canvas_size[0]:
        if np.all(canvas[point[1], point[0]] == [255, 255, 255]):

            return True  # Point is in free space
    return False  # Point is outside the canvas or inside an obstacle

def dijkstra(graph, start, goal):
    """
    Dijkstra's algorithm to find the shortest path between start and goal.
    """
    pq = []  # Priority queue for Dijkstra
    heapq.heappush(pq, (0, start))  # Start node with distance 0
    visited = set()  # Visited nodes
    distances = {node: float('inf') for node in graph}  # Initialize distances to infinity
    distances[start] = 0

    while pq:
        dist, current = heapq.heappop(pq)
        if current == goal:
            return distances[current]  # Shortest distance to goal
        if current in visited:
            continue
        visited.add(current)

        for neighbor, weight in graph[current].items():
            if distances[current] + weight < distances[neighbor]:
                distances[neighbor] = distances[current] + weight
                heapq.heappush(pq, (distances[neighbor], neighbor))

    return float('inf')  # No path found


# Define colors
black = (0, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)

# Create an empty canvas
canvas = np.ones((550, 1250, 3), dtype="uint8") * 255  # White background

# Draw wall
cv2.rectangle(canvas, (0, 0), (1200, 500), black, 1)

# Draw obstacles
cv2.rectangle(canvas, (100, 0), (175, 400), red, -1)
cv2.rectangle(canvas, (275, 500), (350, 100), red, -1)
vertices = np.array([[650,100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]], dtype=np.int32)
cv2.fillPoly(canvas, [vertices], red)
cv2.rectangle(canvas, (1020, 50), (1100, 450), red, -1)
cv2.rectangle(canvas, (900, 375), (1020, 450), red, -1)
cv2.rectangle(canvas, (900, 50), (1020, 125), red, -1)

# Define clearance offset
clearance = 5  # 5 mm

# Create clearance polygons for each object
cv2.rectangle(canvas, (100-clearance, 0), (175+clearance, 400+clearance), green, 1)
cv2.rectangle(canvas, (275-clearance, 500), (350+clearance, 100-clearance), green, 1)
clearance_vertices = vertices + np.array([[-clearance, -clearance], [+clearance, -clearance], [+clearance, +clearance], [+clearance, +clearance], [-clearance, +clearance], [-clearance, -clearance]])
cv2.polylines(canvas, [clearance_vertices], True, green, 1)
cv2.rectangle(canvas, (1020-clearance, 50-clearance), (1100+clearance, 450+clearance), green, 1)
cv2.rectangle(canvas, (900-clearance, 375-clearance), (1020, 450+clearance), green, 1)
cv2.rectangle(canvas, (900-clearance, 50-clearance), (1020, 125+clearance), green, 1)

# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Set start and goal coordinates
start = (50, 50)
goal = (1150, 100)

# Define nodes (points) on the canvas
nodes = []
for i in range(0, 1250, 50):
    for j in range(0, 550, 50):
        if point_in_free_space((j, i), canvas.shape[:2]):
            nodes.append((j, i))

# Define edges between adjacent nodes (up, down, left, right)
graph = {}
for node in nodes:
    graph[node] = {}
    x, y = node
    for dx, dy in [(0, 50), (0, -50), (50, 0), (-50, 0)]:
        next_node = (x + dx, y + dy)
        if next_node in nodes:
            graph[node][next_node] = 1  # Edge weight = 1

# Find the shortest path distance using Dijkstra's algorithm
shortest_distance = dijkstra(graph, start, goal)
if shortest_distance != float('inf'):
    print("Shortest path distance:", shortest_distance)
else:
    print("No valid path found.")

