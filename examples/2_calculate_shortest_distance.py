"""
The MIT License (MIT)

Copyright (c) 2016 Christian August Reksten-Monsen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import pyvisgraph as vg
from haversine import haversine
import time

# In this example we will find the shortest path between two points on a
# sphere, i.e. on earth. To calculate the total distance of that path, we
# need to use the great circle formula. We use the haversine package for this. 

# Example points
start_point = vg.Point(12.568337, 55.676098) # Copenhagen
end_point = vg.Point(103.851959, 1.290270) # Singapore

# Load the visibility graph file. If you do not have this, please run 
# 1_build_graph_from_shapefiles.py first.
graph = vg.VisGraph()
graph.load('GSHHS_c_L1.graph')

# Get the shortest path with dijkstra's
startTime = time.time()
shortest_path = graph.shortest_path(start_point, end_point)
print("Solved with Dijkstra's algorithm in {} seconds.".format(time.time() - startTime))

# Get the shortest path with A*
startTime = time.time()
shortest_path = graph.shortest_path(start_point, end_point, solver = "astar")
print("Solved with A* algorithm in {} seconds.".format(time.time() - startTime))


# Calculate the total distance of the shortest path in km
path_distance = 0
prev_point = shortest_path[0]
for point in shortest_path[1:]:
    # Add miles=True to the end of the haversine call to get result in miles
    path_distance += haversine((prev_point.y, prev_point.x), (point.y, point.x))
    prev_point = point
# If you want to total distance in nautical miles:
# path_distance = path_distance*0.539957

print('Shortest path distance: {}'.format(path_distance))
