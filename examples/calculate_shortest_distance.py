#!/usr/bin/env python

import pyvisgraph as vg
from geopy.distance import great_circle

# Example points
point1 = (12.568337, 55.676098) # Copenhagen
point2 = (103.851959, 1.290270) # Singapore

# Load the graph file
graph = vg.VisGraph()
graph.load('GSHHS_c_L1.graph')

def shortest_path(graph, p1, p2):
    ''' Helper function to get the shortest path'''
    start_point = vg.Point(p1[0], p1[1])
    end_point   = vg.Point(p2[0], p2[1])
    path        = graph.shortest_path(start_point, end_point)
    return path

def shortest_distance(graph, p1, p2):
    ''' Helper function to get the distance of the shortest path'''
    path = shortest_path(graph, p1, p2)
    return distance(path)

def distance(path):
    '''Calculate the distance as the sum of the legs of the journey'''
    totaldistance = 0
    lastelement   = None
    for element in path:
        if lastelement:
            totaldistance += single_distance(lastelement, element)
        lastelement = element
    return totaldistance

def single_distance(p1, p2):
    ''' Get the distance between to points on a sphere in nautical miles'''
    return great_circle((p1.y, p1.x), (p2.y, p2.x)).nm # Change nm to km to change unit, see geopy distance docs

# Calculate the shortest distance and print it
shortest_dist = shortest_distance(graph, point1, point2)
print('Shortest distance: {}'.format(shortest_dist))
