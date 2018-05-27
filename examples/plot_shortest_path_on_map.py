#!/usr/bin/env python

import pyvisgraph as vg
import folium

# Example points
point1 = (12.568337, 55.676098) # Copenhagen
point2 = (103.851959, 1.290270) # Singapore

# Load the graph file
graph = vg.VisGraph()
graph.load('GSHHS_c_L1.graph')

# Helper function to return the shortests path
def shortest_path(graph, p1, p2):
    start_point = vg.Point(p1[0], p1[1])
    end_point   = vg.Point(p2[0], p2[1])
    path        = graph.shortest_path(start_point, end_point)
    return path

# Request the shortest path
geopath_raw = shortest_path(graph, point1,point2)

# Plot of the path using folium
geopath = [[p.y, p.x] for p in geopath_raw]
geomap  = folium.Map([0, 0], zoom_start=2)
for p in geopath:
    folium.Marker(p, popup=str(p)).add_to(geomap)
folium.PolyLine(geopath).add_to(geomap)

# Mark up the original positions, to see how much we moved them out the polygons
folium.Marker(point1[::-1], popup=str(point1), icon=folium.Icon(color='red')).add_to(geomap)
folium.Marker(point2[::-1], popup=str(point2), icon=folium.Icon(color='red')).add_to(geomap)

# Save the interactive plot as a map
output_name = 'example_plot_between_{:.2f}-{:.2f}_and_{:.2f}-{:.2f}.html'.format(point1[0],point1[1],point2[0],point2[1])
geomap.save(output_name)
print('Output saved to: {}\n'.format(output_name))
