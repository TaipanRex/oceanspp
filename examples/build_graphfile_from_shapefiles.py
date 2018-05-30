#!/usr/bin/env python

import pyvisgraph as vg
import shapefile

# In this example a graph will be created from the GSHHS shorelines available at:
# https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html
# 1. Download the shorelines as shape files and select the wanted resolution and level (Crude, L1). 
# 2. Copy all the related files (shx, shp, prj, dbf) to the folder with this script.

input_shapefile  = shapefile.Reader('GSHHS_c_L1')
output_graphfile = 'GSHHS_c_L1.graph'
workers          = 4 # Number of cores on host

# Parse the shape file of the shorelines
shapes = input_shapefile.shapes()
print('The shapefile contains {} shapes.'.format(len(shapes)))

# Create a list of polygons
polygons = []
for shape in shapes:
    shapelist = []
    for point in shape.points:
        shapelist.append(vg.Point(point[0], point[1]))
    polygons.append(shapelist)

# Start building the graph file
graph = vg.VisGraph()
print('Starting building graph')
graph.build(polygons, workers=workers, status=True)
print('Finished building graph')

# Save the output to a graphfile
graph.save(output_graphfile)
print('Saved graph to file: {}'.format(output_graphfile))
