# Pyvisgraph - Python Visibility Graph

[![MIT License](https://img.shields.io/github/license/taipanrex/pyvisgraph.svg?style=flat)](/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/pyvisgraph.svg?style=flat)](https://pypi.python.org/pypi/pyvisgraph)

Given a set of simple obstacle polygons, build a visibility graph and find
the shortest path between two points.

![Figure 1](docs/images/graph.png)

Pyvisgraph is a MIT-licensed Python package for building visibility graphs from
a list of simple obstacle polygons. The visibility graph algorithm (D.T. Lee)
runs in O(n^2 log n) time. The shortest path is found using Djikstra's
algorithm.

To see how visibility graphs work interactively, take a look at the
[Visibility Graph Simulator](https://github.com/TaipanRex/visgraph_simulator)
built with Pyvisgraph.

## Installing Pyvisgraph
```
$ pip install pyvisgraph
```
Pyvisgraph supports Python 2 and 3.

## Usage
Here is an example of building a visibility graph given a list of
simple polygons:
```
>>> import pyvisgraph as vg
>>> polys = [[vg.Point(0.0,1.0), vg.Point(3.0,1.0), vg.Point(1.5,4.0)],
>>>          [vg.Point(4.0,4.0), vg.Point(7.0,4.0), vg.Point(5.5,8.0)]]
>>> g = vg.VisGraph()
>>> g.build(polys)
>>> shortest = g.shortest_path(vg.Point(1.5,0.0), vg.Point(4.0, 6.0))
>>> print shortest
[Point(1.50, 0.00), Point(3.00, 1.00), Point(4.00, 6.00)]
```
Once the visibility graph is built, it can be saved and subsequently loaded.
This is useful for large graphs where build time is long. `pickle` is used
for saving and loading.
```
>>> g.save('graph.pk1')
>>> g2 = VisGraph()
>>> g2.load('graph.pk1')
```
For obstacles with a large number of points, Pyvisgraph can take advantage of
processors with multiple cores using the `multiprocessing` module. Simply
add the number of workers (processes) to the `build` method:
```
>>> g.build(polys, workers=4)
```
Pyvisgraph also has some useful helper functions:
* `g.update([list of Points])`: Updates the visibility graph
  by checking visibility of each `Point` in the list.
* `g.point_in_polygon(Point)`: Check if `Point` is in the interior of any of
  the obstacle polygons. Returns the polygon_id of said polygon, -1 if not
  inside any polygon.
* `g.closest_point(Point, polygon_id)`: Return the closest point outside
  polygon with polygon_id from Point.

For further examples, please look at the scripts provided in the `examples`
folder.

## Example & performance
This example uses a shapefile representing world shorelines as obstacles.
Two vessels were picked randomly and their current location found
using [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system). Red
lines are laden voyage legs (carrying cargo) and dotted blue lines are ballast
legs (no cargo, moving to load destination). Pyvisgraph has the following
performance on a Microsoft Surface Pro 3 (Intel i7-4650U @ 1.7Ghz, 8GB DDR3
RAM), where time is in seconds:
```
Shoreline obstacle graph [points: 4335 edges: 4335]
Using 4 worker processes...
Time to create visibility graph: 554.683238029
Visibility graph edges: 118532
Time to update visgraph & find shortest path: 1.09287905693
Shorest path nodes: 19
Time to find shortest path between existing points: 0.508340835571
```
For one vessel the origin and destination Points were not part of the built
visibility graph and had to first be computed. For the second vessel, these
Points were first added to the visibility graph using `update`, then finding
the shortest path is faster. Using Matplotlib basemap to visualize the routes:
![Figure 2](docs/images/example.png)

For more information about the implementation, see these series of articles:
* [Distance Tables Part 1: Defining the Problem](https://taipanrex.github.io/2016/09/17/Distance-Tables-Part-1-Defining-the-Problem.html)
* [Distance Tables Part 2: Lee's Visibility Graph Algorithm](https://taipanrex.github.io/2016/10/19/Distance-Tables-Part-2-Lees-Visibility-Graph-Algorithm.html)
* More to come...
