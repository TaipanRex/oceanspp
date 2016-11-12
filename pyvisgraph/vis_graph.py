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
from timeit import default_timer
from sys import stdout, version_info
from multiprocessing import Pool
from collections import defaultdict

from pyvisgraph.graph import Graph, Edge
from pyvisgraph.shortest_path import shortest_path
from pyvisgraph.visible_vertices import visible_vertices, point_in_polygon
from pyvisgraph.visible_vertices import closest_point

PYTHON3 = version_info[0] == 3
if PYTHON3:
    xrange = range
    import pickle
else:
    import cPickle as pickle


class VisGraph(object):

    def __init__(self):
        self.graph = None
        self.visgraph = None

    def load(self, filename):
        """Load obstacle graph and visibility graph. """
        with open(filename, 'rb') as load:
            self.graph, self.visgraph = pickle.load(load)

    def save(self, filename):
        """Save obstacle graph and visibility graph. """
        with open(filename, 'wb') as output:
            pickle.dump((self.graph, self.visgraph), output, -1)

    def build(self, input, workers=1, status=False):
        """Build visibility graph based on a list of polygons.

        The input must be a list of polygons, where each polygon is a list of
        in-order (clockwise or counter clockwise) Points. It only one polygon,
        it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
        Point(2,1)]].
        Take advantage of processors with multiple cores by setting workers to
        the number of subprocesses you want. Defaults to 1, i.e. no subprocess
        will be started.
        Set status to True to see progress information for each subprocess:
        [Points done][Points remaining][average time per Point].
        """

        self.graph = Graph(input)
        self.visgraph = Graph([])
        if status: print(" " + "[Done][Rem.][Avg t] " * workers)

        if workers == 1:
            for edge in _vis_graph(self.graph, self.graph.get_points(), 0, status):
                self.visgraph.add_edge(edge)
            if status: print("")
            return None

        points = self.graph.get_points()
        batch_size = int(len(points) / workers)
        batches = [(self.graph, points[i:i + batch_size], i/batch_size, status)
                   for i in xrange(0, len(points), batch_size)]
        pool = Pool(workers)
        results = pool.map_async(_vis_graph_wrapper, batches)
        try:
            for result in results.get():
                for edge in result:
                    self.visgraph.add_edge(edge)
        except KeyboardInterrupt:
            if status: print("")
            raise
        if status: print("")

    def update(self, points, origin=None, destination=None):
        """Update visgraph by checking visibility of Points in list points."""

        for p in points:
            for v in visible_vertices(p, self.graph, origin=origin,
                                      destination=destination):
                self.visgraph.add_edge(Edge(p, v))

    def shortest_path(self, origin, destination):
        """Find and return shortest path between origin and destination.

        Will return in-order list of Points of the shortest path found. If
        origin or destination are not in the visibility graph, the graph will
        first be updated.
        Note that the visgraph will be updated permanently with the origin
        and destination visibility if not computed before.
        """

        origin_exists = origin in self.visgraph
        dest_exists = destination in self.visgraph
        if origin_exists and dest_exists:
            return shortest_path(self.visgraph, origin, destination)
        orgn = None if origin_exists else origin
        dest = None if dest_exists else destination
        if not origin_exists: self.update([origin], destination=dest)
        if not dest_exists: self.update([destination], origin=orgn)
        return shortest_path(self.visgraph, origin, destination)

    def point_in_polygon(self, point):
        """Return polygon_id if point in a polygon, -1 otherwise."""

        return point_in_polygon(point, self.graph)

    def closest_point(self, point, polygon_id, length=0.001):
        """Return closest Point outside polygon from point.

        Note method assumes point is inside the polygon, no check is
        performed.
        """

        return closest_point(point, self.graph, polygon_id, length)

    def _effective_area(self, p1, p2, p3):
        """Calculate the area formed by three given points"""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        return abs(x1*y2 + x2*y3 + x3*y1 - (x1*y3 + x3*y2 + x2*y1))/2.0

    def simplify_polygon(self, polygons, target_resolution):
        """Simplify a list of polygons via Visvalingam's algorithm"""
        new_polygons = []
        for poly in polygons:
            poly_builder = {}

            orig_area = 0.0
            for i, point in enumerate(poly):
                prev_point = poly[(i - 1) % len(poly)]
                next_point = poly[(i + 1) % len(poly)]
                area = self._effective_area(point, prev_point, next_point)

                orig_area += area
                poly_builder[point] = (area, prev_point, next_point)

            current_area = orig_area
            current_res = 1.0
            while (current_res > target_resolution) and len(poly_builder) > 3:
                least_area_point = min(poly_builder,
                                       key=lambda p: poly_builder[p][0])

                a, prev_point, next_point = poly_builder[least_area_point]

                a1, prev_1, prev_2 = (i if i is not least_area_point
                                      else next_point
                                      for i in poly_builder[prev_point])
                a2, next_1, next_2 = (i if i is not least_area_point
                                      else prev_point
                                      for i in poly_builder[next_point])

                del poly_builder[least_area_point]
                current_area -= a

                a1 = self._effective_area(prev_point, prev_1, prev_2)
                poly_builder[prev_point] = (a1, prev_1, prev_2)

                a2 = self._effective_area(next_point, next_1, next_2)
                poly_builder[next_point] = (a2, next_1, next_2)
                current_res = current_area/orig_area

            p, val = poly_builder.popitem()
            a, prev_p, next_p = val

            final_poly = [p]
            while (next_p is not final_poly[0]):
                final_poly.append(next_p)
                next_p = poly_builder[next_p][2]

            new_polygons.append(final_poly)
        return new_polygons


def _vis_graph_wrapper(args):
    try:
        return _vis_graph(*args)
    except KeyboardInterrupt:
        pass


def _vis_graph(graph, points, worker, status):
    total_points = len(points)
    visible_edges = []
    if status:
        t0 = default_timer()
        points_done = 0
    for p1 in points:
        for p2 in visible_vertices(p1, graph, scan='half'):
            visible_edges.append(Edge(p1, p2))
        if status:
            points_done += 1
            avg_time = round((default_timer() - t0) / points_done, 3)
            time_stat = (points_done, total_points-points_done, avg_time)
            status = '\r\033[' + str(21*worker) + 'C[{:4}][{:4}][{:5.3f}] \r'
            stdout.write(status.format(*time_stat))
            stdout.flush()
    return visible_edges
