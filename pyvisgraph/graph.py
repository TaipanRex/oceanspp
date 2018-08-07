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
from collections import defaultdict
EPSILON = 1e-6


def set_epsilon(value):
    """
    set EPSILON value
    """

    EPSILON = float(value)
    return EPSILON


class Point(object):
    __slots__ = ('x', 'y', 'polygon_id')

    def __init__(self, x, y, polygon_id=-1):
        self.x = float(x)
        self.y = float(y)
        self.polygon_id = polygon_id

    def __eq__(self, point):
        return point and abs(self.x - point.x) + abs(self.y - point.y) < EPSILON

    def __ne__(self, point):
        return not self.__eq__(point)

    def __lt__(self, point):
        """ This is only needed for shortest path calculations where heapq is
            used. When there are two points of equal distance, heapq will
            instead evaluate the Points, which doesnt work in Python 3 and
            throw a TypeError."""
        return hash(self) < hash(point)

    def __str__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    def __hash__(self):
        return self.x.__hash__() ^ self.y.__hash__()

    def __repr__(self):
        return "Point(%.2f, %.2f)" % (self.x, self.y)


class Edge(object):
    __slots__ = ('p1', 'p2')

    def __init__(self, point1, point2):
        self.p1 = point1
        self.p2 = point2

    def get_adjacent(self, point):
        if point == self.p1:
            return self.p2
        return self.p1

    def __contains__(self, point):
        return self.p1 == point or self.p2 == point

    def __eq__(self, edge):
        if self.p1 == edge.p1 and self.p2 == edge.p2:
            return True
        if self.p1 == edge.p2 and self.p2 == edge.p1:
            return True
        return False

    def __ne__(self, edge):
        return not self.__eq__(edge)

    def __str__(self):
        return "({}, {})".format(self.p1, self.p2)

    def __repr__(self):
        return "Edge({!r}, {!r})".format(self.p1, self.p2)

    def __hash__(self):
        return self.p1.__hash__() ^ self.p2.__hash__()


class Graph(object):
    """
    A Graph is represented by a dict where the keys are Points in the Graph
    and the dict values are sets containing Edges incident on each Point.
    A separate set *edges* contains all Edges in the graph.

    The input must be a list of polygons, where each polygon is a list of
    in-order (clockwise or counter clockwise) Points. If only one polygon,
    it must still be a list in a list, i.e. [[Point(0,0), Point(2,0),
    Point(2,1)]].

    *polygons* dictionary: key is a integer polygon ID and values are the
    edges that make up the polygon. Note only polygons with 3 or more Points
    will be classified as a polygon. Non-polygons like just one Point will be
    given a polygon ID of -1 and not maintained in the dict.
    """

    def __init__(self, polygons):
        self.graph = defaultdict(set)
        self.edges = set()
        self.polygons = defaultdict(set)
        pid = 0
        for polygon in polygons:
            if polygon[0] == polygon[-1] and len(polygon) > 1:
                polygon.pop()
            for i, point in enumerate(polygon):
                sibling_point = polygon[(i + 1) % len(polygon)]
                edge = Edge(point, sibling_point)
                if len(polygon) > 2:
                    point.polygon_id = pid
                    sibling_point.polygon_id = pid
                    self.polygons[pid].add(edge)
                self.add_edge(edge)
            if len(polygon) > 2:
                pid += 1

    def get_adjacent_points(self, point):
        return [edge.get_adjacent(point) for edge in self[point]]

    def get_points(self):
        return list(self.graph)

    def get_edges(self):
        return self.edges

    def add_edge(self, edge):
        self.graph[edge.p1].add(edge)
        self.graph[edge.p2].add(edge)
        self.edges.add(edge)

    def __contains__(self, item):
        if isinstance(item, Point):
            return item in self.graph
        if isinstance(item, Edge):
            return item in self.edges
        return False

    def __getitem__(self, point):
        if point in self.graph:
            return self.graph[point]
        return set()

    def __str__(self):
        res = ""
        for point in self.graph:
            res += "\n" + str(point) + ": "
            for edge in self.graph[point]:
                res += str(edge)
        return res

    def __repr__(self):
        return self.__str__()
