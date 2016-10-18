
# Copyright (C) 2016 to Amir Livne Bar-on

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Usage:
#   modify the name of the graph in the last line
#   no command line arguments are processed


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math


def orthogonal_projection(v, subspace):
    # the subspace is given by orthonormal basis
    return np.sum([u * np.sum(u*v) for u in subspace], axis=0)

def null_space(a, rtol=1e-8):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:]

def random_direction(subspace):
    coefs = np.random.normal(size=len(subspace))
    coefs /= np.sum(coefs**2) ** 0.5
    return np.sum([c*v for c,v in zip(coefs, subspace)], axis=0)

def minimize_newton(x0, functional, gradient, apply_hessian, tol=1e-6, DEBUG=True):
    value = functional(x0)
    if DEBUG:
        # print out the distance from unit embedding every once in a while
        # in my experience, values consistently >0.1 indicate numerical errors dominate
        if np.random.uniform() < 0.01:
            print(value)
    while True:
        grad = gradient(x0)
        norm2 = np.sum(grad**2)
        if norm2 < tol:
            return x0
        coef = -norm2 / np.sum(grad * apply_hessian(x0, grad))
        x1 = x0 + coef * grad
        value1 = functional(x1)
        if value1 > value - tol:
            return x0
        x0 = x1
        value = value1


class EmbeddedGraph(object):

    def __init__(self, n_vertices, edges, positions):
        self._n_vertices = n_vertices
        self._edges = edges
        self._positions = np.array([float(coord) for pos in positions for coord in pos])
        self._fix_positions()

    def allowed_motions_basis(self):
        # returns orthonormal basis

        n_edges = len(self._edges)
        mat = np.zeros(shape=[n_edges + 3, self._n_vertices * 2])
        for i_edge, edge in enumerate(self._edges):
            # only allow motions not changing the length of this edge
            for axis in range(2):
                for direction in range(2):
                    end = edge[direction]*2 + axis
                    start = edge[1-direction]*2 + axis
                    mat[i_edge, end] = self._positions[end] - self._positions[start]
        for axis in range(2):
            # forbid changing the center of gravity
            mat[n_edges + axis, :] = np.array([axis, 1-axis] * self._n_vertices)
        # don't allow any total angular momentum
        positions_perp = np.ravel(np.column_stack([self._positions[1::2], -self._positions[0::2]]))
        mat[n_edges + 2, :] = positions_perp

        # find directions satisfying constraints
        return null_space(mat)

    @staticmethod
    def normalize_motion(motion):
        # the norm is maximum of the speeds of the points
        norm2 = np.max(motion[0::2]**2 + motion[1::2]**2)
        return motion * (norm2**-0.5)

    def translate(self, motion):
        self._positions += motion
        self._fix_positions()

    def _fix_positions(self):
        # fix the positions to satisfy constraints of unit length and centering
        def functional(x):
            error = 0.0
            for edge in self._edges:
                v1, v2 = edge
                distance2 = (x[v1*2] - x[v2*2])**2 + (x[v1*2+1] - x[v2*2+1])**2
                error += (distance2 - 1.0)**2
            for axis in range(2):
                error += np.sum(x[axis::2]) ** 2
            return error
        def gradient(x):
            grad = np.zeros(self._n_vertices * 2)
            for edge in self._edges:
                v1, v2 = edge
                distance2 = (x[v1*2] - x[v2*2])**2 + (x[v1*2+1] - x[v2*2+1])**2
                factor = 4.0 * (distance2 - 1.0)
                for axis in range(2):
                    for direction in range(2):
                        end = edge[direction]*2 + axis
                        start = edge[1-direction]*2 + axis
                        grad[end] += factor * (x[end] - x[start])
            for axis in range(2):
                factor = 2.0 * np.sum(x[axis::2])
                for i in range(self._n_vertices):
                    grad[i*2 + axis] += factor
            return grad
        def apply_hessian(x, v):
            res = np.zeros(self._n_vertices * 2)
            for edge in self._edges:
                # TODO: document why this is the correct formula
                for axis in range(2):
                    for direction in range(2):
                        d1 = x[edge[direction]*2+axis] - x[edge[1-direction]*2+axis]
                        d2 = x[edge[direction]*2+1-axis] - x[edge[1-direction]*2+1-axis]
                        a = 12 * d1**2 + 4 * d2**2 - 4
                        b = 8 * d1 * d2
                        res[edge[direction]*2+axis] += a * (v[edge[direction]*2+axis] - v[edge[1-direction]*2+axis])
                        res[edge[direction]*2+axis] += b * (v[edge[direction]*2+1-axis] - edge[1-direction]*2+1-axis)
            # this part is 2s on the diagonal and 1s everywhere else
            res += v
            for axis in range(2):
                res[axis::2] += np.sum(v[axis::2])
            return res
        self._positions = minimize_newton(self._positions, functional, gradient, apply_hessian)

    def draw(self, plt_edges, plt_vertices):
        all_xs, all_ys = [], []
        for edge in self._edges:
            xs, ys = [[self._positions[node*2+axis] for node in edge] for axis in range(2)]
            all_xs += xs + [None]
            all_ys += ys + [None]
        plt_edges.set_data(all_xs, all_ys)
        plt_vertices.set_data(self._positions[0::2], self._positions[1::2])


class FlexingAnimation(object):
    def __init__(self, graph, window_size, max_coord):
        self._graph = graph

        self._fig = plt.figure(figsize=(window_size, window_size))
        self._ax = self._fig.add_axes([0, 0, 1, 1], frameon=False)
        self._ax.set_xlim(-max_coord, max_coord)
        self._ax.set_xticks([])
        self._ax.set_ylim(-max_coord, max_coord)
        self._ax.set_yticks([])

        self._plt_edges, = self._ax.plot([], [], 'k')
        self._plt_vertices, = self._ax.plot([], [], 'ob')

        # choose random initial motion direction
        self._motion = EmbeddedGraph.normalize_motion(random_direction(self._graph.allowed_motions_basis()))

    def _update(self, frame_number):
        self._graph.draw(self._plt_edges, self._plt_vertices)
        self._graph.translate(self._motion * 1e-2)
        # move motion along geodesic of configuration space
        self._motion = EmbeddedGraph.normalize_motion(orthogonal_projection(self._motion, self._graph.allowed_motions_basis()))

    def run(self):
        anim = animation.FuncAnimation(self._fig, self._update, interval=20)
        plt.show()


# 1 degree of freedom
square = EmbeddedGraph(4, [(0,1), (1,2), (2,3), (3,0)],
                       [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])

# surprising variety
pentagon = EmbeddedGraph(5, [(0,1), (1,2), (2,3), (3,4), (4,0)],
                         [(math.cos(i*math.pi*0.4), math.sin(i*math.pi*0.4)) for i in range(5)])

# 3x3 grid of squares
grid3 = EmbeddedGraph(9, [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
                          (0,3), (3,6), (1,4), (4,7), (2,5), (5,8)],
                      [(-2, 0), (-1, 1), (0, 2),
                       (-1, -1), (0, 0), (1, 1),
                       (0, -2), (1, -1), (2, 0)])

# a 3-d cube projected to the plane
cube = EmbeddedGraph(8, [(0,1), (2,3), (4,5), (6,7),
                         (0,2), (1,3), (4,6), (5,7),
                         (0,4), (1,5), (2,6), (3,7)],
                     [(x*0.5+z*8**-0.5, y*0.5-z*8**-0.5) for x in [1,-1] for y in [1,-1] for z in [1,-1]])

# the Peterson graph
peterson = EmbeddedGraph(
    10,
    [(0,2), (2,4), (4,1), (1,3), (3,0),
     (5,6), (6,7), (7,8), (8,9), (9,5),
     (0,5), (1,6), (2,7), (3,8), (4,9)],
    [(math.cos(i*math.pi*0.4)*0.5257, math.sin(i*math.pi*0.4)*0.5257) for i in range(5)] +
    [(math.cos((i+1.25)*math.pi*0.4)*0.8507, math.sin((i+1.25)*math.pi*0.4)*0.8507) for i in range(5)]
)

# the McGee graph, has nice symmetrical animation by Greg Egan elsewhere
circle_in = [(math.cos(i*math.pi*0.25)*0.5, math.sin(i*math.pi*0.25)*0.5) for i in range(8)]
x1, y1, x2, y2 = 1.35286334, 0.31641005, 0.99317177, 0.61666121
circle_out = [(x1, y1), (x2, y2), (y2, x2), (y1, x1)]
for i in range(3):
    circle_out.extend([(-y, x) for (x,y) in circle_out[-4:]])
mcgee = EmbeddedGraph(
    24,
    [(0, 1), (0, 8), (0, 9), (1, 10), (1, 11), (2, 3), (2, 12), (2, 14), (3, 13), (3, 15), (4, 7), (4, 16), (4, 20), (5, 6), (5, 17), (5, 21), (6, 18), (6, 22), (7, 19), (7, 23), (8, 16), (8, 22), (9, 17), (9, 23), (10, 18), (10, 20), (11, 19), (11, 21), (12, 17), (12, 20), (13, 16), (13, 21), (14, 19), (14, 22), (15, 18), (15, 23)],
    [circle_in[i] for i in [2,6,0,4,1,3,7,5]] + [circle_out[i] for i in [1,6,14,9,2,5,13,10,3,4,12,11,0,7,15,8]]
)


# change the name of the graph here to see flows through their unit-distance embeddings configuration space
FlexingAnimation(mcgee, 6, 2).run()
