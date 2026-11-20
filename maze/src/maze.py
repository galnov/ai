"""
3D mazes on a grid of cubic cells (4 m each).

A maze with n x n x n "rooms" lives on a grid of (2n+1)^3 cells: rooms are at the odd coordinates, and the
cells between two rooms are either wall or a passage. All the other cells are solid rock.

Generation: depth-first search ("recursive backtracker") from the first room, choosing a random unvisited
neighbor room in any of the 6 directions (also up and down) and carving the passage to it. The result is
a "perfect" maze: every cell can be reached, and there is exactly one way between any two cells,
with long winding corridors and dead ends in 3D.

The exit is the room farthest from the start. A breadth-first search (BFS) from the exit then gives every
empty cell its distance to the exit and the direction to go - this is the "navigation beacon" the
ship receives, and it is also used to measure progress for the reward.
"""
from collections import deque

import numpy as np

from . import config as C

DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


class Maze:
    def __init__(self, seed, rooms=C.MAZE_ROOMS):
        self.seed, self.rooms = int(seed), int(rooms)
        G = 2 * rooms + 1
        self.G = G
        rng = np.random.default_rng(self.seed)
        solid = np.ones((G, G, G), bool)
        visited = np.zeros((rooms,) * 3, bool)
        stack = [(0, 0, 0)]
        visited[0, 0, 0] = True
        solid[1, 1, 1] = False
        while stack:
            room = stack[-1]
            options = []
            for d in DIRS[rng.permutation(6)]:
                nb = (room[0] + d[0], room[1] + d[1], room[2] + d[2])
                if all(0 <= c < rooms for c in nb) and not visited[nb]:
                    options.append((nb, d))
            if not options:
                stack.pop()
                continue
            nb, d = options[0]
            visited[nb] = True
            a = 2 * np.array(room) + 1
            solid[tuple(a + d)] = False                 # the passage
            solid[tuple(a + 2 * d)] = False             # the next room
            stack.append(nb)
        self.solid = solid
        self.start = (1, 1, 1)

        # the exit: the room farthest from the start
        d_start = self._bfs(self.start)
        rooms_mask = np.zeros_like(solid)
        rooms_mask[1::2, 1::2, 1::2] = True
        far = np.where(rooms_mask & (d_start >= 0), d_start, -1)
        self.exit = tuple(int(i) for i in np.unravel_index(far.argmax(), far.shape))

        # distance to the exit and the direction to go, for every empty cell
        self.dist = self._bfs(self.exit)
        self.next_dir = np.zeros((G, G, G, 3), np.int64)
        for cell in zip(*np.nonzero(self.dist > 0)):
            for d in DIRS:
                nb = tuple(np.array(cell) + d)
                if self.dist[nb] == self.dist[cell] - 1:
                    self.next_dir[cell] = d
                    break
        self.path = self.route(self.start)

    def _bfs(self, src):
        dist = np.full(self.solid.shape, -1, np.int64)
        dist[src] = 0
        q = deque([src])
        G = self.G
        while q:
            c = q.popleft()
            for d in DIRS:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if 0 <= nb[0] < G and 0 <= nb[1] < G and 0 <= nb[2] < G and not self.solid[nb] and dist[nb] < 0:
                    dist[nb] = dist[c] + 1
                    q.append(nb)
        return dist

    def route(self, cell):
        """The cells from `cell` to the exit, following the beacon."""
        out = [tuple(int(v) for v in cell)]
        while out[-1] != self.exit:
            out.append(tuple(int(v) for v in np.array(out[-1]) + self.next_dir[out[-1]]))
        return out

    def center(self, cell):
        return (np.asarray(cell) + 0.5) * C.CELL

    def start_direction(self):
        return self.next_dir[self.start].astype(float)

    def path_length(self):
        return (len(self.path) - 1) * C.CELL

    def to_json(self):
        return {
            "seed": self.seed, "rooms": self.rooms, "G": self.G, "cell": C.CELL,
            "solid": "".join("1" if s else "0" for s in self.solid.ravel()),   # index = (x * G + y) * G + z
            "start": list(self.start), "exit": list(self.exit),
            "path": [list(c) for c in self.path],
            "next": self.next_dir.reshape(-1, 3)[~self.solid.ravel()].tolist(),
        }
