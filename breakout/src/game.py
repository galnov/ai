"""
The game: many Breakout boards at once (NumPy), and the screen the network sees.

Every board has a paddle, a ball, 5 x 8 bricks and 3 lives. One decision (1/15 s) is 4 physics steps:
  1. the paddle moves left, right or stays (the action)
  2. the ball moves; it bounces off the side walls and the top wall
  3. a brick under the ball breaks (+1); the ball bounces up/down if it came from the row above or below,
     sideways if it came from the next brick in the same row. The ball gets 2% faster
  4. the paddle: where the ball lands on it decides the angle it goes back up (the edge: 60 degrees,
     the middle: straight up) - this is how a player aims
  5. a ball that falls below the paddle is lost (-1) and a new one is served; after 3 lost balls the game ends
When all the bricks are gone, a new wall of bricks appears (the next level).

The screen: 48 x 48 gray pixels (0 = black). Bricks 110-230 (a different gray in each row), paddle and ball 255.
The network gets the last 2 screens.
"""
import numpy as np

from . import config as C

S = C.SCREEN
ROW_GRAY = np.array([230, 200, 170, 140, 110], np.uint8)[: C.BRICK_ROWS]

# the pixels of one row of bricks: (BRICK_H, S) with 1-pixel gaps between bricks and below them
_cell = np.zeros((C.BRICK_H, C.BRICK_W), bool)
_cell[: C.BRICK_H - 1, : C.BRICK_W - 1] = True
CELL_MASK = np.tile(_cell, (1, C.BRICK_COLS))


def render(bricks, px, bx, by, alive=None):
    """bricks (n, ROWS, COLS) bool, paddle center px (n,), ball (bx, by) (n,) -> screens (n, S, S) uint8.
    alive (n,) bool: draw the ball (a board whose game is over shows no ball)."""
    n = len(px)
    img = np.zeros((n, S, S), np.uint8)
    # bricks
    full = np.repeat(np.repeat(bricks, C.BRICK_H, axis=1), C.BRICK_W, axis=2)        # (n, ROWS*H, S)
    gray = np.repeat(ROW_GRAY, C.BRICK_H)[None, :, None]
    top = C.BRICK_TOP
    img[:, top:top + C.BRICK_ROWS * C.BRICK_H] = np.where(full & np.tile(CELL_MASK, (C.BRICK_ROWS, 1)), gray, 0)
    # paddle: 2 pixels tall
    x0 = np.floor(px - C.PADDLE_W / 2).astype(int)
    cols = np.clip(x0[:, None] + np.arange(C.PADDLE_W), 0, S - 1)
    rows = np.arange(n)[:, None]
    img[rows, C.PADDLE_Y, cols] = 255
    img[rows, C.PADDLE_Y + 1, cols] = 255
    # ball: 2 x 2 pixels around its center
    bx0, by0 = np.floor(bx - 1).astype(int), np.floor(by - 1).astype(int)
    for dy in (0, 1):
        for dx in (0, 1):
            x, y = bx0 + dx, by0 + dy
            ok = (x >= 0) & (x < S) & (y >= 0) & (y < S)
            if alive is not None:
                ok &= alive
            img[np.flatnonzero(ok), y[ok], x[ok]] = 255
    return img


class Breakout:
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.px = np.zeros(n)
        self.bx, self.by, self.vx, self.vy, self.speed = (np.zeros(n) for _ in range(5))
        self.bricks = np.ones((n, C.BRICK_ROWS, C.BRICK_COLS), bool)
        self.lives = np.zeros(n, int)
        self.score = np.zeros(n, int)
        self.steps = np.zeros(n, int)
        self.frames = np.zeros((n, C.FRAMES, S, S), np.uint8)
        self.reset(np.arange(n))

    def reset(self, idx):
        self.px[idx] = S / 2
        self.bricks[idx] = True
        self.lives[idx] = C.LIVES
        self.score[idx] = 0
        self.steps[idx] = 0
        self._serve(idx)
        scr = render(self.bricks[idx], self.px[idx], self.bx[idx], self.by[idx])
        self.frames[idx] = scr[:, None]

    def _serve(self, idx):
        """A new ball leaves from the paddle's center, up, at a random angle."""
        k = len(idx)
        ang = np.radians(self.rng.uniform(*C.SERVE_ANGLE, k)) * self.rng.choice([-1, 1], k)
        self.speed[idx] = C.BALL_SPEED
        self.bx[idx] = self.px[idx]
        self.by[idx] = C.PADDLE_Y - 1.0
        self.vx[idx] = C.BALL_SPEED * np.sin(ang)
        self.vy[idx] = -C.BALL_SPEED * np.cos(ang)

    def observe(self):
        """(n, FRAMES, S, S) uint8: the last screens, the oldest first."""
        return self.frames

    def step(self, actions):
        """actions (n,) in {0 stay, 1 left, 2 right}.
        Returns (reward, lost (a ball was lost this step), done (the game ended; the board starts a new game))."""
        move = np.array([0.0, -1.0, 1.0])[actions] * C.PADDLE_SPEED
        h = C.DT / C.SUBSTEPS
        reward = np.zeros(self.n)
        lost = np.zeros(self.n, bool)
        broke = np.zeros(self.n, int)
        half = C.PADDLE_W / 2
        for _ in range(C.SUBSTEPS):
            self.px = np.clip(self.px + move * h, half, S - half)
            ox, oy = self.bx.copy(), self.by.copy()
            self.bx += self.vx * h
            self.by += self.vy * h
            # walls (the ball's center stays 1 pixel from the edge)
            left, right, top = self.bx < 1, self.bx > S - 1, self.by < 1
            self.bx = np.where(left, 2 - self.bx, np.where(right, 2 * (S - 1) - self.bx, self.bx))
            self.vx = np.where(left, np.abs(self.vx), np.where(right, -np.abs(self.vx), self.vx))
            self.by = np.where(top, 2 - self.by, self.by)
            self.vy = np.where(top, np.abs(self.vy), self.vy)
            # bricks
            row = np.floor((self.by - C.BRICK_TOP) / C.BRICK_H).astype(int)
            col = np.floor(self.bx / C.BRICK_W).astype(int)
            inside = (row >= 0) & (row < C.BRICK_ROWS) & (col >= 0) & (col < C.BRICK_COLS)
            i = np.flatnonzero(inside)
            hit = np.zeros(self.n, bool)
            hit[i] = self.bricks[i, row[i], col[i]]
            if hit.any():
                j = np.flatnonzero(hit)
                self.bricks[j, row[j], col[j]] = False
                reward[j] += C.BRICK_REWARD
                broke[j] += 1
                orow = np.floor((oy[j] - C.BRICK_TOP) / C.BRICK_H).astype(int)
                vertical = orow != row[j]                     # came from the row above or below
                self.vy[j] = np.where(vertical, -self.vy[j], self.vy[j])
                self.vx[j] = np.where(vertical, self.vx[j], -self.vx[j])
                self.by[j] = np.where(vertical, oy[j], self.by[j])       # step back out of the brick
                self.bx[j] = np.where(vertical, self.bx[j], ox[j])
                self.speed[j] = np.minimum(self.speed[j] * C.SPEEDUP, C.BALL_SPEED_MAX)
                self._fix_speed(j)
            # the paddle: the ball's bottom edge (by + 1) crosses the paddle's top edge
            land = C.PADDLE_Y - 1
            paddle = (self.vy > 0) & (oy <= land) & (self.by > land) & (np.abs(self.bx - self.px) <= half + 1)
            if paddle.any():
                j = np.flatnonzero(paddle)
                off = np.clip((self.bx[j] - self.px[j]) / (half + 1), -1, 1)
                ang = np.radians(off * C.MAX_ANGLE)
                self.vx[j] = self.speed[j] * np.sin(ang)
                self.vy[j] = -self.speed[j] * np.cos(ang)
                self.by[j] = 2 * land - self.by[j]
            # a lost ball
            miss = self.by > S + 1
            if miss.any():
                j = np.flatnonzero(miss)
                lost[j] = True
                reward[j] += C.MISS_REWARD
                self.lives[j] -= 1
                self._serve(j)
            # a cleared wall: the next level
            clear = ~self.bricks.any(axis=(1, 2))
            if clear.any():
                self.bricks[clear] = True
        self.steps += 1
        self.score += broke
        done = (self.lives <= 0) | (self.steps >= C.MAX_STEPS)
        scr = render(self.bricks, self.px, self.bx, self.by)
        self.frames = np.concatenate([self.frames[:, 1:], scr[:, None]], axis=1)
        if done.any():
            self.reset(np.flatnonzero(done))
        return reward, lost, done

    def _fix_speed(self, j):
        """Keep the ball at its speed, and never too flat (a ball going sideways forever is no fun)."""
        sp = self.speed[j]
        n = np.hypot(self.vx[j], self.vy[j]) + 1e-9
        vx, vy = self.vx[j] / n * sp, self.vy[j] / n * sp
        flat = np.abs(vy) < C.MIN_VY * sp
        vy = np.where(flat, np.sign(vy + 1e-9) * C.MIN_VY * sp, vy)
        vx = np.where(flat, np.sign(vx + 1e-9) * np.sqrt(np.maximum(sp ** 2 - vy ** 2, 0)), vx)
        self.vx[j], self.vy[j] = vx, vy


def tracker(env, skill=1.0, rng=None):
    """A simple scripted player (for tests): move the paddle toward the ball."""
    d = env.bx - env.px
    a = np.where(d < -1, 1, np.where(d > 1, 2, 0))
    if skill < 1:
        rnd = (rng or np.random).random(env.n) > skill
        a = np.where(rnd, (rng or np.random).integers(0, 3, env.n), a)
    return a
