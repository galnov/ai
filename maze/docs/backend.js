// =====================================================================================
//  The "server" of the static version: everything app/server.py did, done in the browser.
//    info  - the pilots (the exported networks + the autopilot) and the ship's physics
//    maze  - a new random 3D maze (the same algorithm as src/maze.py)
//    fly   - the AI's flight through a maze: the ship's sensors (src/ship.py), the pilot's decisions
//            and the physics (ship.js), recorded as frames for playback
//  The networks come from docs/model (python -m src.export_web) and run in mlp.js.
//  The mazes are made with a JavaScript random generator, so maze #123 here is not the same
//  maze as seed 123 in Python - but it is made by exactly the same rules.
// =====================================================================================
(function (root) {
  const { MazeData, Ship, V } = root.MazeSim;

  // the same constants as src/config.py (keep them in sync)
  const C = {
    CELL: 4.0, MAZE_ROOMS: 4,
    SHIP_RADIUS: 0.6, V_MIN: 2.0, V_MAX: 11.0, ACCEL_K: 2.5, PITCH_RATE: 2.2, YAW_RATE: 2.2, DT: 0.1, SUBSTEPS: 2,
    RAYS_X: 9, RAYS_Y: 9, FOV: 100.0, RAY_RANGE: 24.0, BEACON_SCALE: 8.0, TIME_LIMIT: 120.0,
  };
  const PHYSICS = {
    cell: C.CELL, radius: C.SHIP_RADIUS, v_min: C.V_MIN, v_max: C.V_MAX, accel_k: C.ACCEL_K,
    pitch_rate: C.PITCH_RATE, yaw_rate: C.YAW_RATE, dt: C.DT, substeps: C.SUBSTEPS,
    rays_x: C.RAYS_X, rays_y: C.RAYS_Y, fov: C.FOV, ray_range: C.RAY_RANGE,
  };
  const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

  // the maze generator lives in ship.js (the leaderboard server uses it too)
  const makeMaze = (seed, rooms) => root.MazeSim.makeMaze(seed, rooms, C.CELL);

  // ---------------------------------------------------------------- the sensors (src/ship.py observe)
  function observe(ship, prevA) {
    const m = ship.m, G = m.G, clip = (c) => c.map((v) => Math.max(0, Math.min(G - 1, v)));
    const c = clip(m.cellOf(ship.pos));
    const c1 = clip(V.add(c, m.nextDir(c)));
    const c2 = clip(V.add(c1, m.nextDir(c1)));
    const r = ship.right();
    const body = (v) => [V.dot(v, r), V.dot(v, ship.u), V.dot(v, ship.f)];
    const b1 = body(V.sub(m.center(c1), ship.pos)).map((v) => v / C.BEACON_SCALE);
    const b2 = body(V.sub(m.center(c2), ship.pos)).map((v) => v / C.BEACON_SCALE);
    const vel = body(ship.vel).map((v) => v / C.V_MAX);
    return [...ship.depth, ...b1, ...b2, ...vel, ...prevA];
  }

  // ---------------------------------------------------------------- the autopilot (src/autopilot.py)
  const GAIN = 2.5, LOOK = 0.25, N_RAYS = C.RAYS_X * C.RAYS_Y;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  function autopilot(o) {
    const b1 = o.slice(N_RAYS, N_RAYS + 3).map((v) => v * C.BEACON_SCALE);
    const b2 = o.slice(N_RAYS + 3, N_RAYS + 6).map((v) => v * C.BEACON_SCALE);
    const aim = V.add(b1, V.mul(V.sub(b2, b1), LOOK));
    const yaw = Math.atan2(aim[0], aim[2]);
    const pitch = Math.atan2(aim[1], Math.hypot(aim[0], aim[2]));
    const off = Math.atan2(Math.hypot(aim[0], aim[1]), aim[2]);
    const seg = V.sub(b2, b1), segLen = V.len(seg);
    const fwd = segLen > 1e-6 ? seg[2] / Math.max(segLen, 1e-6) : 1.0;
    const turnAhead = clamp(1 - fwd, 0, 1);
    const err = clamp(off / 0.8, 0, 1);
    const close = clamp(1 - V.len(b1) / (2 * C.CELL), 0, 1);
    const slow = Math.max(turnAhead * (0.5 + 0.5 * close), err);
    const speed = C.V_MAX - (C.V_MAX - C.V_MIN) * slow;
    return [clamp(GAIN * pitch, -1, 1), clamp(GAIN * yaw, -1, 1),
            clamp((2 * (speed - C.V_MIN)) / (C.V_MAX - C.V_MIN) - 1, -1, 1)];
  }

  // ---------------------------------------------------------------- a whole flight (src/flights.py fly)
  const r3 = (a, k) => a.map((v) => Math.round(v * 10 ** k) / 10 ** k);
  function frame(ship) {
    return { pos: r3(ship.pos, 3), f: r3(ship.f, 4), u: r3(ship.u, 4), speed: V.len(ship.vel), depth: r3(ship.depth, 2) };
  }

  function fly(pilot, mazeJson) {
    const m = new MazeData(mazeJson);
    const ship = new Ship(m, PHYSICS);
    const timeLimit = Math.max(C.TIME_LIMIT, 0.45 * m.pathLength());
    const maxSteps = Math.round(timeLimit / C.DT);
    let prev = [0, 0, 0], steps = 0, dist = 0, end = "timeout";
    ship.depth = ship.castRays();
    const frames = [frame(ship)], actions = [];
    while (steps < maxSteps) {
      const a = pilot(observe(ship, prev)).map((v) => clamp(v, -1, 1));
      const before = ship.pos;
      ship.step(a);
      steps += 1;
      dist += V.len(V.sub(ship.pos, before));
      ship.depth = ship.castRays();
      prev = a;
      actions.push(r3(a, 3));
      const fr = frame(ship);
      if (ship.crashed || ship.reached || steps >= maxSteps) {
        end = ship.reached ? "exit" : ship.crashed ? "crash" : "timeout";
        fr.end = end;
        frames.push(fr);
        break;
      }
      frames.push(fr);
    }
    const time = end === "timeout" ? timeLimit : steps * C.DT;
    return { frames, actions, end, time, progress: clamp(ship.progress(), 0, 1), speed: dist / Math.max(time, 1e-6),
             path_length: m.pathLength() };
  }

  // ---------------------------------------------------------------- the interface used by app.js
  let pilots = null, labels = null, message = null;
  const cache = new Map();

  async function loadPilots() {
    pilots = new Map(); labels = new Map();
    try {
      const model = await root.MLP.load("model");
      if (model) {
        for (const spec of model.meta.networks) {
          const net = model.nets[spec.id];
          pilots.set(spec.id, (obs) => net.forward(obs).out);
          labels.set(spec.id, spec.label);
        }
      } else {
        message = "No trained network has been exported to this page yet, so the ship is flown by the hand-written autopilot. " +
                  "To add it: train, then run  python -m src.export_web  and publish again.";
      }
    } catch (e) {
      message = `Could not load the network (${e.message}). The ship is flown by the autopilot.`;
    }
    pilots.set("autopilot", autopilot);
    labels.set("autopilot", "Autopilot (hand-written rules)");
  }

  root.Backend = {
    async info() {
      if (!pilots) await loadPilots();
      return { pilots: [...labels].map(([id, label]) => ({ id, label })), message, physics: PHYSICS };
    },
    async maze(seed, rooms) {
      return makeMaze(seed & 0x7fffffff, Math.max(2, Math.min(7, rooms | 0)));
    },
    async fly({ seed, rooms, pilot }) {
      if (!pilots) await loadPilots();
      const pid = pilots.has(pilot) ? pilot : pilots.keys().next().value;
      const key = `${seed}/${rooms}/${pid}`;
      if (cache.has(key)) return cache.get(key);
      await new Promise((r) => setTimeout(r, 0));          // let the page show "Computing..." first
      const out = { pilot: pid, label: labels.get(pid), ...fly(pilots.get(pid), await this.maze(seed, rooms)) };
      cache.set(key, out);
      if (cache.size > 40) cache.delete(cache.keys().next().value);
      return out;
    },
  };
  root.MazeBackend = { makeMaze, observe, autopilot, fly, C, PHYSICS };
})(typeof window !== "undefined" ? window : globalThis);
