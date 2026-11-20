// =====================================================================================
//  The neon wireframe renderer (WebGL2), written for this project.
//
//  Cockpit view: every wall face of the maze is a black square. The fragment shader draws a glowing
//  green grid on it, 1 m between lines, computed from the position on the wall - so lines stay sharp at
//  any distance (fwidth gives the width of one pixel in grid units). Far walls fade into a dark gray
//  fog, which gives the tunnel its depth. The exit cell glows in a different color.
//
//  Map view: the whole maze as thin lines, seen from outside and slowly turning, with the way out,
//  the ship's trail, and the ship.
// =====================================================================================
(function (root) {
  const M = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    lookAt(e, t, up) {
      let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2];
      let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
      let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
      l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
      const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
      return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
        -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
    },
    multiply(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    // world point -> [x, y] in pixels, or null if behind the camera
    project(vp, p, w, h) {
      const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12];
      const y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13];
      const ww = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
      if (ww <= 0.05) return null;
      return [(x / ww * 0.5 + 0.5) * w, (1 - (y / ww * 0.5 + 0.5)) * h];
    },
  };

  function program(gl, vs, fs) {
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function fit(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  // ---------------------------------------------------------------- cockpit view
  const WALL_VS = `#version 300 es
  in vec3 aPos; in vec3 aNormal;
  uniform mat4 uVP;
  out vec3 vWorld; out vec3 vNormal;
  void main() { vWorld = aPos; vNormal = aNormal; gl_Position = uVP * vec4(aPos, 1.0); }`;

  const WALL_FS = `#version 300 es
  precision highp float;
  in vec3 vWorld; in vec3 vNormal;
  uniform vec3 uCam; uniform vec3 uLine; uniform vec3 uExitLine; uniform vec3 uFogColor;
  uniform vec2 uFog; uniform vec3 uExitMin; uniform vec3 uExitMax; uniform float uTime; uniform float uWidth;
  out vec4 frag;
  void main() {
    vec3 n = abs(vNormal);
    vec2 uv = n.x > 0.5 ? vWorld.zy : (n.y > 0.5 ? vWorld.xz : vWorld.xy);   // position on the wall (m)
    vec2 g = abs(fract(uv - 0.5) - 0.5) / fwidth(uv);                         // distance to a line, in pixels
    float d = min(g.x, g.y);
    float core = 1.0 - smoothstep(0.5 * uWidth, 0.5 * uWidth + 1.0, d);
    float glow = exp(-d * 0.45) * 0.45;
    bool isExit = all(greaterThan(vWorld, uExitMin)) && all(lessThan(vWorld, uExitMax));
    vec3 lineCol = isExit ? uExitLine * (0.75 + 0.25 * sin(uTime * 5.0)) : uLine;
    float dist = length(vWorld - uCam);
    float fog = smoothstep(uFog.x, uFog.y, dist);
    vec3 col = vec3(0.012, 0.016, 0.014) + lineCol * (core + glow) * (1.0 - 0.8 * fog);
    if (isExit) col += uExitLine * 0.06;
    col = mix(col, uFogColor, fog * 0.9);
    frag = vec4(col, 1.0);
  }`;

  class CockpitView {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl2", { antialias: true });
      if (!gl) throw new Error("WebGL2 is not available in this browser");
      this.gl = gl;
      this.prog = program(gl, WALL_VS, WALL_FS);
      this.u = {};
      for (const n of ["uVP", "uCam", "uLine", "uExitLine", "uFogColor", "uFog", "uExitMin", "uExitMax", "uTime", "uWidth"])
        this.u[n] = gl.getUniformLocation(this.prog, n);
      this.vao = gl.createVertexArray();
      this.buffers = [];
      this.count = 0;
      this.vp = null;
    }

    setMaze(m) {
      const gl = this.gl, c = m.cell, pos = [], nrm = [];
      const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      for (let x = 0; x < m.G; x++) for (let y = 0; y < m.G; y++) for (let z = 0; z < m.G; z++) {
        if (m.isSolidCell([x, y, z])) continue;
        for (const d of DIRS) {
          if (!m.isSolidCell([x + d[0], y + d[1], z + d[2]])) continue;
          const ax = d[0] ? 0 : d[1] ? 1 : 2, s = d[ax];
          const o = [(ax + 1) % 3, (ax + 2) % 3];
          const base = [x * c, y * c, z * c];
          base[ax] += s > 0 ? c : 0;
          const corner = (i, j) => { const p = base.slice(); p[o[0]] += i * c; p[o[1]] += j * c; return p; };
          const quad = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 0), corner(1, 1), corner(0, 1)];
          for (const p of quad) { pos.push(...p); nrm.push(-d[0], -d[1], -d[2]); }
        }
      }
      for (const b of this.buffers) gl.deleteBuffer(b);
      this.buffers = [];
      gl.bindVertexArray(this.vao);
      for (const [name, data] of [["aPos", pos], ["aNormal", nrm]]) {
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(this.prog, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
        this.buffers.push(b);
      }
      this.count = pos.length / 3;
      this.exitMin = m.exit.map((v) => v * c - 0.01);
      this.exitMax = m.exit.map((v) => (v + 1) * c + 0.01);
    }

    render(cam, time, frameMs = 16, fovDeg = 78) {
      const gl = this.gl;
      fit(this.canvas);
      const W = this.canvas.width, H = this.canvas.height;
      const proj = M.perspective(fovDeg * Math.PI / 180, W / H, 0.05, 200);
      this.vp = M.multiply(proj, M.lookAt(cam.pos, [cam.pos[0] + cam.f[0], cam.pos[1] + cam.f[1], cam.pos[2] + cam.f[2]], cam.u));
      gl.viewport(0, 0, W, H);
      gl.clearColor(0.13, 0.13, 0.13, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      const u = this.u;
      gl.uniformMatrix4fv(u.uVP, false, this.vp);
      gl.uniform3fv(u.uCam, cam.pos);
      gl.uniform3f(u.uLine, 0.22, 1.0, 0.12);
      gl.uniform3f(u.uExitLine, 1.0, 0.25, 0.85);
      gl.uniform3f(u.uFogColor, 0.13, 0.13, 0.13);
      gl.uniform2f(u.uFog, 3.0, 34.0);
      gl.uniform3fv(u.uExitMin, this.exitMin);
      gl.uniform3fv(u.uExitMax, this.exitMax);
      gl.uniform1f(u.uTime, time);
      gl.uniform1f(u.uWidth, Math.max(1.5, H / 420));
      gl.drawArrays(gl.TRIANGLES, 0, this.count);
    }
  }

  // ---------------------------------------------------------------- map view
  const LINE_VS = `#version 300 es
  in vec3 aPos; uniform mat4 uVP; uniform float uSize;
  void main() { gl_Position = uVP * vec4(aPos, 1.0); gl_PointSize = uSize; }`;
  const LINE_FS = `#version 300 es
  precision highp float; uniform vec4 uColor; out vec4 frag;
  void main() { frag = uColor; }`;

  class MapView {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl2", { antialias: true });
      if (!gl) throw new Error("WebGL2 is not available");
      this.gl = gl;
      this.prog = program(gl, LINE_VS, LINE_FS);
      this.uVP = gl.getUniformLocation(this.prog, "uVP");
      this.uColor = gl.getUniformLocation(this.prog, "uColor");
      this.uSize = gl.getUniformLocation(this.prog, "uSize");
      this.buf = gl.createBuffer();
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      const loc = gl.getAttribLocation(this.prog, "aPos");
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    }

    setMaze(m) {
      const c = m.cell, edges = [];
      // the edges between corridor and rock: draw each empty cell as a wire box
      for (let x = 0; x < m.G; x++) for (let y = 0; y < m.G; y++) for (let z = 0; z < m.G; z++) {
        if (m.isSolidCell([x, y, z])) continue;
        const p = (i, j, k) => [(x + i) * c, (y + j) * c, (z + k) * c];
        const E = [[0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1]];
        for (const e of E) edges.push(...p(e[0], e[1], e[2]), ...p(e[3], e[4], e[5]));
      }
      this.edges = new Float32Array(edges);
      const path = [];
      for (let i = 0; i + 1 < m.path.length; i++) path.push(...m.center(m.path[i]), ...m.center(m.path[i + 1]));
      this.path = new Float32Array(path);
      this.center = [m.G * c / 2, m.G * c / 2, m.G * c / 2];
      this.radius = m.G * c * 1.25;
      this.exit = new Float32Array(m.center(m.exit));
      this.startP = new Float32Array(m.center(m.start));
    }

    draw(mode, data, color, size = 1) {
      const gl = this.gl;
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      gl.uniform4fv(this.uColor, color);
      gl.uniform1f(this.uSize, size);
      gl.drawArrays(mode, 0, data.length / 3);
    }

    render(angle, trail, ships) {
      const gl = this.gl;
      fit(this.canvas);
      const W = this.canvas.width, H = this.canvas.height, c = this.center;
      const eye = [c[0] + Math.cos(angle) * this.radius, c[1] + this.radius * 0.55, c[2] + Math.sin(angle) * this.radius];
      const vp = M.multiply(M.perspective(0.75, W / H, 0.5, 500), M.lookAt(eye, c, [0, 1, 0]));
      gl.viewport(0, 0, W, H);
      gl.clearColor(0.02, 0.03, 0.025, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.uniformMatrix4fv(this.uVP, false, vp);
      this.draw(gl.LINES, this.edges, [0.2, 0.9, 0.15, 0.13]);
      this.draw(gl.LINES, this.path, [0.95, 0.85, 0.2, 0.55]);
      if (trail && trail.length > 1) {
        const t = [];
        for (let i = 0; i + 1 < trail.length; i++) t.push(...trail[i], ...trail[i + 1]);
        this.draw(gl.LINES, new Float32Array(t), [0.3, 0.95, 1.0, 0.95]);
      }
      this.draw(gl.POINTS, this.exit, [1, 0.3, 0.85, 1], 12);
      for (const s of ships) this.draw(gl.POINTS, new Float32Array(s.pos), s.color, s.size || 10);
    }
  }

  root.Neon = { CockpitView, MapView, M };
})(window);
