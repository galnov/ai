// =====================================================================================
//  A tiny 3D engine (WebGL2), written for this project instead of using a library.
//  Everything in the scene is a box. Each box has a position, size, rotation, color and an
//  optional surface pattern (windows, stripes, railway sleepers) computed in the shader.
//
//  The math, in short:
//    model matrix  = translate * rotateY * rotateX * scale     (box -> world)
//    view matrix   = lookAt(camera, target)                     (world -> camera)
//    projection    = perspective(field of view, aspect, near, far)  (camera -> screen)
//    light         = ambient (from the sky) + diffuse (sun):  color * (a + d * max(0, n·L))
//    fog           = mix(color, sky color, how far away)      -> hides the end of the world
// =====================================================================================
(function (root) {
  const VS = `#version 300 es
  in vec3 aPos; in vec3 aNormal;
  uniform mat4 uViewProj, uModel; uniform vec3 uScale;
  out vec3 vNormal; out vec3 vWorld; out vec3 vLocal;
  void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vWorld = w.xyz;
    vNormal = mat3(uModel) * aNormal;
    vLocal = aPos * uScale;            // position inside the box, in meters (for patterns)
    gl_Position = uViewProj * w;
  }`;

  const FS = `#version 300 es
  precision highp float;
  in vec3 vNormal; in vec3 vWorld; in vec3 vLocal;
  uniform vec4 uColor; uniform vec3 uColor2; uniform int uPattern; uniform float uScroll;
  uniform vec3 uCam; uniform vec3 uSky; uniform vec2 uFog; uniform vec3 uSun;
  out vec4 frag;
  void main() {
    vec3 n = normalize(vNormal);
    vec3 base = uColor.rgb;
    float glow = 0.0;
    if (uPattern == 1 && abs(n.y) < 0.5) {                 // windows on the sides
      vec2 p = vec2(abs(n.x) > 0.5 ? vLocal.z : vLocal.x, vLocal.y);
      vec2 c = fract(p / vec2(1.7, 1.5));
      float w = step(0.22, c.x) * step(c.x, 0.78) * step(0.28, c.y) * step(c.y, 0.8);
      base = mix(base, uColor2, w);
      glow = w * 0.25;
    }
    if (uPattern == 2) {                                    // warning stripes
      float s = step(0.5, fract((vLocal.x + vLocal.y) * 2.2));
      base = mix(base, uColor2, s);
    }
    if (uPattern == 3) {                                    // railway: sleepers + two rails
      float sl = step(0.68, fract((vWorld.z + uScroll) / 0.9));
      base = mix(base, uColor2, sl * step(abs(vLocal.x), 0.95));
      float rail = step(abs(abs(vLocal.x) - 0.72), 0.05);
      base = mix(base, vec3(0.78, 0.8, 0.86), rail);
    }
    if (uPattern == 4) {                                    // sidewalk tiles
      vec2 c = fract(vec2(vLocal.x, vWorld.z + uScroll) / 1.2);
      float line = 1.0 - step(0.06, c.x) * step(0.06, c.y);
      base = mix(base, base * 0.82, line);
    }
    float diff = max(dot(n, normalize(uSun)), 0.0);
    float sky = 0.55 + 0.45 * n.y;
    vec3 col = base * (0.42 * sky + 0.7 * diff) + base * glow;
    float d = length(vWorld - uCam);
    float f = clamp((d - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0);
    frag = vec4(mix(col, uSky, f * f), uColor.a);
  }`;

  // ---------------------------------------------------------------- 4x4 matrices (column-major)
  const M = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    lookAt(e, t, up = [0, 1, 0]) {
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
    // translate * rotateY(ry) * rotateX(rx) * scale
    model(x, y, z, sx, sy, sz, ry = 0, rx = 0) {
      const cy = Math.cos(ry), sy_ = Math.sin(ry), cx = Math.cos(rx), sx_ = Math.sin(rx);
      // columns of R = Ry * Rx
      const c0 = [cy, 0, -sy_], c1 = [sy_ * sx_, cx, cy * sx_], c2 = [sy_ * cx, -sx_, cy * cx];
      return new Float32Array([c0[0] * sx, c0[1] * sx, c0[2] * sx, 0, c1[0] * sy, c1[1] * sy, c1[2] * sy, 0,
        c2[0] * sz, c2[1] * sz, c2[2] * sz, 0, x, y, z, 1]);
    },
  };

  function cubeGeometry() {
    const faces = [
      [[1, 0, 0], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]],
      [[-1, 0, 0], [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1]]],
      [[0, 1, 0], [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]]],
      [[0, -1, 0], [[-1, -1, 1], [-1, -1, -1], [1, -1, -1], [1, -1, 1]]],
      [[0, 0, 1], [[1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]]],
      [[0, 0, -1], [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]]],
    ];
    const pos = [], nrm = [];
    for (const [n, q] of faces) {
      for (const i of [0, 1, 2, 0, 2, 3]) { pos.push(...q[i].map((v) => v * 0.5)); nrm.push(...n); }
    }
    return { pos: new Float32Array(pos), nrm: new Float32Array(nrm) };
  }

  const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];

  class Engine {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl2", { antialias: true });
      if (!gl) throw new Error("WebGL2 is not available in this browser");
      this.gl = gl;
      const sh = (type, src) => {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      this.prog = p;
      this.u = {};
      for (const n of ["uViewProj", "uModel", "uScale", "uColor", "uColor2", "uPattern", "uScroll", "uCam", "uSky", "uFog", "uSun"])
        this.u[n] = gl.getUniformLocation(p, n);
      const geo = cubeGeometry();
      this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
      for (const [name, data] of [["aPos", geo.pos], ["aNormal", geo.nrm]]) {
        const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(p, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      }
      this.sky = hex("#bfe3f2"); this.fog = [40, 120]; this.sun = [-0.45, 0.9, 0.4];
      this.queue = []; this.eye = [0, 3, 6]; this.viewProj = null; this.scroll = 0;
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    }

    camera(eye, target, fov = 1.05) {
      this.resize();
      this.eye = eye;
      const proj = M.perspective(fov, this.canvas.width / this.canvas.height, 0.1, 400);
      this.viewProj = M.multiply(proj, M.lookAt(eye, target));
    }

    // add a box to this frame. color: "#rrggbb" or [r,g,b]
    box(x, y, z, sx, sy, sz, color, o = {}) {
      this.queue.push({ x, y, z, sx, sy, sz, c: typeof color === "string" ? hex(color) : color,
        c2: o.color2 ? (typeof o.color2 === "string" ? hex(o.color2) : o.color2) : [1, 1, 1],
        pattern: o.pattern || 0, alpha: o.alpha ?? 1, ry: o.ry || 0, rx: o.rx || 0 });
    }

    render() {
      const gl = this.gl, u = this.u;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(...this.sky, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.uniformMatrix4fv(u.uViewProj, false, this.viewProj);
      gl.uniform3fv(u.uCam, this.eye); gl.uniform3fv(u.uSky, this.sky);
      gl.uniform2fv(u.uFog, this.fog); gl.uniform3fv(u.uSun, this.sun);
      gl.uniform1f(u.uScroll, this.scroll);

      const opaque = this.queue.filter((b) => b.alpha >= 1);
      const glass = this.queue.filter((b) => b.alpha < 1)
        .sort((a, b) => Math.hypot(b.x - this.eye[0], b.z - this.eye[2]) - Math.hypot(a.x - this.eye[0], a.z - this.eye[2]));
      const draw = (b) => {
        gl.uniformMatrix4fv(u.uModel, false, M.model(b.x, b.y, b.z, b.sx, b.sy, b.sz, b.ry, b.rx));
        gl.uniform3f(u.uScale, b.sx, b.sy, b.sz);
        gl.uniform4f(u.uColor, b.c[0], b.c[1], b.c[2], b.alpha);
        gl.uniform3fv(u.uColor2, b.c2);
        gl.uniform1i(u.uPattern, b.pattern);
        gl.drawArrays(gl.TRIANGLES, 0, 36);
      };
      gl.disable(gl.BLEND); gl.depthMask(true);
      opaque.forEach(draw);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      glass.forEach(draw);
      gl.depthMask(true);
      this.queue = [];
    }
  }

  root.Engine3D = { Engine, M, hex };
})(window);
