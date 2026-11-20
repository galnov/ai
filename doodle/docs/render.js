// =====================================================================================
//  A drawing (list of strokes) -> a 28x28 image, exactly like the Quick Draw images.
//  The JavaScript twin of src/render.py - the network only recognizes drawings well if they
//  go through the same processing as its training images.
//    1. Move the drawing to (0,0), scale its largest side to 255, simplify each stroke (RDP, eps = 2)
//    2. Center it on a 256 x 256 canvas with padding, draw 16-wide round lines,
//       at 8x resolution, then shrink to 28 x 28 by averaging blocks of 8 x 8 pixels
// =====================================================================================
(function (root) {
  const SIDE = 28, ORIGINAL = 256, LINE = 16, PADDING = 16, SUPER = 8;

  function rdp(pts, eps) {
    if (pts.length < 3) return pts;
    const [sx, sy] = pts[0], [ex, ey] = pts[pts.length - 1];
    const dx = ex - sx, dy = ey - sy, norm = Math.hypot(dx, dy);
    let best = -1, bi = 0;
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i];
      const d = norm === 0 ? Math.hypot(x - sx, y - sy) : Math.abs(dx * (y - sy) - dy * (x - sx)) / norm;
      if (d > best) { best = d; bi = i; }
    }
    if (best > eps) {
      const left = rdp(pts.slice(0, bi + 1), eps), right = rdp(pts.slice(bi), eps);
      return left.slice(0, -1).concat(right);
    }
    return [pts[0], pts[pts.length - 1]];
  }

  function simplify(strokes, eps = 2) {
    const ss = strokes.filter((s) => s.length === 2 && s[0].length > 0).map((s) => s[0].map((x, i) => [x, s[1][i]]));
    if (!ss.length) return [];
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const s of ss) for (const [x, y] of s) { mnx = Math.min(mnx, x); mny = Math.min(mny, y); mxx = Math.max(mxx, x); mxy = Math.max(mxy, y); }
    const span = Math.max(mxx - mnx, mxy - mny), k = span > 0 ? 255 / span : 1;
    return ss.map((s) => rdp(s.map(([x, y]) => [(x - mnx) * k, (y - mny) * k]), eps));
  }

  let canvas = null;
  function rasterize(simplified) {
    const big = SIDE * SUPER;
    if (!canvas) { canvas = document.createElement("canvas"); canvas.width = canvas.height = big; }
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, big, big);
    const out = new Uint8Array(SIDE * SIDE);
    if (!simplified.length) return out;
    const total = PADDING * 2 + LINE, k = big / (ORIGINAL + total);
    let bx = 0, by = 0;
    for (const s of simplified) for (const [x, y] of s) { bx = Math.max(bx, x); by = Math.max(by, y); }
    const ox = (ORIGINAL - bx) / 2 + total / 2, oy = (ORIGINAL - by) / 2 + total / 2;
    const w = LINE * k;
    ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff";
    ctx.lineWidth = Math.max(1, Math.round(w)); ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const s of simplified) {
      const pts = s.map(([x, y]) => [(x + ox) * k, (y + oy) * k]);
      if (pts.length > 1) {
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.stroke();
      }
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, w / 2, 0, Math.PI * 2); ctx.fill(); }   // round caps and joints
    }
    const data = ctx.getImageData(0, 0, big, big).data;
    for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) {       // average 8 x 8 blocks
      let sum = 0;
      for (let j = 0; j < SUPER; j++) {
        const row = ((y * SUPER + j) * big + x * SUPER) * 4;
        for (let i = 0; i < SUPER; i++) sum += data[row + i * 4];
      }
      out[y * SIDE + x] = Math.round(sum / (SUPER * SUPER));
    }
    return out;
  }

  root.DoodleRender = { simplify, rasterize, strokesToImage: (strokes) => rasterize(simplify(strokes)) };
})(window);
