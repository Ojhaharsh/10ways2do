/* ═══════════════════════════════════════════════════════════════════════════
   10ways2do — Radar Chart Engine
   Renders animated, interactive 8-axis radar charts on HTML Canvas.
   ═══════════════════════════════════════════════════════════════════════════ */

const Charts = (() => {
  const AXIS_LABELS = ['Accuracy', 'Speed', 'Cost', 'Robustness', 'Fairness', 'Consistency', 'Generalization', 'Efficiency'];
  const AXIS_COLORS = ['#ff6b6b', '#ffa94d', '#ffd43b', '#69db7c', '#38d9a9', '#4dabf7', '#845ef7', '#e599f7'];

  const ACCENT = '#845ef7';
  const ACCENT_GLOW = 'rgba(132, 94, 247, 0.25)';
  const FILL_COLOR = 'rgba(132, 94, 247, 0.12)';
  const STROKE_COLOR = 'rgba(132, 94, 247, 0.8)';
  const GRID_COLOR = 'rgba(255, 255, 255, 0.06)';
  const LABEL_COLOR = '#8b8d97';

  /**
   * Draw a radar chart on a canvas element.
   * @param {HTMLCanvasElement} canvas
   * @param {number[]} scores - Array of 8 values, each 0.0 - 1.0
   * @param {Object} opts - Options
   */
  function drawRadar(canvas, scores, opts = {}) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;

    // Handle HiDPI
    if (!canvas.__scaled) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.scale(dpr, dpr);
      canvas.__scaled = true;
    }

    const cx = w / 2;
    const cy = h / 2;
    // Reduce maxR to leave more room for long labels like "Generalization"
    const maxR = Math.min(cx, cy) * 0.62;
    const n = AXIS_LABELS.length;
    const angleStep = (2 * Math.PI) / n;
    const startAngle = -Math.PI / 2; // Start from top

    const fillColor = opts.fillColor || FILL_COLOR;
    const strokeColor = opts.strokeColor || STROKE_COLOR;
    const showLabels = opts.showLabels !== false;
    const showDots = opts.showDots !== false;
    const showValues = opts.showValues || false;
    const animated = opts.animated !== false;

    ctx.clearRect(0, 0, w, h);

    // ── Grid circles ──
    for (let ring = 1; ring <= 5; ring++) {
      const r = (ring / 5) * maxR;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ── Axis lines ──
    for (let i = 0; i < n; i++) {
      const angle = startAngle + i * angleStep;
      const x = cx + Math.cos(angle) * maxR;
      const y = cy + Math.sin(angle) * maxR;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ── Labels ──
    if (showLabels) {
      ctx.font = '500 11px "Inter", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const labelR = maxR + 28;
        const x = cx + Math.cos(angle) * labelR;
        const y = cy + Math.sin(angle) * labelR;

        ctx.fillStyle = AXIS_COLORS[i];
        ctx.fillText(AXIS_LABELS[i], x, y);
      }
    }

    // ── Data polygon ──
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const angle = startAngle + i * angleStep;
      const r = (scores[i] || 0) * maxR;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();

    // Glow effect
    ctx.shadowColor = ACCENT_GLOW;
    ctx.shadowBlur = 20;
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // ── Data dots ──
    if (showDots) {
      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const r = (scores[i] || 0) * maxR;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        // Outer glow
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = AXIS_COLORS[i] + '30';
        ctx.fill();

        // Inner dot
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = AXIS_COLORS[i];
        ctx.fill();
      }
    }

    // ── Values ──
    if (showValues) {
      ctx.font = '600 10px "JetBrains Mono", monospace';
      for (let i = 0; i < n; i++) {
        const angle = startAngle + i * angleStep;
        const r = (scores[i] || 0) * maxR + 16;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        ctx.fillStyle = '#e8e8ec';
        ctx.textAlign = 'center';
        ctx.fillText((scores[i] * 100).toFixed(0) + '%', x, y);
      }
    }
  }

  /**
   * Animate a radar chart from zeros to final scores.
   */
  function animateRadar(canvas, targetScores, opts = {}, duration = 1200) {
    const startTime = performance.now();
    const currentScores = new Array(8).fill(0);

    function frame(now) {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / duration);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - t, 3);

      for (let i = 0; i < 8; i++) {
        currentScores[i] = (targetScores[i] || 0) * ease;
      }

      drawRadar(canvas, currentScores, opts);

      if (t < 1) requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  /**
   * Draw a mini radar chart (for leaderboard rows).
   */
  function drawMiniRadar(canvas, scores) {
    drawRadar(canvas, scores, {
      showLabels: false,
      showDots: false,
      showValues: false,
    });
  }

  /**
   * Compare two models on the same radar chart.
   */
  function drawComparison(canvas, scoresA, scoresB, opts = {}) {
    const ctx = canvas.getContext('2d');

    // Draw A first (slightly dimmer)
    drawRadar(canvas, scoresA, {
      fillColor: 'rgba(255, 107, 107, 0.1)',
      strokeColor: 'rgba(255, 107, 107, 0.6)',
      showDots: true,
      ...opts,
    });

    // Draw B on top
    const n = AXIS_LABELS.length;
    const dpr = window.devicePixelRatio || 1;
    const w = parseInt(canvas.style.width || canvas.width / dpr);
    const h = parseInt(canvas.style.height || canvas.height / dpr);
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.min(cx, cy) * 0.62;
    const angleStep = (2 * Math.PI) / n;
    const startAngle = -Math.PI / 2;

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const angle = startAngle + i * angleStep;
      const r = (scoresB[i] || 0) * maxR;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(132, 94, 247, 0.1)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(132, 94, 247, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  /**
   * Draw a Performance vs Cost Scatter Plot
   */
  function drawScatterPlot(canvas, dataPoints, opts = {}) {
    if (!canvas.__sized) {
      const rect = canvas.parentElement.getBoundingClientRect();
      const style = window.getComputedStyle(canvas.parentElement);
      canvas.width = rect.width - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      canvas.height = rect.height - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
      canvas.__sized = true;
    }

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;

    if (!canvas.__scaled) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.scale(dpr, dpr);
      canvas.__scaled = true;
    }

    ctx.clearRect(0, 0, w, h);

    // Increase internal padding to prevent axis label cutoff
    const padding = { top: 20, right: 30, bottom: 60, left: 70 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // Find min/max
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    dataPoints.forEach(p => {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    });

    // Add some padding to bounds
    const xRange = maxX - minX;
    const yRange = maxY - minY;
    minX = Math.max(0, minX - xRange * 0.1);
    maxX = maxX + xRange * 0.1;
    minY = Math.max(0, minY - yRange * 0.1);
    maxY = Math.min(1, maxY + yRange * 0.1);

    // Helpers
    const scaleX = val => padding.left + ((val - minX) / (maxX - minX)) * plotW;
    const scaleY = val => padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH;

    // Draw grid & axes
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    
    // Y-axis grid
    for(let i = 0; i <= 5; i++) {
      const yVal = minY + (maxY - minY) * (i / 5);
      const y = scaleY(yVal);
      ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(w - padding.right, y); ctx.stroke();
      
      // Label
      ctx.fillStyle = LABEL_COLOR;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(yVal.toFixed(2), padding.left - 10, y);
    }

    // X-axis grid
    for(let i = 0; i <= 5; i++) {
      const xVal = minX + (maxX - minX) * (i / 5);
      const x = scaleX(xVal);
      ctx.beginPath(); ctx.moveTo(x, padding.top); ctx.lineTo(x, h - padding.bottom); ctx.stroke();
      
      // Label
      ctx.fillStyle = LABEL_COLOR;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      // Use 4 decimal places for cost since it can be very small
      ctx.fillText('$' + xVal.toFixed(4), x, h - padding.bottom + 10);
    }

    // Axis titles
    ctx.fillStyle = '#e8e8ec';
    ctx.font = '12px "Inter", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Estimated Cost per 1k Tokens ($)', padding.left + plotW/2, h - 10);
    
    ctx.save();
    ctx.translate(15, padding.top + plotH/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('Aggregate Performance Score', 0, 0);
    ctx.restore();

    // Draw points
    dataPoints.forEach(p => {
      const x = scaleX(p.x);
      const y = scaleY(p.y);

      // Glow
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI*2);
      ctx.fillStyle = (p.color || ACCENT) + '40';
      ctx.fill();

      // Core
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI*2);
      ctx.fillStyle = p.color || ACCENT;
      ctx.fill();

      // Label
      ctx.fillStyle = '#e8e8ec';
      ctx.font = '600 11px "Inter", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.label, x + 12, y);
    });
  }

  return { drawRadar, animateRadar, drawMiniRadar, drawComparison, drawScatterPlot, AXIS_LABELS, AXIS_COLORS };
})();
