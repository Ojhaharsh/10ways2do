/* ═══════════════════════════════════════════════════════════════════════════
   10ways2do — Application Logic
   Interactive features: radar demos, scoring profiles, stats counters,
   leaderboard data loading, and model submission form.
   ═══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initHeroRadar();
  initDemoRadar();
  initScatterPreview();
  initH2HComparison();
  initProfileSwitcher();
  initCounters();
  loadLeaderboardData();
  initSubmitForm();
});

/* ── Axis metadata ── */
const AXES_META = {
  accuracy:       { weight: { balanced: 20, enterprise: 20, research: 30, safety: 10 }, metric: 'Exact match + partial credit', desc: 'How often the model produces correct answers across all difficulty levels and challenge types.' },
  speed:          { weight: { balanced: 10, enterprise: 15, research: 5,  safety: 5  }, metric: 'Average latency (ms)',         desc: 'Response speed normalized by challenge difficulty. Faster responses for easy problems score higher.' },
  cost:           { weight: { balanced: 10, enterprise: 20, research: 5,  safety: 5  }, metric: 'USD per challenge',            desc: 'Dollar cost per evaluation challenge. Lower cost = higher score. Local models score perfectly here.' },
  robustness:     { weight: { balanced: 15, enterprise: 20, research: 10, safety: 30 }, metric: 'Score stability across difficulty', desc: 'How well performance holds from trivial to expert difficulty. Low drop-off = high robustness.' },
  fairness:       { weight: { balanced: 10, enterprise: 10, research: 5,  safety: 25 }, metric: 'Score variance across types',  desc: 'Consistent performance regardless of challenge content or type. Low variance = fair.' },
  consistency:    { weight: { balanced: 10, enterprise: 5,  research: 10, safety: 15 }, metric: 'Interquartile range',          desc: 'Reliability of producing similar quality responses. Low spread = consistent.' },
  generalization: { weight: { balanced: 15, enterprise: 5,  research: 25, safety: 5  }, metric: 'Cross-domain breadth',         desc: 'Performance across different domains. A model that excels everywhere generalizes well.' },
  efficiency:     { weight: { balanced: 10, enterprise: 5,  research: 10, safety: 5  }, metric: 'Tokens per correct answer',    desc: 'Quality per token spent. Concise correct answers score higher than verbose ones.' },
};

const AXES_ORDER = ['accuracy', 'speed', 'cost', 'robustness', 'fairness', 'consistency', 'generalization', 'efficiency'];

/* ── Hero Radar (animated on load) ── */
function initHeroRadar() {
  const canvas = document.getElementById('hero-radar');
  if (!canvas) return;

  // Demo scores that look impressive
  const demoScores = [0.88, 0.72, 0.95, 0.68, 0.75, 0.82, 0.60, 0.78];
  Charts.animateRadar(canvas, demoScores, { showLabels: true, showDots: true }, 1800);
}

/* ── Demo Radar (interactive axis explorer) ── */
function initDemoRadar() {
  const canvas = document.getElementById('demo-radar');
  if (!canvas) return;

  const scores = [0.92, 0.65, 0.88, 0.71, 0.79, 0.85, 0.58, 0.73];
  Charts.animateRadar(canvas, scores, { showLabels: true, showDots: true, showValues: true }, 2000);

  // Click on canvas to select nearest axis
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    const angle = Math.atan2(y, x) + Math.PI / 2;
    const normalizedAngle = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const idx = Math.round((normalizedAngle / (2 * Math.PI)) * 8) % 8;
    showAxisDetail(AXES_ORDER[idx]);
  });
}

function showAxisDetail(axisKey) {
  const meta = AXES_META[axisKey];
  if (!meta) return;

  const nameEl = document.getElementById('axis-name');
  const descEl = document.getElementById('axis-desc');
  const weightEl = document.getElementById('axis-weight');
  const metricEl = document.getElementById('axis-metric');
  const dotEl = document.querySelector('.axis-dot');

  if (nameEl) nameEl.textContent = axisKey.charAt(0).toUpperCase() + axisKey.slice(1);
  if (descEl) descEl.textContent = meta.desc;
  if (weightEl) weightEl.textContent = meta.weight.balanced + '%';
  if (metricEl) metricEl.textContent = meta.metric;

  const colorIdx = AXES_ORDER.indexOf(axisKey);
  if (dotEl && Charts.AXIS_COLORS[colorIdx]) {
    dotEl.style.background = Charts.AXIS_COLORS[colorIdx];
    dotEl.style.boxShadow = `0 0 12px ${Charts.AXIS_COLORS[colorIdx]}50`;
  }
}

/* ── Scatter Preview ── */
async function initScatterPreview() {
  const canvas = document.getElementById('scatter-canvas');
  if (!canvas) return;

  // Mock fallback data
  const mockModels = [
    { label: 'Gemini 2.5 Flash', x: 0.15, y: 0.82, color: '#38d9a9' },
    { label: 'Llama 3 (70B)', x: 0.60, y: 0.85, color: '#4dabf7' },
    { label: 'Claude 3 Haiku', x: 0.25, y: 0.78, color: '#ff6b6b' },
    { label: 'Qwen 2.5 Coder', x: 0.35, y: 0.88, color: '#ffd43b' },
    { label: 'GPT-4o', x: 5.00, y: 0.94, color: '#845ef7' },
    { label: 'Gemini 1.5 Pro', x: 3.50, y: 0.91, color: '#e599f7' }
  ];

  let modelsToPlot = mockModels;

  try {
    const resp = await fetch('../results/challenge_registry.json');
    if (resp.ok) {
      const registry = await resp.json();
      const bestScores = {};
      
      for (const [sessionId, evals] of Object.entries(registry.evaluations || {})) {
        for (const ev of evals) {
          const mid = ev.model_id;
          const score = ev.summary?.mean_overall || 0;
          if (!bestScores[mid] || score > bestScores[mid].y) {
            const hash = mid.split('').reduce((acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0);
            const color = Charts.AXIS_COLORS[Math.abs(hash) % Charts.AXIS_COLORS.length];
            bestScores[mid] = { label: mid, x: ev.summary?.total_cost_usd || 0, y: score, color: color };
          }
        }
      }
      const realModels = Object.values(bestScores);
      if (realModels.length > 0) {
        modelsToPlot = realModels;
      }
    }
  } catch (e) {}

  Charts.drawScatterPlot(canvas, modelsToPlot);

  // Redraw on resize
  window.addEventListener('resize', () => {
    canvas.__scaled = false;
    Charts.drawScatterPlot(canvas, modelsToPlot);
  });
}

/* ── Helper to load individual model profiles ── */
async function loadModelProfile(mid) {
  try {
    const filename = mid.replace('/', '_');
    const resp = await fetch(`../results/EVAL_${filename}.json`);
    if (resp.ok) {
      const data = await resp.json();
      return data.aggregate_profile;
    }
  } catch (e) {
    console.warn("Could not load profile for", mid);
  }
  return null;
}

/* ── Head-to-Head Comparison ── */
async function initH2HComparison() {
  const canvas = document.getElementById('h2h-radar');
  const selectA = document.getElementById('model-a-select');
  const selectB = document.getElementById('model-b-select');
  if (!canvas || !selectA || !selectB) return;

  let profiles = {
    'gemini-flash': [0.82, 0.95, 0.90, 0.75, 0.80, 0.85, 0.70, 0.88],
    'qwen-coder':   [0.88, 0.80, 0.85, 0.85, 0.75, 0.80, 0.65, 0.75],
    'llama-3':      [0.85, 0.60, 0.70, 0.88, 0.82, 0.90, 0.85, 0.65]
  };

  try {
    const resp = await fetch('../results/challenge_registry.json');
    if (resp.ok) {
      const registry = await resp.json();
      const evaluatedModels = new Set();
      for (const [sessionId, evals] of Object.entries(registry.evaluations || {})) {
        for (const ev of evals) {
          evaluatedModels.add(ev.model_id);
        }
      }
      
      if (evaluatedModels.size > 0) {
        profiles = {};
        selectA.innerHTML = '';
        selectB.innerHTML = '';
        
        for (const mid of evaluatedModels) {
          const profile = await loadModelProfile(mid);
          if (profile) {
            profiles[mid] = profile.scores_array || new Array(8).fill(0);
            const optA = document.createElement('option');
            optA.value = mid; optA.textContent = mid;
            const optB = document.createElement('option');
            optB.value = mid; optB.textContent = mid;
            selectA.appendChild(optA);
            selectB.appendChild(optB);
          }
        }
        
        // Auto-select different models if possible
        if (selectA.options.length > 1) {
          selectB.selectedIndex = 1;
        }
      }
    }
  } catch (e) {}

  function updateH2H() {
    const pA = profiles[selectA.value] || new Array(8).fill(0);
    const pB = profiles[selectB.value] || new Array(8).fill(0);
    
    // Update legend labels
    const legends = document.querySelectorAll('.h2h-legend .legend-item');
    if (legends.length >= 2) {
      legends[0].innerHTML = `<span class="legend-color" style="background:var(--accent);"></span> ${selectA.options[selectA.selectedIndex]?.text || 'Model A'}`;
      legends[1].innerHTML = `<span class="legend-color" style="background:var(--accent-2);"></span> ${selectB.options[selectB.selectedIndex]?.text || 'Model B'}`;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0, canvas.width, canvas.height);
    setTimeout(() => {
      Charts.drawComparison(canvas, pA, pB);
    }, 50);
  }

  selectA.addEventListener('change', updateH2H);
  selectB.addEventListener('change', updateH2H);
  
  // Initial draw
  updateH2H();
}

/* ── Scoring Profile Switcher ── */
function initProfileSwitcher() {
  const cards = document.querySelectorAll('.profile-card');
  if (!cards.length) return;

  cards.forEach(card => {
    card.addEventListener('click', () => {
      cards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      renderProfileWeights(card.dataset.profile);
    });
  });

  renderProfileWeights('balanced');
}

function renderProfileWeights(profile) {
  const container = document.getElementById('profile-weights');
  if (!container) return;

  container.innerHTML = '';

  AXES_ORDER.forEach((axis) => {
    const meta = AXES_META[axis];
    const weight = meta.weight[profile] || 0;
    const maxWeight = Math.max(...AXES_ORDER.map(a => AXES_META[a].weight[profile] || 0));
    const pct = (weight / maxWeight) * 100;

    const el = document.createElement('div');
    el.className = 'weight-bar';
    el.innerHTML = `
      <div class="weight-bar-label">
        <span>${axis.charAt(0).toUpperCase() + axis.slice(1)}</span>
        <span>${weight}%</span>
      </div>
      <div class="weight-bar-track">
        <div class="weight-bar-fill" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(el);
  });
}

/* ── Stat Counters ── */
function initCounters() {
  loadRegistryStats();
}

async function loadRegistryStats() {
  try {
    const resp = await fetch('../results/challenge_registry.json');
    if (!resp.ok) return;
    const data = await resp.json();

    let uniqueModels = new Set();
    let totalChallenges = 0;
    
    for (const [sessionId, evals] of Object.entries(data.evaluations || {})) {
      for (const ev of evals) {
        uniqueModels.add(ev.model_id);
        totalChallenges += ev.summary?.n_challenges || 0;
      }
    }

    animateCounter('stat-models', uniqueModels.size);
    animateCounter('stat-challenges', totalChallenges);
  } catch {
    // Registry doesn't exist yet — show zeros
  }
}

function animateCounter(elementId, target) {
  const el = document.querySelector(`#${elementId} .counter`);
  if (!el) return;

  el.setAttribute('data-target', target);

  let current = 0;
  const duration = 1500;
  const step = target / (duration / 16);
  const start = performance.now();

  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(1, elapsed / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = Math.round(target * eased);
    el.textContent = current.toLocaleString();
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

/* ── Leaderboard Data Loading ── */
async function loadLeaderboardData() {
  const tbody = document.getElementById('leaderboard-body');
  if (!tbody) return;

  try {
    const resp = await fetch('../results/challenge_registry.json');
    if (!resp.ok) throw new Error('No registry');
    const registry = await resp.json();

    // Get best results per model
    const models = {};
    for (const [sessionId, evals] of Object.entries(registry.evaluations || {})) {
      for (const ev of evals) {
        const mid = ev.model_id;
        const score = ev.summary?.mean_overall || 0;
        if (!models[mid] || score > models[mid].score) {
          models[mid] = { model_id: mid, score, accuracy: ev.summary?.accuracy || 0, cost: ev.summary?.total_cost_usd || 0, session_id: sessionId };
        }
      }
    }

    const rows = Object.values(models).sort((a, b) => b.score - a.score);

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:40px;">No models evaluated yet. Run your first evaluation to see results here.</td></tr>';
      return;
    }

    rows.forEach((row, idx) => {
      const rank = idx + 1;
      const rankClass = rank <= 3 ? `rank-${rank}` : '';

      // Try to load full profile for radar chart
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="rank-badge ${rankClass}">${rank}</span></td>
        <td><strong>${row.model_id}</strong></td>
        <td class="score-cell ${row.score >= 0.7 ? 'score-high' : row.score >= 0.4 ? 'score-mid' : 'score-low'}">${(row.score * 100).toFixed(1)}%</td>
        <td class="score-cell">${(row.accuracy * 100).toFixed(1)}%</td>
        <td>$${row.cost.toFixed(4)}</td>
        <td class="mini-radar-cell"><canvas width="60" height="60"></canvas></td>
      `;
      tbody.appendChild(tr);

      // Load and render mini radar
      loadModelProfile(row.model_id).then(profile => {
        if (profile) {
          const miniCanvas = tr.querySelector('canvas');
          Charts.drawMiniRadar(miniCanvas, profile.scores_array || [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        }
      });
    });
  } catch {
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-dim);padding:40px;">No evaluation data available. Run <code>python main.py --evaluate-model --provider mock --model mock-v1</code> to get started.</td></tr>';
    }
  }
}

async function loadModelProfile(modelId) {
  // Try to load the full eval JSON
  const safeName = modelId.replace(/\//g, '_');
  const parts = safeName.split('_');
  const provider = parts[0];
  const model = parts.slice(1).join('_');

  try {
    const resp = await fetch(`../results/EVAL_${provider}_${model}.json`);
    if (!resp.ok) return null;
    const data = await resp.json();
    return data.aggregate_profile || null;
  } catch {
    return null;
  }
}

/* ── Submit Form ── */
function initSubmitForm() {
  const form = document.getElementById('submit-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const provider = form.querySelector('#provider').value;
    const model = form.querySelector('#model-name').value;
    const challenges = form.querySelector('#n-challenges').value || '20';
    const profile = form.querySelector('#scoring-profile').value || 'balanced';

    const resultDiv = document.getElementById('submit-result');
    if (resultDiv) {
      resultDiv.innerHTML = `
        <div style="padding: 20px; background: rgba(132, 94, 247, 0.08); border: 1px solid rgba(132, 94, 247, 0.3); border-radius: 12px; margin-top: 20px;">
          <h3 style="margin-bottom: 8px;">Run this command:</h3>
          <code style="display:block;padding:14px;background:rgba(0,0,0,0.4);border-radius:8px;font-family:var(--font-mono);color:#22c55e;font-size:0.88rem;word-break:break-all;">python main.py --evaluate-model --provider ${provider} --model ${model} --eval-challenges ${challenges} --scoring-profile ${profile}</code>
          <p style="color:var(--text-dim);margin-top:12px;font-size:0.88rem;">After running, refresh the leaderboard to see results.</p>
        </div>
      `;
    }
  });
}
