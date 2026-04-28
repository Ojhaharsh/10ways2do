let benchmarkSnapshot = window.__BENCHMARK_SNAPSHOT__ || {
  generatedAt: '2026-04-23T17:41:30.581908+00:00',
  source: 'results/BENCHMARK_CARD.json',
  coverage: {
    expected: 11,
    observed: 5,
    missingInFrontier: ['domain_f', 'domain_g', 'domain_h', 'domain_i', 'domain_j', 'domain_k'],
    missingManifest: ['domain_h', 'domain_i', 'domain_j', 'domain_k'],
  },
  protocolVersions: [
    { version: '1.0.0', count: 7 },
  ],
  commitHashes: 4,
  weights: {
    quality: 0.45,
    speed: 0.25,
    resilience: 0.2,
    consistency: 0.1,
  },
  champions: [
    { domain: 'domain_a', domainName: 'Information Extraction', name: 'Prompt-Based LLM', score: 0.92989 },
    { domain: 'domain_b', domainName: 'Anomaly Detection', name: 'Statistical (MAD)', score: 0.875 },
    { domain: 'domain_c', domainName: 'Recommendation', name: 'Matrix Factorization (SVD)', score: 0.875 },
    { domain: 'domain_d', domainName: 'Time Series Forecasting', name: 'Exponential Smoothing', score: 0.875 },
    { domain: 'domain_e', domainName: 'Tabular Decisioning', name: 'Linear (Logistic Regression)', score: 0.875 },
  ],
  topGeneralists: [
    { name: 'Prompt-Based LLM', score: 0.92989, domainsCovered: 1 },
    { name: 'Rule-Based IE', score: 0.905466, domainsCovered: 1 },
    { name: 'Statistical (MAD)', score: 0.875, domainsCovered: 1 },
    { name: 'Matrix Factorization (SVD)', score: 0.875, domainsCovered: 1 },
    { name: 'Exponential Smoothing', score: 0.875, domainsCovered: 1 },
  ],
};

const domains = [
  {
    code: 'A',
    name: 'Information Extraction',
    summary: 'Structured fact extraction from text and documents.',
    tags: ['NLP', 'transformers', 'text-to-structure'],
    group: 'foundations',
    groupLabel: 'Foundations',
    path: 'src/domain_a_information_extraction/'
  },
  {
    code: 'B',
    name: 'Anomaly Detection',
    summary: 'Outlier detection in high-dimensional data and systems logs.',
    tags: ['unsupervised', 'distance-based', 'tree-based'],
    group: 'foundations',
    groupLabel: 'Foundations',
    path: 'src/domain_b_anomaly_detection/'
  },
  {
    code: 'C',
    name: 'Recommendation',
    summary: 'Personalization, ranking, and user-item preference modeling.',
    tags: ['collaborative', 'content-based', 'hybrid'],
    group: 'foundations',
    groupLabel: 'Foundations',
    path: 'src/domain_c_recommendation/'
  },
  {
    code: 'D',
    name: 'Time Series Forecasting',
    summary: 'Temporal prediction with classical and neural methods.',
    tags: ['forecasting', 'seasonality', 'transformers'],
    group: 'foundations',
    groupLabel: 'Foundations',
    path: 'src/domain_d_time_series/'
  },
  {
    code: 'E',
    name: 'Tabular Decisioning',
    summary: 'General-purpose structured-data classification and ranking.',
    tags: ['tabular', 'tree models', 'mlp'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_e_tabular_decisioning/'
  },
  {
    code: 'F',
    name: 'Cyber Threat Hunting',
    summary: 'Security telemetry, signal extraction, and suspicious pattern detection.',
    tags: ['security', 'detection', 'ops'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_f_cyber_threat_hunting/'
  },
  {
    code: 'G',
    name: 'Operations Optimization',
    summary: 'Resource allocation, planning, and cost-aware optimization.',
    tags: ['optimization', 'business', 'decisioning'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_g_operations_optimization/'
  },
  {
    code: 'H',
    name: 'Fraud Risk Assessment',
    summary: 'Financial risk scoring and fraud pattern discovery.',
    tags: ['fraud', 'finance', 'classification'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_h_fraud_risk_assessment/'
  },
  {
    code: 'I',
    name: 'Capacity Planning',
    summary: 'Predicting future load and provisioning requirements.',
    tags: ['operations', 'forecasting', 'planning'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_i_capacity_planning/'
  },
  {
    code: 'J',
    name: 'Model Risk Monitoring',
    summary: 'Drift, performance decay, and governance monitoring.',
    tags: ['mlops', 'monitoring', 'governance'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_j_model_risk_monitoring/'
  },
  {
    code: 'K',
    name: 'Infrastructure Cost Forecasting',
    summary: 'Cloud and infrastructure spend prediction and optimization.',
    tags: ['cloud', 'cost', 'forecasting'],
    group: 'systems',
    groupLabel: 'Systems',
    path: 'src/domain_k_infrastructure_cost_forecasting/'
  },
  {
    code: 'L',
    name: 'Computer Vision',
    summary: 'Image classification with CNNs, ViTs, ensembles, and self-supervision.',
    tags: ['vision', 'cnn', 'vit'],
    group: 'frontier',
    groupLabel: 'Frontier',
    path: 'src/domain_l_computer_vision/'
  },
  {
    code: 'M',
    name: 'Graph Neural Networks',
    summary: 'Graph representation learning and link prediction.',
    tags: ['graphs', 'gcn', 'gat'],
    group: 'frontier',
    groupLabel: 'Frontier',
    path: 'src/domain_m_graph_neural_networks/'
  },
  {
    code: 'N',
    name: 'Few-Shot Learning',
    summary: 'Meta-learning for extremely small labeled datasets.',
    tags: ['meta-learning', 'maml', 'prototypical'],
    group: 'frontier',
    groupLabel: 'Frontier',
    path: 'src/domain_n_few_shot_learning/'
  },
  {
    code: 'O',
    name: 'Reinforcement Learning',
    summary: 'Bandits and online adaptation under reward feedback.',
    tags: ['rl', 'bandits', 'online learning'],
    group: 'frontier',
    groupLabel: 'Frontier',
    path: 'src/domain_o_reinforcement_learning/'
  },
  {
    code: 'P',
    name: 'Multimodal Learning',
    summary: 'Vision-language systems and cross-modal alignment.',
    tags: ['multimodal', 'clip', 'llava'],
    group: 'frontier',
    groupLabel: 'Frontier',
    path: 'src/domain_p_multimodal_learning/'
  }
];

const grid = document.getElementById('domain-grid');
const countLabel = document.getElementById('visible-count');
const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
const resultsMetrics = document.getElementById('results-metrics');
const generalistList = document.getElementById('generalist-list');
const championGrid = document.getElementById('champion-grid');
const coverageSummary = document.getElementById('coverage-summary');
const coverageList = document.getElementById('coverage-list');
const snapshotSource = document.getElementById('snapshot-source');
const snapshotGenerated = document.getElementById('snapshot-generated');

function formatScore(value) {
  return Number.parseFloat(value.toFixed(5)).toString();
}

function renderBenchmarkSnapshot() {
  if (snapshotSource) {
    snapshotSource.textContent = `Source: ${benchmarkSnapshot.source}`;
  }

  if (snapshotGenerated) {
    snapshotGenerated.textContent = `Generated: ${benchmarkSnapshot.generatedAt.replace('T', ' ').replace('+00:00', ' UTC')}`;
  }

  if (resultsMetrics) {
    const protocolSummary = benchmarkSnapshot.protocolVersions
      .map((item) => `${item.version} × ${item.count}`)
      .join(', ');

    resultsMetrics.innerHTML = [
      {
        label: 'Expected domains',
        value: benchmarkSnapshot.coverage.expected,
        detail: 'Original release scope',
      },
      {
        label: 'Observed domains',
        value: benchmarkSnapshot.coverage.observed,
        detail: `${Math.round((benchmarkSnapshot.coverage.observed / benchmarkSnapshot.coverage.expected) * 100)}% of expected`,
      },
      {
        label: 'Protocol versions',
        value: protocolSummary,
        detail: 'Release gate verified',
      },
      {
        label: 'Unique commits',
        value: benchmarkSnapshot.commitHashes,
        detail: 'Tracked in the card bundle',
      },
      {
        label: 'Weights',
        value: '45 / 25 / 20 / 10',
        detail: 'Quality, speed, resilience, consistency',
      },
    ]
      .map(
        (item) => `
          <article class="result-metric">
            <span>${item.label}</span>
            <strong>${item.value}</strong>
            <small>${item.detail}</small>
          </article>
        `,
      )
      .join('');
  }

  if (coverageSummary) {
    const missingFrontier = benchmarkSnapshot.coverage.missingInFrontier.join(', ');
    const missingManifest = benchmarkSnapshot.coverage.missingManifest.join(', ');
    coverageSummary.textContent = `Expected ${benchmarkSnapshot.coverage.expected} domains, observed ${benchmarkSnapshot.coverage.observed}. Frontier gaps: ${missingFrontier}. Manifest gaps: ${missingManifest}.`;
  }

  if (coverageList) {
    coverageList.innerHTML = [
      ...benchmarkSnapshot.coverage.missingInFrontier.map((domain) => ({ label: domain, kind: 'frontier gap' })),
      ...benchmarkSnapshot.coverage.missingManifest.map((domain) => ({ label: domain, kind: 'manifest gap' })),
    ]
      .map(
        (item) => `
          <span class="coverage-chip">
            <strong>${item.label}</strong>
            <small>${item.kind}</small>
          </span>
        `,
      )
      .join('');
  }

  if (generalistList) {
    generalistList.innerHTML = benchmarkSnapshot.topGeneralists
      .map(
        (item) => `
          <article class="generalist-item">
            <div>
              <strong>${item.name}</strong>
              <p>${item.domainsCovered} domain${item.domainsCovered === 1 ? '' : 's'} covered</p>
            </div>
            <div class="generalist-score">${formatScore(item.score)}</div>
          </article>
        `,
      )
      .join('');
  }

  if (championGrid) {
    championGrid.innerHTML = benchmarkSnapshot.champions
      .map(
        (item) => `
          <article class="champion-card">
            <span class="rank">${item.domain}</span>
            <h4>${item.name}</h4>
            <p>${item.domainName}</p>
            <div class="score">${formatScore(item.score)}</div>
          </article>
        `,
      )
      .join('');
  }
}

function createDomainCard(domain) {
  const card = document.createElement('a');
  card.className = 'domain-card';
  card.href = `domain.html?domain=${domain.code.toLowerCase()}`;
  card.setAttribute('aria-label', `Open the ${domain.name} domain page`);
  card.dataset.group = domain.group;

  const tags = domain.tags.map((tag) => `<span class="tag">${tag}</span>`).join('');

  card.innerHTML = `
    <div class="domain-top">
      <span class="domain-code">DOMAIN ${domain.code}</span>
      <span class="domain-phase">${domain.groupLabel}</span>
    </div>
    <h3>${domain.name}</h3>
    <p>${domain.summary}</p>
    <div class="domain-meta">
      <span>10 approaches</span>
      <span>${domain.path}</span>
    </div>
    <div class="tag-list">${tags}</div>
    <div class="card-action">Open domain page</div>
  `;

  return card;
}

function updateVisibleCount() {
  if (!countLabel) {
    return;
  }

  const visible = grid ? grid.querySelectorAll('.domain-card:not([hidden])').length : domains.length;
  countLabel.textContent = `${visible} visible`;
}

function setActiveFilter(activeFilter) {
  filterButtons.forEach((button) => {
    const isActive = button.dataset.filter === activeFilter;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function applyFilter(filter) {
  if (!grid) {
    return;
  }

  grid.querySelectorAll('.domain-card').forEach((card) => {
    const matches = filter === 'all' || card.dataset.group === filter;
    card.hidden = !matches;
  });

  setActiveFilter(filter);
  updateVisibleCount();
}

function renderDomains() {
  if (!grid) {
    return;
  }

  domains.forEach((domain) => {
    grid.appendChild(createDomainCard(domain));
  });

  updateVisibleCount();
}

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    applyFilter(button.dataset.filter || 'all');
  });
});

renderDomains();
renderBenchmarkSnapshot();

window.addEventListener('benchmark-snapshot-ready', () => {
  benchmarkSnapshot = window.__BENCHMARK_SNAPSHOT__ || benchmarkSnapshot;
  renderBenchmarkSnapshot();
});
