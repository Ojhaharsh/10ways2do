const phaseProfiles = {
  foundations: {
    compareTitle: 'Structure, latency, and label fidelity all matter here.',
    compareIntro:
      'Foundational domains reward methods that recover signal from noisy inputs without becoming opaque or slow.',
    readingOrder:
      'Start with the domain data generator, then inspect the orchestration file, and finally compare the strongest approach family against the simpler baselines.',
    familyPills: ['Rules', 'Classical ML', 'Sequence models', 'Transformers', 'Hybrid systems'],
    familyNotes: ['Baseline anchor', 'Classical reference', 'Sequence family', 'Transformer family', 'Hybrid finish'],
    approachLadder: [
      { title: 'Baseline extractor', note: 'Fast, transparent, and easy to debug.' },
      { title: 'Classical learner', note: 'Adds supervised signal while staying simple.' },
      { title: 'Tree ensemble', note: 'Good for tabular or feature-engineered variants.' },
      { title: 'Recurrent model', note: 'Useful when sequence context matters.' },
      { title: 'Convolutional sequence model', note: 'Captures local patterns in text spans.' },
      { title: 'Transformer scratch', note: 'Learns contextual relationships from data.' },
      { title: 'Pretrained backbone', note: 'Transfers outside knowledge into the domain.' },
      { title: 'Prompted system', note: 'Uses prompting or instruction tuning.' },
      { title: 'Hybrid stack', note: 'Combines rules with learned components.' },
      { title: 'Operational wrapper', note: 'Focuses on deployment discipline and guardrails.' },
    ],
  },
  systems: {
    compareTitle: 'Operational utility matters as much as raw predictive power.',
    compareIntro:
      'Systems domains reward approaches that keep false alarms low, preserve cost discipline, and remain stable under changing load.',
    readingOrder:
      'Read the data generator first, then the run harness, and then compare the model family with the strongest system-aware baseline.',
    familyPills: ['Heuristics', 'Tree models', 'Deep systems', 'Ensembles', 'Governance-aware hybrids'],
    familyNotes: ['Operational baseline', 'Risk-aware model', 'Production reference', 'Ensemble layer', 'Governance hybrid'],
    approachLadder: [
      { title: 'Heuristic baseline', note: 'Quick reference point for operational trade-offs.' },
      { title: 'Classical control model', note: 'Uses clear feature-driven decisioning.' },
      { title: 'Tree-based policy', note: 'Strong on structured operational data.' },
      { title: 'Deep signal model', note: 'Learns more expressive patterns from telemetry.' },
      { title: 'Temporal model', note: 'Tracks sequences and changing load over time.' },
      { title: 'Ensemble system', note: 'Mixes signals for better stability.' },
      { title: 'Cost-aware variant', note: 'Balances accuracy with budget or latency.' },
      { title: 'Governance-aware hybrid', note: 'Reflects monitoring and policy constraints.' },
      { title: 'Resilience layer', note: 'Handles shifts, drift, and noisy inputs.' },
      { title: 'Production guardrail', note: 'Makes the method usable at scale.' },
    ],
  },
  frontier: {
    compareTitle: 'Transfer, adaptation, and scale are the point of the exercise.',
    compareIntro:
      'Frontier domains reward methods that adapt quickly, scale cleanly, and still produce interpretable benchmark outputs.',
    readingOrder:
      'Look at the synthetic task setup, then inspect the orchestration layer, and then compare modern architectures against leaner baselines.',
    familyPills: ['Baselines', 'Pretrained systems', 'Specialized deep models', 'Adaptation layers', 'Modern hybrids'],
    familyNotes: ['Small-data baseline', 'Pretrained backbone', 'Specialized model', 'Adaptation layer', 'Fusion or policy head'],
    approachLadder: [
      { title: 'Scratch baseline', note: 'Shows what the task looks like without transfer.' },
      { title: 'Pretrained backbone', note: 'Brings in external representations.' },
      { title: 'Specialized model', note: 'Uses a domain-shaped architecture.' },
      { title: 'Adapter or fine-tune', note: 'Keeps most knowledge while adapting quickly.' },
      { title: 'Sequence-aware variant', note: 'Models order and temporal structure.' },
      { title: 'Ensemble or fusion', note: 'Combines complementary signals.' },
      { title: 'Modern large model', note: 'Represents the frontier family directly.' },
      { title: 'Efficiency variant', note: 'Optimizes memory, cost, or latency.' },
      { title: 'Cross-modal or policy head', note: 'Adds task-specific adaptation.' },
      { title: 'System-level wrapper', note: 'Shows how the model is used in practice.' },
    ],
  },
};

const domainCatalog = {
  a: {
    code: 'A',
    name: 'Information Extraction',
    phase: 'foundations',
    path: 'src/domain_a_information_extraction/',
    summary: 'Structured fact extraction from text and documents.',
    focus: 'Extract the right fields from messy text before downstream systems amplify the error.',
    tagline: 'NLP task extraction with token-level and field-level scoring.',
    metrics: ['Exact match', 'Partial match', 'Field-level F1'],
    compare: [
      'Rules vs learned sequence models vs prompt-driven extraction',
      'Exact match quality vs partial match tolerance',
      'Accuracy vs latency when text gets noisy',
    ],
    families: ['Rules', 'CRF-style ML', 'Tree-based', 'RNN/LSTM', 'Transformer scratch', 'Prompt LLM', 'Hybrid'],
  },
  b: {
    code: 'B',
    name: 'Anomaly Detection',
    phase: 'foundations',
    path: 'src/domain_b_anomaly_detection/',
    summary: 'Outlier detection in high-dimensional data and systems logs.',
    focus: 'Catch rare anomalies without flooding analysts with false positives.',
    tagline: 'Anomaly scoring where recall and precision both matter.',
    metrics: ['AUROC', 'Precision@k', 'False alarm rate'],
    compare: [
      'Statistical baselines vs learned detectors',
      'Point anomalies vs clustered or seasonal anomalies',
      'Detection quality vs false alarm pressure',
    ],
    families: ['Statistical', 'Distance-based', 'Tree-based', 'Autoencoder', 'LSTM', 'Transformer', 'Ensemble', 'Hybrid'],
  },
  c: {
    code: 'C',
    name: 'Recommendation',
    phase: 'foundations',
    path: 'src/domain_c_recommendation/',
    summary: 'Personalization, ranking, and user-item preference modeling.',
    focus: 'Rank items people actually want while keeping the system diverse and stable.',
    tagline: 'Recommendation quality with ranking-aware evaluation.',
    metrics: ['Recall@k', 'NDCG', 'MAP'],
    compare: [
      'Popularity vs personalization',
      'Matrix factorization vs deep recommenders',
      'Ranking quality vs coverage and diversity',
    ],
    families: ['Popularity', 'Collaborative filtering', 'Content-based', 'Matrix factorization', 'Deep learning', 'Sequence-based', 'Graph', 'Hybrid'],
  },
  d: {
    code: 'D',
    name: 'Time Series Forecasting',
    phase: 'foundations',
    path: 'src/domain_d_time_series/',
    summary: 'Temporal prediction with classical and neural methods.',
    focus: 'Forecast the next step from noisy history without overfitting the past.',
    tagline: 'Temporal forecasting with trend, seasonality, and drift.',
    metrics: ['MAE', 'MAPE', 'sMAPE'],
    compare: [
      'Seasonality vs trend drift',
      'Statistical smoothing vs neural forecasters',
      'Accuracy vs calibration of the forecast band',
    ],
    families: ['Statistical', 'Exponential smoothing', 'Tree-based', 'RNN/LSTM', 'CNN temporal', 'Transformer', 'Neural Prophet', 'Hybrid'],
  },
  e: {
    code: 'E',
    name: 'Tabular Decisioning',
    phase: 'systems',
    path: 'src/domain_e_tabular_decisioning/',
    summary: 'General-purpose structured-data classification and ranking.',
    focus: 'Make reliable decisions from business records without hiding the reasoning.',
    tagline: 'Tabular decisioning where calibration and explainability matter.',
    metrics: ['Accuracy', 'AUC', 'Calibration'],
    compare: [
      'Interpretability vs raw accuracy',
      'Linear models vs trees vs MLPs',
      'Operational risk vs model complexity',
    ],
    families: ['Linear', 'Tree boosting', 'Random forest', 'MLP', 'Ensemble', 'Hybrid systems'],
  },
  f: {
    code: 'F',
    name: 'Cyber Threat Hunting',
    phase: 'systems',
    path: 'src/domain_f_cyber_threat_hunting/',
    summary: 'Security telemetry, signal extraction, and suspicious pattern detection.',
    focus: 'Surface malicious behavior early enough to matter to analysts and responders.',
    tagline: 'Detection under noisy security telemetry.',
    metrics: ['Precision', 'Recall', 'Alert latency'],
    compare: [
      'Heuristic detectors vs learned anomaly models',
      'Recall vs false-positive pressure',
      'Static signals vs evolving attack patterns',
    ],
    families: ['Rule-based', 'Statistical', 'Tree-based', 'Autoencoder', 'LSTM', 'Transformer', 'Ensemble', 'Hybrid'],
  },
  g: {
    code: 'G',
    name: 'Operations Optimization',
    phase: 'systems',
    path: 'src/domain_g_operations_optimization/',
    summary: 'Resource allocation, planning, and cost-aware optimization.',
    focus: 'Use limited resources where they create the most value.',
    tagline: 'Optimization decisions with explicit utility and cost.',
    metrics: ['Utility', 'Cost', 'Latency'],
    compare: [
      'Operational utility vs total cost',
      'Exact optimization vs heuristic search',
      'Hard constraints vs flexible approximations',
    ],
    families: ['Heuristics', 'Linear programming', 'Tree-based', 'Deep optimization', 'Ensemble', 'Hybrid'],
  },
  h: {
    code: 'H',
    name: 'Fraud Risk Assessment',
    phase: 'systems',
    path: 'src/domain_h_fraud_risk_assessment/',
    summary: 'Financial risk scoring and fraud pattern discovery.',
    focus: 'Catch fraud while keeping review queues manageable.',
    tagline: 'Risk scoring where false positives are expensive.',
    metrics: ['AUC', 'Recall', 'False positive rate'],
    compare: [
      'Fraud catch rate vs manual review load',
      'Cost-sensitive models vs generic classifiers',
      'Short-term signal vs shifting fraud patterns',
    ],
    families: ['Logistic', 'Gradient boosting', 'Tree-based', 'Deep learning', 'Cost-sensitive', 'Hybrid'],
  },
  i: {
    code: 'I',
    name: 'Capacity Planning',
    phase: 'systems',
    path: 'src/domain_i_capacity_planning/',
    summary: 'Predicting future load and provisioning requirements.',
    focus: 'Forecast demand early enough to keep systems available without overprovisioning.',
    tagline: 'Capacity forecasting with stability under spikes.',
    metrics: ['MAPE', 'Bias', 'Forecast stability'],
    compare: [
      'Forecast bias vs variance',
      'Short-term spikes vs long-horizon load',
      'Stability vs responsiveness',
    ],
    families: ['Statistical', 'Exponential smoothing', 'Tree-based', 'LSTM', 'Transformer', 'Ensemble'],
  },
  j: {
    code: 'J',
    name: 'Model Risk Monitoring',
    phase: 'systems',
    path: 'src/domain_j_model_risk_monitoring/',
    summary: 'Drift, performance decay, and governance monitoring.',
    focus: 'Detect degradation early enough to act before the model goes stale.',
    tagline: 'Monitoring for drift, decay, and governance risk.',
    metrics: ['Drift score', 'Alert lead time', 'Recovery time'],
    compare: [
      'Alert latency vs drift sensitivity',
      'Threshold rules vs learned monitors',
      'Precision of alerts vs noise in the stream',
    ],
    families: ['Drift rules', 'Statistical monitors', 'Tree-based', 'Autoencoders', 'Transformers', 'Ensemble'],
  },
  k: {
    code: 'K',
    name: 'Infrastructure Cost Forecasting',
    phase: 'systems',
    path: 'src/domain_k_infrastructure_cost_forecasting/',
    summary: 'Cloud and infrastructure spend prediction and optimization.',
    focus: 'Predict budget drift before the bill arrives.',
    tagline: 'Spend forecasting with explicit overrun risk.',
    metrics: ['Cost error', 'Overspend risk', 'Forecast bias'],
    compare: [
      'Budget accuracy vs overspend risk',
      'Short-term spend vs long-horizon planning',
      'Stability under changing usage patterns',
    ],
    families: ['Statistical', 'Tree-based', 'Forecasting', 'Neural nets', 'Ensemble', 'Hybrid'],
  },
  l: {
    code: 'L',
    name: 'Computer Vision',
    phase: 'frontier',
    path: 'src/domain_l_computer_vision/',
    summary: 'Image classification with CNNs, ViTs, ensembles, and self-supervision.',
    focus: 'Compare image classifiers on accuracy, compute, and generalization.',
    tagline: 'Vision models from scratch, transfer learning, and NAS.',
    metrics: ['Top-1 acc', 'Top-5 acc', 'FLOPs'],
    compare: [
      'Top-1 vs top-5 accuracy',
      'Edge efficiency vs model capacity',
      'Transfer learning vs training from scratch',
    ],
    families: ['CNN', 'VGG-style', 'ResNet-style', 'EfficientNet', 'ViT', 'MobileNet', 'Ensemble', 'NAS'],
  },
  m: {
    code: 'M',
    name: 'Graph Neural Networks',
    phase: 'frontier',
    path: 'src/domain_m_graph_neural_networks/',
    summary: 'Graph representation learning and link prediction.',
    focus: 'Measure whether graph models actually learn structure instead of shortcuts.',
    tagline: 'Graph learning with link prediction and message passing.',
    metrics: ['Link AUC', 'Hits@K', 'Latency'],
    compare: [
      'Link AUC vs hits@K',
      'Local neighborhoods vs global graph structure',
      'Scalability vs expressiveness',
    ],
    families: ['GCN', 'GraphSAGE', 'GAT', 'ChebNet', 'GIN', 'Similarity', 'SkipGram', 'Ensemble'],
  },
  n: {
    code: 'N',
    name: 'Few-Shot Learning',
    phase: 'frontier',
    path: 'src/domain_n_few_shot_learning/',
    summary: 'Meta-learning for extremely small labeled datasets.',
    focus: 'Test how quickly a method adapts when the support set is tiny.',
    tagline: 'Adaptation quality under severe data scarcity.',
    metrics: ['Few-shot acc', 'Adaptation steps', 'Stability'],
    compare: [
      'Support-set size vs adaptation quality',
      'Fast adaptation vs stable generalization',
      'Task transfer vs specialization',
    ],
    families: ['Prototypical', 'Matching', 'Relation', 'MAML', 'Transductive', 'Optimal Transport', 'Ensemble'],
  },
  o: {
    code: 'O',
    name: 'Reinforcement Learning',
    phase: 'frontier',
    path: 'src/domain_o_reinforcement_learning/',
    summary: 'Bandits and online adaptation under reward feedback.',
    focus: 'Judge policies by reward, regret, and sample efficiency.',
    tagline: 'Online decision making with exploration and exploitation.',
    metrics: ['Reward', 'Regret', 'Sample efficiency'],
    compare: [
      'Reward vs regret',
      'Exploration vs exploitation',
      'Sample efficiency vs convergence speed',
    ],
    families: ['Epsilon-Greedy', 'UCB', 'Thompson', 'LinUCB', 'Contextual', 'Neural', 'Meta-bandit'],
  },
  p: {
    code: 'P',
    name: 'Multimodal Learning',
    phase: 'frontier',
    path: 'src/domain_p_multimodal_learning/',
    summary: 'Vision-language systems and cross-modal alignment.',
    focus: 'Align images and text without losing either signal.',
    tagline: 'Vision-language systems and cross-modal fusion.',
    metrics: ['Alignment', 'Retrieval score', 'Multimodal accuracy'],
    compare: [
      'Alignment vs retrieval quality',
      'Image signal vs text signal',
      'Single-modal baselines vs fused systems',
    ],
    families: ['CLIP', 'VisualBERT', 'ALBEF', 'Flamingo', 'LLaVA', 'BLIPv2', 'Cross-modal', 'Adaptive'],
  },
};

const domainOrder = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'];

function getDomainKey() {
  const params = new URLSearchParams(window.location.search);
  const key = (params.get('domain') || 'a').toLowerCase();
  return domainCatalog[key] ? key : 'a';
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) {
    node.textContent = value;
  }
}

function setLink(id, href) {
  const node = document.getElementById(id);
  if (node) {
    node.href = href;
  }
}

function renderList(containerId, items, renderer) {
  const node = document.getElementById(containerId);
  if (!node) {
    return;
  }
  node.innerHTML = items.map(renderer).join('');
}

function renderDomainPage() {
  const key = getDomainKey();
  const domain = domainCatalog[key];
  const profile = phaseProfiles[domain.phase];
  const index = domainOrder.indexOf(key);
  const prevKey = domainOrder[(index - 1 + domainOrder.length) % domainOrder.length];
  const nextKey = domainOrder[(index + 1) % domainOrder.length];

  document.title = `${domain.name} • 10ways2do`;

  setText('domain-phase', `${domain.code} • ${domain.phase.toUpperCase()}`);
  setText('domain-title', domain.name);
  setText('domain-summary', domain.summary);
  setText('domain-focus', domain.focus);
  setText('domain-code-badge', `DOMAIN ${domain.code}`);
  setText('domain-approach-count', '10 approaches');
  setText('domain-tagline', domain.tagline);
  setText('domain-metric-1', domain.metrics[0]);
  setText('domain-metric-2', domain.metrics[1]);
  setText('domain-metric-3', domain.metrics[2]);
  setText('domain-folder-line', `Folder: ${domain.path}`);
  setText('compare-title', profile.compareTitle);
  setText('compare-intro', profile.compareIntro);
  setText('reading-order', profile.readingOrder);
  setLink('open-folder-link', `../${domain.path}`);

  renderList('compare-grid', domain.compare, (item, index) => `
    <article class="compare-card">
      <span>${String(index + 1).padStart(2, '0')}</span>
      <p>${item}</p>
    </article>
  `);

  renderList('family-grid', domain.families, (item, index) => `
    <article class="family-card">
      <span>${String(index + 1).padStart(2, '0')}</span>
      <strong>${item}</strong>
      <small>${profile.familyNotes[index % profile.familyNotes.length]}</small>
    </article>
  `);

  renderList('approach-grid', profile.approachLadder, (item, index) => `
    <article class="approach-card">
      <span>${String(index + 1).padStart(2, '0')}</span>
      <strong>${item.title}</strong>
      <p>${item.note}</p>
      <small>${profile.familyPills[index % profile.familyPills.length]}</small>
    </article>
  `);

  renderList('file-grid', [
    { label: 'Run orchestration', href: `../${domain.path}run_all.py`, note: 'Entry point for the full domain sweep.' },
    { label: 'Data generator', href: `../${domain.path}data_generator.py`, note: 'Synthetic data and task setup.' },
    { label: 'Approach folder', href: `../${domain.path}`, note: 'Ten competing methods for this domain.' },
    { label: 'Benchmark report', href: '../results/REPORT.md', note: 'The cross-domain narrative and release outputs.' },
  ], (item) => `
    <a class="file-card" href="${item.href}">
      <span class="file-label">${item.label}</span>
      <strong>${item.href}</strong>
      <p>${item.note}</p>
    </a>
  `);

  renderList('neighbor-grid', [prevKey, 'atlas', nextKey], (item) => {
    if (item === 'atlas') {
      return `
        <a class="neighbor-card" href="index.html#collection">
          <span>Atlas</span>
          <strong>Return to collection</strong>
          <p>Browse the full set of domains again.</p>
        </a>
      `;
    }

    const neighbor = domainCatalog[item];
    return `
      <a class="neighbor-card" href="domain.html?domain=${neighbor.code.toLowerCase()}">
        <span>DOMAIN ${neighbor.code}</span>
        <strong>${neighbor.name}</strong>
        <p>${neighbor.summary}</p>
      </a>
    `;
  });

  const resultsLink = document.getElementById('results-link');
  if (resultsLink) {
    resultsLink.href = '../results/REPORT.md';
  }
}

renderDomainPage();
