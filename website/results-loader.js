(function () {
  const snapshotUrl = '../results/BENCHMARK_CARD.json';

  function normalizeSnapshot(payload) {
    const coverage = payload.domain_coverage || {};
    const protocolVersions = Object.entries(payload.protocol_versions || {}).map(([version, count]) => ({
      version,
      count,
    }));

    return {
      generatedAt: payload.generated_at_utc || '',
      source: snapshotUrl,
      coverage: {
        expected: coverage.expected || 0,
        observed: coverage.observed || 0,
        missingInFrontier: Array.isArray(coverage.missing_in_frontier) ? coverage.missing_in_frontier : [],
        missingManifest: Array.isArray(coverage.missing_manifest) ? coverage.missing_manifest : [],
      },
      protocolVersions,
      commitHashes: Array.isArray(payload.git_commit_hashes) ? payload.git_commit_hashes.length : 0,
      weights: payload.weights || {},
      champions: Array.isArray(payload.champions)
        ? payload.champions.map((row) => ({
            domain: row.domain || 'unknown',
            domainName: row.domain_name || row.domain || 'unknown',
            name: row.champion || 'unknown',
            score: row.extraordinary_index || 0,
          }))
        : [],
      topGeneralists: Array.isArray(payload.top_generalists)
        ? payload.top_generalists.map((row) => ({
            name: row.name || 'unknown',
            score: row.avg_extraordinary_index || 0,
            domainsCovered: row.domains_covered || 0,
          }))
        : [],
    };
  }

  async function loadSnapshot() {
    try {
      const response = await fetch(snapshotUrl, { cache: 'no-store' });
      if (!response.ok) {
        return;
      }

      const payload = await response.json();
      window.__BENCHMARK_SNAPSHOT__ = normalizeSnapshot(payload);
      window.__BENCHMARK_SNAPSHOT_LOADED__ = true;
      window.dispatchEvent(new CustomEvent('benchmark-snapshot-ready'));
    } catch (error) {
      // Keep the embedded fallback when the site is opened directly from disk.
    }
  }

  loadSnapshot();
}());