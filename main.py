#!/usr/bin/env python3
"""
ML Philosophy Benchmark: Main Entry Point

Run all benchmarks across all domains and generate reports.
"""

import argparse
import sys
from pathlib import Path


def run_domain(domain: str, **kwargs):
    """Run benchmark for a specific domain."""
    common_kwargs = {
        'save_results': kwargs.get('save_results', True),
        'n_runs': kwargs.get('n_runs', 1),
        'seed': kwargs.get('seed', 42),
        'seed_list': kwargs.get('seed_list'),
        'smoke_test': kwargs.get('smoke_test', False),
    }

    if domain == 'a' or domain == 'ie':
        from src.domain_a_information_extraction.run_all import run_all_approaches
        domain_kwargs = {
            'n_train': kwargs.get('n_train', 1000),
            'n_test': kwargs.get('n_test', 200),
            'output_dir': kwargs.get('output_dir', 'results/domain_a'),
        }
        return run_all_approaches(**common_kwargs, **domain_kwargs)
    elif domain == 'b' or domain == 'anomaly':
        from src.domain_b_anomaly_detection.run_all import run_all_approaches
        domain_kwargs = {
            'n_train': kwargs.get('n_train', 1000),
            'n_test': kwargs.get('n_test', 200),
            'output_dir': kwargs.get('output_dir', 'results/domain_b'),
        }
        return run_all_approaches(**common_kwargs, **domain_kwargs)
    elif domain == 'c' or domain == 'rec':
        from src.domain_c_recommendation.run_all import run_all_approaches
        domain_kwargs = {
            'n_users': kwargs.get('n_train', 500),
            'n_items': kwargs.get('n_test', 200),
            'output_dir': kwargs.get('output_dir', 'results/domain_c'),
        }
        return run_all_approaches(**common_kwargs, **domain_kwargs)
    elif domain == 'd' or domain == 'ts':
        from src.domain_d_time_series.run_all import run_all_approaches
        domain_kwargs = {
            'n_samples': kwargs.get('n_train', 1000),
            'forecast_horizon': kwargs.get('n_test', 24),
            'output_dir': kwargs.get('output_dir', 'results/domain_d'),
        }
        return run_all_approaches(**common_kwargs, **domain_kwargs)
    elif domain == 'e' or domain == 'tabular':
        from src.domain_e_tabular_decisioning.run_all import run_all_approaches
        domain_kwargs = {
            'n_samples': kwargs.get('n_train', 6000),
            'n_features': kwargs.get('n_test', 24),
            'output_dir': kwargs.get('output_dir', 'results/domain_e'),
        }
        return run_all_approaches(**common_kwargs, **domain_kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def run_all(**kwargs):
    """Run all domains."""
    print("=" * 80)
    print("ML PHILOSOPHY BENCHMARK: Running All Domains")
    print("=" * 80)
    
    domains = ['a', 'b', 'c', 'd', 'e']
    results = {}
    base_output_dir = kwargs.get('output_dir', 'results')
    
    for domain in domains:
        print(f"\n\n{'#' * 80}")
        print(f"# DOMAIN {domain.upper()}")
        print(f"{'#' * 80}\n")
        
        try:
            domain_kwargs = dict(kwargs)
            domain_kwargs['output_dir'] = f"{base_output_dir}/domain_{domain}"
            results[domain] = run_domain(domain, **domain_kwargs)
        except Exception as e:
            print(f"Error running domain {domain}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def generate_report():
    """Generate comprehensive report."""
    from src.analysis.report_generator import ReportGenerator
    
    generator = ReportGenerator()
    generator.save_report("results/REPORT.md")
    print("Report generated: results/REPORT.md")


def validate_artifacts(results_dir: str = "results"):
    """Validate benchmark artifacts across all configured domains."""
    from src.core.artifact_validator import ArtifactValidationError, validate_results_tree

    try:
        validate_results_tree(results_dir=results_dir)
    except ArtifactValidationError as exc:
        print("Artifact validation: FAILED")
        print(str(exc))
        raise

    print("Artifact validation: PASSED")


def run_release_gate_checks(results_dir: str = "results", require_report: bool = True):
    """Run full release-gate checks for artifacts, manifests, and report sections."""
    from src.core.release_gate import ReleaseGateError, run_release_gate

    try:
        run_release_gate(results_dir=results_dir, require_report=require_report)
    except ReleaseGateError as exc:
        print("Release gate: FAILED")
        print(str(exc))
        raise

    print("Release gate: PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="ML Philosophy Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                    # Run all domains
  python main.py --domain a               # Run only Domain A (IE)
  python main.py --domain b --n-train 1000  # Run Domain B with custom size
  python main.py --report                 # Generate report from existing results
    python main.py --validate               # Validate artifact completeness/shape
    python main.py --release-gate           # Validate release readiness checks
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all domains')
    parser.add_argument('--domain', '-d', type=str, 
                        choices=['a', 'b', 'c', 'd', 'e', 'ie', 'anomaly', 'rec', 'ts', 'tabular'],
                        help='Run specific domain')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--validate', action='store_true', help='Validate benchmark artifacts')
    parser.add_argument('--release-gate', action='store_true', help='Run full release-gate checks')
    parser.add_argument('--no-report-check', action='store_true',
                        help='Skip REPORT.md checks when running --release-gate')
    parser.add_argument('--n-train', type=int, default=1000, help='Training set size')
    parser.add_argument('--n-test', type=int, default=200, help='Test set size')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--n-runs', type=int, default=1, help='Number of repeated runs with different seeds')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--seed-list', type=int, nargs='+', default=None,
                        help='Explicit seed list (overrides --n-runs and --seed)')
    parser.add_argument('--smoke-test', action='store_true', help='Run a lightweight subset of approaches for quick validation')
    
    args = parser.parse_args()
    
    if args.report:
        generate_report()
    elif args.release_gate:
        try:
            run_release_gate_checks(
                results_dir=args.output_dir,
                require_report=not args.no_report_check,
            )
        except Exception:
            sys.exit(1)
    elif args.validate:
        try:
            validate_artifacts(results_dir=args.output_dir)
        except Exception:
            sys.exit(1)
    elif args.all:
        run_all(
            n_runs=args.n_runs,
            seed=args.seed,
            seed_list=args.seed_list,
            n_train=args.n_train,
            n_test=args.n_test,
            output_dir=args.output_dir,
            smoke_test=args.smoke_test,
        )
        generate_report()
    elif args.domain:
        kwargs = {
            'n_train': args.n_train,
            'n_test': args.n_test,
            'output_dir': f"{args.output_dir}/domain_{args.domain[0]}",
            'n_runs': args.n_runs,
            'seed': args.seed,
            'seed_list': args.seed_list,
            'smoke_test': args.smoke_test,
        }
        run_domain(args.domain, **kwargs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()