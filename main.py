#!/usr/bin/env python3
"""
ML Philosophy Benchmark: Main Entry Point

Run all benchmarks across all domains and generate reports.
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional


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


def generate_report(results_dir: str = "results"):
    """Generate comprehensive report."""
    from src.analysis.report_generator import ReportGenerator
    
    generator = ReportGenerator(results_dir=results_dir)
    report_path = Path(results_dir) / "REPORT.md"
    generator.save_report(str(report_path))
    print(f"Report generated: {report_path}")


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


def generate_release_snapshot(tag: str, results_dir: str = "results", snapshots_dir: str = "releases"):
    """Generate a versioned release snapshot after successful gate checks."""
    from src.core.release_snapshot import ReleaseSnapshotError, create_release_snapshot

    try:
        output_dir = create_release_snapshot(
            tag=tag,
            results_dir=results_dir,
            snapshots_root=snapshots_dir,
        )
    except ReleaseSnapshotError as exc:
        print("Release snapshot: FAILED")
        print(str(exc))
        raise

    print(f"Release snapshot: PASSED ({output_dir})")
    return output_dir


def run_publish_ready(
    publish_tag: str,
    results_dir: str = "results",
    snapshots_dir: str = "releases",
    n_train: int = 1000,
    n_test: int = 200,
    n_runs: int = 1,
    seed: int = 42,
    seed_list=None,
    skip_smoke: bool = False,
    prune_nightly_keep: Optional[int] = None,
    protect_tag_prefixes=None,
):
    """Run a one-command publish-ready pipeline and write summary artifacts."""
    from src.core.publish_ready import save_publish_ready_summary

    stages = []

    def _record_stage(name: str, fn, details: str = ""):
        start = time.time()
        fn()
        stages.append(
            {
                "name": name,
                "status": "PASS",
                "duration_seconds": round(time.time() - start, 3),
                "details": details,
            }
        )

    if not skip_smoke:
        _record_stage(
            "smoke_benchmark",
            lambda: run_all(
                n_runs=n_runs,
                seed=seed,
                seed_list=seed_list,
                n_train=n_train,
                n_test=n_test,
                output_dir=results_dir,
                smoke_test=True,
            ),
            details="Ran --all in smoke mode",
        )

    _record_stage(
        "report_generation",
        lambda: generate_report(results_dir=results_dir),
        details="Generated REPORT.md from latest artifacts",
    )
    _record_stage(
        "artifact_validation",
        lambda: validate_artifacts(results_dir=results_dir),
        details="Validated domain artifact contract",
    )
    _record_stage(
        "release_gate",
        lambda: run_release_gate_checks(results_dir=results_dir, require_report=True),
        details="Verified release-gate protocol/report checks",
    )

    snapshot_output = None

    def _snapshot_stage():
        nonlocal snapshot_output
        snapshot_output = generate_release_snapshot(
            tag=publish_tag,
            results_dir=results_dir,
            snapshots_dir=snapshots_dir,
        )

    _record_stage(
        "release_snapshot",
        _snapshot_stage,
        details="Generated versioned snapshot package",
    )

    if prune_nightly_keep is not None:
        from src.core.snapshot_retention import prune_snapshot_directories

        retention_start = time.time()
        retention_summary = prune_snapshot_directories(
            snapshots_root=snapshots_dir,
            keep_nightly=prune_nightly_keep,
            protect_prefixes=protect_tag_prefixes or ["v", "stable", "release"],
        )
        stages.append(
            {
                "name": "snapshot_retention",
                "status": "PASS",
                "duration_seconds": round(time.time() - retention_start, 3),
                "details": (
                    f"Kept nightly={prune_nightly_keep}, "
                    f"deleted={len(retention_summary.get('deleted', []))}, "
                    f"protected={len(retention_summary.get('protected', []))}"
                ),
            }
        )

    summary_output = save_publish_ready_summary(
        publish_tag=publish_tag,
        results_dir=results_dir,
        snapshot_dir=snapshot_output,
        stages=stages,
    )
    print(f"Publish-ready: PASSED ({summary_output})")


def restore_snapshot_from_releases(
    snapshot_tag: str,
    snapshots_dir: str = "releases",
    output_dir: str = "results",
):
    """Restore benchmark artifacts from a versioned snapshot."""
    from src.core.snapshot_restore import restore_snapshot
    
    print("=" * 80)
    print(f"RESTORING SNAPSHOT: {snapshot_tag}")
    print("=" * 80)
    
    result = restore_snapshot(
        snapshot_tag=snapshot_tag,
        snapshots_root=Path(snapshots_dir),
        output_dir=Path(output_dir),
    )
    
    if result["status"] == "FAIL":
        raise RuntimeError(result.get("error", "Unknown restoration error"))
    
    print(f"\n✓ Restored {len(result['domains_restored'])} domains from {snapshot_tag}")
    print(f"  Domains: {', '.join(result['domains_restored'])}")
    print(f"  Protocol version: {result['protocol_version']}")
    print(f"  Details: {result['details']}")
    
    if result.get("error"):
        print(f"\n⚠ Warnings: {result['error']}")


def list_snapshots(snapshots_dir: str = "releases"):
    """List available snapshots with summary metadata."""
    from src.core.snapshot_restore import list_available_snapshots

    rows = list_available_snapshots(Path(snapshots_dir))
    if not rows:
        print(f"No snapshots found in: {snapshots_dir}")
        return

    print(f"Found {len(rows)} snapshots in {snapshots_dir}:")
    for row in rows:
        print(
            f"- {row['tag']} | generated={row['generated_at_utc'] or 'N/A'} | "
            f"protocol={row['protocol_version'] or 'N/A'} | domains={row['domain_count']} | "
            f"report={'yes' if row['has_report'] else 'no'} | "
            f"restorable={'yes' if row['restorable'] else 'no'} | "
            f"valid={'yes' if row['valid'] else 'no'}"
        )


def show_snapshot_info(snapshot_tag: str, snapshots_dir: str = "releases"):
    """Show detailed metadata for a single snapshot tag."""
    from src.core.snapshot_restore import get_snapshot_info

    info = get_snapshot_info(snapshot_tag=snapshot_tag, snapshots_root=Path(snapshots_dir))
    print("=" * 80)
    print(f"SNAPSHOT INFO: {info['tag']}")
    print("=" * 80)
    print(f"Path: {info['path']}")
    print(f"Generated UTC: {info['generated_at_utc']}")
    print(f"Protocol: {info['protocol_version']}")
    print(f"Domains ({info['domain_count']}): {', '.join(info['domains'])}")
    print(f"Has REPORT.md: {'yes' if info['has_report'] else 'no'}")
    print(f"Restorable: {'yes' if info['restorable'] else 'no'}")


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
  python main.py --snapshot-tag v1.1      # Create versioned release snapshot
  python main.py --restore-snapshot v1.1  # Restore artifacts from snapshot
    python main.py --list-snapshots          # List available snapshots
    python main.py --snapshot-info v1.1      # Show snapshot metadata
  python main.py --publish-ready-tag v1.1 # One-command publish-ready pipeline
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all domains')
    parser.add_argument('--domain', '-d', type=str, 
                        choices=['a', 'b', 'c', 'd', 'e', 'ie', 'anomaly', 'rec', 'ts', 'tabular'],
                        help='Run specific domain')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--validate', action='store_true', help='Validate benchmark artifacts')
    parser.add_argument('--release-gate', action='store_true', help='Run full release-gate checks')
    parser.add_argument('--snapshot-tag', type=str, default=None,
                        help='Create versioned release snapshot (runs release-gate first)')
    parser.add_argument('--publish-ready-tag', type=str, default=None,
                        help='Run smoke+report+gates+snapshot and write publish-ready summary')
    parser.add_argument('--restore-snapshot', type=str, default=None,
                        help='Restore benchmark artifacts from a versioned snapshot (tag name)')
    parser.add_argument('--list-snapshots', action='store_true',
                        help='List available snapshots in --snapshots-dir')
    parser.add_argument('--snapshot-info', type=str, default=None,
                        help='Show detailed metadata for a snapshot tag')
    parser.add_argument('--snapshots-dir', type=str, default='releases',
                        help='Output directory root for --snapshot-tag')
    parser.add_argument('--skip-smoke', action='store_true',
                        help='Skip smoke run when using --publish-ready-tag')
    parser.add_argument('--prune-nightly-keep', type=int, default=None,
                        help='After publish-ready, keep only latest N nightly-* snapshots in --snapshots-dir')
    parser.add_argument('--protect-tag-prefixes', nargs='+', default=['v', 'stable', 'release'],
                        help='Protected tag prefixes for snapshot retention pruning')
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
    
    if args.list_snapshots:
        list_snapshots(snapshots_dir=args.snapshots_dir)
    elif args.snapshot_info:
        try:
            show_snapshot_info(snapshot_tag=args.snapshot_info, snapshots_dir=args.snapshots_dir)
        except Exception as exc:
            print(f"Snapshot info: FAILED ({exc})")
            sys.exit(1)
    elif args.restore_snapshot:
        try:
            restore_snapshot_from_releases(
                snapshot_tag=args.restore_snapshot,
                snapshots_dir=args.snapshots_dir,
                output_dir=args.output_dir,
            )
        except Exception as exc:
            print(f"Restore snapshot: FAILED ({exc})")
            sys.exit(1)
    elif args.report:
        generate_report(results_dir=args.output_dir)
    elif args.publish_ready_tag:
        try:
            run_publish_ready(
                publish_tag=args.publish_ready_tag,
                results_dir=args.output_dir,
                snapshots_dir=args.snapshots_dir,
                n_train=args.n_train,
                n_test=args.n_test,
                n_runs=args.n_runs,
                seed=args.seed,
                seed_list=args.seed_list,
                skip_smoke=args.skip_smoke,
                prune_nightly_keep=args.prune_nightly_keep,
                protect_tag_prefixes=args.protect_tag_prefixes,
            )
        except Exception as exc:
            print(f"Publish-ready: FAILED ({exc})")
            sys.exit(1)
    elif args.snapshot_tag:
        try:
            generate_release_snapshot(
                tag=args.snapshot_tag,
                results_dir=args.output_dir,
                snapshots_dir=args.snapshots_dir,
            )
        except Exception:
            sys.exit(1)
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
        generate_report(results_dir=args.output_dir)
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