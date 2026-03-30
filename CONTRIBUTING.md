# Contributing to 10ways2do

Thank you for contributing.

This project accepts improvements in:

- New approach implementations
- New domain implementations
- Evaluation and reporting improvements
- Documentation and test quality

## Ground rules

1. Keep runs reproducible.
2. Keep interfaces consistent.
3. Keep changes scoped.
4. Add or update tests.
5. Document assumptions and limitations.

## Development setup

```bash
python -m venv .venv
# activate env
pip install -r requirements.txt
```

## Before opening a PR

```bash
pytest tests -v
python main.py --all --smoke-test --n-runs 1 --seed 42
python main.py --validate
python main.py --release-gate
```

## Add a new approach (existing domain)

1. Create a new file in the target domain folder:
   - src/domain_x/approach_XX_name.py
2. Follow the same public interface used by existing approaches.
3. Register the approach in that domain's run_all.py.
4. Ensure output metrics match the domain schema.
5. Run tests and smoke benchmark.

## Add a new domain

1. Create a new folder under src/ with:
   - data_generator.py
   - run_all.py
   - approach_01_... through approach_10_...
2. Reuse core utilities in src/core:
   - benchmark_utils.py
   - benchmark_schema.py
3. Add domain routing in main.py.
4. Add tests under tests/.
5. Update README and docs.

## Coding expectations

- Use clear names and type hints where practical.
- Keep default configs modest and reproducible.
- Avoid hard-coded machine-specific paths.
- Fail explicitly with readable errors.

## Pull request checklist

- [ ] Code compiles/runs locally
- [ ] Tests pass
- [ ] Smoke benchmark passes
- [ ] Artifact validation passes
- [ ] Release gate passes
- [ ] Docs updated when behavior changes
- [ ] No unrelated formatting churn

## Review criteria

Maintainers review for:

- Correctness
- Reproducibility
- Benchmark fairness impact
- Long-term maintainability

## Questions

Open an issue with:

- What you are changing
- Why it is useful
- How you validated it
