# Baseline report — 2026-09-21

All **167 selected regression tests passed with zero skips**, including actual
Cyna tracing, the QA period-five coil checkpoint, and three prescribed-background
tokamak-like checkpoints. The new required-lane guards passed **23 tests**.
The finalized runner also passed a second 45-test production lane with exact
collection-count enforcement and a verified interactive allocation.

## Source and environment

- ESSOS baseline: `3c1dc4ffe0b255eb05389d91803db8591255026b`.
- pyna baseline: `4f032d7128cbc5edd62350b42c1015587c0d684c`.
- Both checkouts use `manifold-optimization`; baseline algorithms and physical
  test sources were unchanged during this run. Integration tooling and docs
  were being added; each run records its dirty state and configuration checksum.
- Isolated workspace environment: Python 3.12.13, JAX/jaxlib 0.6.2, NumPy 2.2.6,
  SciPy 1.15.3, pytest 8.4.2. The complete resolved CPU dependency lock is
  `benchmarks/manifold_optimization/requirements-cpu-py312.txt`.
- Native Cyna built from the pinned pyna source with GCC 13.2.1 and xmake 3.0.1
  (`3350cc5abe65608f6ec06f348aeded9ade1a1237`), using workspace-local tools.
- `pip check` passed after both editable installations.
- Perlmutter interactive allocation `58701910`, one exclusive GPU node; each
  numerical step requested 16 CPU cores and one GPU. The reference explicitly
  ran CPU JAX with float64 and CPU Cyna, with OpenMP/BLAS threads set to one.
- The allocation was released after validation.

## Executed inventory

| Lane | Passed | Skipped / failed | Elapsed seconds |
| --- | ---: | ---: | ---: |
| analytic | 118 | 0 / 0 | 508.7 |
| production | 45 | 0 / 0 | 57.8 |
| qa | 1 | 0 / 0 | 290.9 |
| tokamak | 3 | 0 / 0 | 105.9 |

Times sum per-selection subprocess wall time, including Python/pytest startup
and JAX compilation. Additional guard and strict-production checks overlapped
part of this run on the same allocation. These are regression execution times,
not controlled CPU/GPU performance measurements or optimizer benchmarks.

The QA test checks a converged primitive period-five orbit, Cyna/JAX agreement,
labelled manifold and wall-strike behavior, and refreshed physical validation
under its existing tolerances. The wall remains the original frozen regression
geometry. The tokamak-like tests retain a prescribed background field. Passing
either case does not establish a new divertor design or self-consistent equilibrium.

## Artifacts and reproducibility

Within the parent workspace, `run_artifacts/baseline-20260921/` contains:

- `cpu/`: all 167 tests, manifest, exact resolved configuration, execution audits,
  per-selection logs and JUnit XML.
- `strict-production/`: the final count/allocation enforcement run (45 tests),
  including compiler, source/tool/configuration checksums, and Slurm QoS.
- `audit-tests.log`: the 23 guard regressions.
- Native-tool fetch/build and package-install logs, including resolved setup failures.

The original full run established the observed module counts. The versioned
`baseline.json` now freezes those counts; every saved full-run audit was also
checked against the final count predicate. The exact original configuration was
saved after verifying its SHA-256 against the original manifest. The subsequent
production run exercised the count gate end to end.

Python syntax, workflow YAML, and Git whitespace checks passed. The new GitHub
Actions workflow is prepared locally; it has not been pushed or executed on GitHub.
Nothing has been committed or published as part of this baseline work.

## Plan progress and remaining G0 work

- Established the pinned source pair, isolated CPU environment, native build,
  selected fixture inventory, required-test audit, and existing physical checkpoints.
- Added the shared convention note in
  `pyna/docs/manifold_optimization_conventions.md`; it identifies the physical
  versus covariant field interfaces and current frozen-reference derivative semantics.
- R04 now has a local integration lane that fails on missing backends, skips,
  empty/deselected tests, and count mismatches. CI execution and repository merge
  requirements remain to be verified/configured separately.
- G0 remains open for numerical-error/observable tables and refinement studies,
  installed-wheel validation outside source roots, and the broader fixture inventory.
- R01–R03 and R05–R11 remain unresolved scientific/engineering work. The next
  bounded task is the finite-time versus long-time transform regression and
  reconciliation of the torus helper definition (NEXT-04), followed by moving-seed
  and snapshot-integrity diagnostics. No proposed benchmark advantage has been measured.

## Per-selection results

| Repository | Selection | Tests | Seconds |
| --- | --- | ---: | ---: |
| pyna | `tests/test_jax_poincare.py` | 6 | 6.4 |
| pyna | `tests/test_jax_periodic.py` | 8 | 24.7 |
| pyna | `tests/test_jax_manifold.py` | 7 | 6.5 |
| pyna | `tests/test_jax_strike.py` | 13 | 154.4 |
| pyna | `tests/test_torus_deformation.py` | 25 | 2.4 |
| pyna | `tests/test_manifold_correspondence.py` | 5 | 2.2 |
| pyna | `tests/test_manifold_strike_correspondence.py` | 7 | 2.1 |
| essos | `tests/test_periodic_xline_loss.py` | 6 | 24.7 |
| essos | `tests/test_return_map_surface_loss.py` | 5 | 7.5 |
| essos | `tests/test_manifold_branch_loss.py` | 5 | 19.2 |
| essos | `tests/test_manifold_optimization.py` | 8 | 23.6 |
| essos | `tests/test_manifold_strike_optimization.py` | 5 | 75.5 |
| essos | `tests/test_manifold_heat_optimization.py` | 9 | 117.5 |
| essos | `tests/test_manifold_leg_optimization.py` | 9 | 42.0 |
| pyna | `tests/test_jax_poincare_cyna.py` | 2 | 3.3 |
| pyna | `tests/test_jax_manifold_cyna.py` | 2 | 3.3 |
| pyna | `tests/test_manifold_correspondence_cyna.py` | 4 | 3.4 |
| pyna | `tests/test_manifold_refresh_cyna.py` | 4 | 2.4 |
| pyna | `tests/test_manifold_leg_clearance.py` | 12 | 3.1 |
| essos | `tests/test_manifold_adapter.py` | 9 | 11.6 |
| essos | `tests/test_manifold_validation.py` | 9 | 21.4 |
| essos | `tests/test_manifold_strike_validation.py` | 3 | 9.5 |
| essos | `tests/test_qa_manifold_coils.py::test_qa_period_five_xline_has_differentiable_multi_turn_strike` | 1 | 290.9 |
| essos | `tests/test_manifold_coils.py::test_coil_current_moves_a_hyperbolic_xline_and_its_manifold` | 1 | 51.7 |
| essos | `tests/test_manifold_coils.py::test_tokamak_pf_coil_candidate_passes_full_outer_validation` | 1 | 15.9 |
| essos | `tests/test_manifold_coils.py::test_tokamak_stable_leg_clearance_and_rejected_margin` | 1 | 38.3 |
