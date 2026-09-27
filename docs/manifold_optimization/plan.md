# ESSOS–pyna Manifold Optimization: Implementation and Validation Plan

**Baseline date:** September 21, 2026  
**Status:** Proposed implementation plan; no implementation task is marked complete.  
**ESSOS baseline:** `hongkelu/ESSOS`, branch `manifold-optimization`, commit `3c1dc4f`.  
**pyna baseline:** `WenyinWei/pyna`, branch `manifold-optimization`, commit `4f032d7`.  
**Primary objective:** Turn the prototype into a reliable, reusable gradient-based framework for magnetic-field topology optimization, covering open field lines and closed flux surfaces, with vacuum demonstrations first and self-consistent finite-beta optimization later.

> **Working rule:** Every objective must identify the physical quantity being optimized; every gradient must identify exactly what varies and what is frozen; every accepted design must pass refreshed physical validation. Superiority is a hypothesis to test, not an assumed project outcome.

## Contents

1. [Mission, outcomes, and scope](#1-mission-outcomes-and-scope)
2. [Evidence baseline and priority issues](#2-evidence-baseline-and-priority-issues)
3. [Architecture and repository ownership](#3-architecture-and-repository-ownership)
4. [Mathematical and physical conventions](#4-mathematical-and-physical-conventions)
5. [Mathematical work packages](#5-mathematical-work-packages)
6. [Optimization and acceptance algorithm](#6-optimization-and-acceptance-algorithm)
7. [Wall geometry, clearance, and physical objectives](#7-wall-geometry-clearance-and-physical-objectives)
8. [Gated implementation milestones](#8-gated-implementation-milestones)
9. [Demonstration A: vacuum open-field-line control](#9-demonstration-a-vacuum-open-field-line-control)
10. [Demonstration B: vacuum closed-surface control](#10-demonstration-b-vacuum-closed-surface-control)
11. [Comparative experiments and claim criteria](#11-comparative-experiments-and-claim-criteria)
12. [Verification and numerical error budgets](#12-verification-and-numerical-error-budgets)
13. [Performance engineering](#13-performance-engineering)
14. [Continuous integration, packaging, and compatibility](#14-continuous-integration-packaging-and-compatibility)
15. [Reproducibility, artifacts, and execution safeguards](#15-reproducibility-artifacts-and-execution-safeguards)
16. [Finite-beta extension](#16-finite-beta-extension)
17. [Pull-request sequence and dependency map](#17-pull-request-sequence-and-dependency-map)
18. [Risk register and fallback decisions](#18-risk-register-and-fallback-decisions)
19. [First implementation backlog](#19-first-implementation-backlog)
20. [Release definition of done](#20-release-definition-of-done)
21. [Source register](#21-source-register)

---

## 1. Mission, outcomes, and scope

### 1.1 Scientific mission

Develop a framework that differentiates magnetic objects with respect to coil or field parameters and uses those sensitivities in constrained optimization. The objects of interest are:

- Periodic field lines and their stability properties.
- Stable and unstable manifolds of tracked hyperbolic periodic orbits.
- General labelled open-field-line bundles, including wall intersections and connection lengths.
- Closed invariant surfaces, their geometry, and their rotational transform.

The long-term scope includes tokamaks and stellarators. The first release must demonstrate a narrower, well-validated subset rather than claim universal topology control.

### 1.2 Engineering outcomes

The project should produce:

1. A clean ESSOS–pyna integration with stable ownership boundaries and no duplicated field-line algorithms.
2. Explicit numerical and physical validity contracts for every differentiable object.
3. A complete multi-variable optimization driver with correct acceptance, refresh, restart, and rollback behavior.
4. Two reproducible vacuum demonstrations with fair baselines, error budgets, and end-to-end performance measurements.
5. A field-provider interface that supports future equilibrium response without coupling topology algorithms to a particular equilibrium code.

### 1.3 First-release scope

The primary demonstrations will be two **coil-only vacuum stellarator cases**, distinguished by the object controlled:

- **Demonstration A:** open-field-line divertor-footprint and connection-length control with a protected core and engineering constraints.
- **Demonstration B:** controlled deformation of closed invariant surfaces and, where well-posed, their rotational-transform characteristics.

Retain a tokamak-like prescribed-current-background fixture as a portability and verification test. A prescribed plasma-current contribution must be labelled explicitly; it is not a coil-only vacuum tokamak equilibrium. Tokamak coverage remains part of the project and should not disappear simply because the first two flagship cases are stellarator vacuum cases.

### 1.4 Explicit non-goals for the first release

Do not make these prerequisites for the initial vacuum milestones:

- Self-consistent finite-beta equilibrium optimization.
- Transport-resolved divertor heat-flux prediction.
- Differentiability through orbit creation, island destruction, first-hit switching, or other topology changes.
- Guaranteed existence of an invariant torus for arbitrary requested geometry, flux label, and rotational transform.
- A global optimizer or proof of global optimality.
- A wholesale rewrite of ESSOS, pyna, their field models, or their production tracers.

Approximate surfaces, frozen-reference derivatives, local wall models, and prescribed power deposition may remain useful capabilities. They must be named and validated as such.

### 1.5 Project success

A successful project shows both that the software is reliable and that the proposed method adds measurable value. Those are separate conclusions.

A benchmark may show improved design capability without a speed advantage, or an implementation speed advantage without a new mathematical capability. Report that outcome precisely. A scientifically useful negative comparison is preferable to an unsupported superiority claim.

---

## 2. Evidence baseline and priority issues

### 2.1 Evidence status

This plan develops the preceding source review. Selected files at the two pinned commits were re-opened while preparing it. Repository test suites and benchmark runs were **not executed** during preparation. The task lists below describe work to perform, not completed work.

Source observations are linked through the source register. Mathematical counterexamples and proposed designs are identified separately. In particular, a convention-dependent formula must be checked against its intended definition before being described as an unconditional implementation bug.

The torus-deformation module refers to an internal theoretical document. Its full contents were not available for this plan. Review any authorized local copy during implementation; do not infer its full assumptions from the module header. [S01]

### 2.2 Preserve the existing strengths

The ESSOS adapter already describes a one-way dependency: ESSOS supplies active magnetic fields and design variables, while pyna owns maps and topology. Preserve that split. [S02]

The manifold correspondence layer already carries orbit, stability, branch-side, generation, seed, orientation, and phase information. Preserve these identities rather than replacing them with nearest-neighbor matching inside a gradient calculation. [S03]

The JAX branch is an optimization backend; the production tracing path is an independent acceptance/reference backend. Neither should be treated as infallible ground truth: each requires convergence tests, and the production field representation introduces its own approximation. [S04]

### 2.3 Priority issue ledger

| ID | Priority | Observation or concern | Required disposition |
|---|---|---|---|
| R01 | P0: scientific correctness | `iota_variation_pf()` is documented as a transform variation but uses a finite-turn harmonic averaging factor. [S01] | Resolve its precise quantity and assumptions; add analytic counterexamples; correct or rename before physical use. |
| R02 | P0: gradient semantics | Manifold eigendirections and seed spacing are frozen in the differentiable branch implementation. [S04] | Expose frozen-reference mode; implement and validate moving-geometry sensitivities. |
| R03 | P0: optimizer correctness | The existing backtracking helper contracts until topology validation accepts; it does not independently enforce objective or merit decrease. [S05] | Add a full optimization acceptance layer without misrepresenting the helper's current purpose. |
| R04 | P0: reproducibility | The inspected ESSOS workflow does not explicitly install pyna; integration tests include optional-import skips. [S06] [S07] | Add a required pinned-pair job that fails when required backends or tests are absent. |
| R05 | P1: state integrity | Frozen branch dataclasses contain writable arrays and mutable metadata. [S03] | Implement owned, non-aliased snapshots and tested serialization/rollback. |
| R06 | P1: conditioning | Implicit periodic roots and wall events require more than finite output or a small residual. [S08] [S09] | Gate derivatives using scaled conditioning, convergence, and topology validity. |
| R07 | P1: closed-surface formulation | Endpoint-to-target-surface residuals do not themselves solve an invariant torus. [S02] | Add a well-posed torus predictor/corrector and an explicit surface label/gauge. |
| R08 | P1: architectural coupling | The continuation state is tied to a manifold target even when return-map stages are used. [S10] | Permit independent torus-only, manifold-only, and general open-line problems. |
| R09 | P1: geometry semantics | Section-based wall clearance and angle-fraction terminal exclusions need explicit physical interpretation. [S11] | Distinguish 3-D clearance, sampled validation, local surrogates, and physical target exemptions. |
| R10 | P2: benchmark strength | The QA fixture uses a frozen regression wall and a deliberately controlled integration setup. [S07] | Preserve it as a regression; build independent, multi-variable design experiments. |
| R11 | P2: packaging | ESSOS explicitly lists its top-level package in build configuration. [S12] | If new subpackages are introduced, test package discovery and installed-wheel contents. |

**Priority meaning:** P0 issues block affected scientific claims. P1 issues block a reliable integrated release. P2 work strengthens usability, scaling, and evidence. Not every concern requires a new abstraction or a new module.

---

## 3. Architecture and repository ownership

### 3.1 Ownership contract

| Responsibility | Owner | Boundary |
|---|---|---|
| Coil curves, currents, field evaluation, design-variable transforms | ESSOS | pyna must not reimplement ESSOS coil models. |
| Coil constraints, objective composition, scaling, optimization orchestration | ESSOS | Generic topology kernels remain in pyna. |
| Field-line maps, periodic orbits, manifold parameterizations, invariant-circle solvers | pyna | Consume a field callable/provider; do not import ESSOS. |
| Wall events, topology identity, continuation reports, production tracing | pyna | Expose physical results and diagnostics through documented interfaces. |
| Independent candidate validation | pyna algorithms, orchestrated by ESSOS | Separate validity from optimization acceptance. |
| Cross-repository fixtures and benchmark drivers | ESSOS integration layer | Keep reusable analytic topology tests in pyna. |
| Equilibrium state and plasma-response derivative | Future field/equilibrium adapter | Do not put equilibrium-solver internals into topology kernels. |

### 3.2 Three objects that must remain distinct

**Physical problem definition:** immutable targets, engineering limits, surface labels, field model, parameter scaling, and objective definitions.

**Tracking state:** orbit identities, branch signs, reference orientations, launch labels, first-hit identities, invariant-circle phase choices, and permitted continuation windows.

**Evaluation state:** a particular candidate's field, corrected roots, trajectories, residuals, derivative operators, numerical diagnostics, validation reports, and timing data.

A topology refresh may update tracking state. It must not silently move the physical target or redefine success.

### 3.3 Proposed interfaces, not existing APIs

Use the following concepts to guide incremental refactoring. Names are provisional and should be adapted to the existing public APIs rather than introduced mechanically.

| Concept | Required contents |
|---|---|
| `FieldProvider` | Cartesian field evaluation, parameter schema, differentiability capabilities, frame/units, content/version identity. |
| `DesignProblem` | Immutable targets, objective components, constraints, parameter bounds/scales, requested objects. |
| `TopologySnapshot` | Typed tracked objects, discrete identities, continuous reference data, schema version, validation provenance. |
| `EvaluationRequest` | Requested outputs and derivative mode; no unused expensive computations. |
| `CandidateEvaluation` | Shared numerical results, JVP/VJP operators, diagnostics, cache identity, timing record. |
| `ValidityReport` | Typed status and reason codes, tolerances, residuals, conditioning, topology checks. |
| `AcceptedState` | Parameters, snapshot, optimizer history, objective version, numerical configuration, reproducibility manifest. |

JVP means Jacobian-vector product; VJP means vector-Jacobian product. Dense Jacobians may be offered for small problems but must not be required by the core interface.

### 3.4 Dependency rules

The permitted dependency direction is:

```text
ESSOS design variables and fields
          |
          v
ESSOS adapters and objective composition
          |
          v
pyna field-line / topology / event primitives
          |
          +--> differentiable local evaluation
          +--> production tracing and validation
```

Do not introduce a third general-purpose framework unless a concrete duplication problem survives the small-interface refactor. Preserve current import paths through compatibility wrappers while new contracts stabilize.

### 3.5 Snapshot integrity and caching

Accepted snapshots must own their data and must not alias trial arrays. Read-only NumPy arrays reduce accidental mutation, but the stronger requirement is observable snapshot integrity across all public operations.

- Copy incoming mutable data; freeze nested metadata or return defensive copies.
- Separate static labels from differentiable numerical leaves.
- Include schema and numerical-configuration versions in serialized checkpoints.
- Key candidate caches by field/design identity, topology snapshot, wall/target versions, and numerical settings.
- Never key a scientific cache only by Python object identity or an uncontrolled mutable dictionary.
- Test rejected candidates, failed callbacks, restart, and serialization round trips for state contamination.

---

## 4. Mathematical and physical conventions

### 4.1 Publish a single conventions document

Before changing derivatives, record conventions shared by both repositories:

| Quantity | Required convention |
|---|---|
| Cartesian position | Metres; explicit right-handed frame. |
| Cylindrical coordinates | `(R, phi, Z)` with angles in radians; explicit handedness. |
| Field components | Physical orthonormal components versus covariant/contravariant components must be named distinctly. |
| Design parameters | Physical values and normalized optimizer values, with an invertible mapping. |
| Poincaré map | Starting section, signed toroidal span, field-period count, map power. |
| Rotational transform | Long-time poloidal turns per toroidal turn; specify angular lift and surface label. |
| Fourier modes | State the sign convention and whether toroidal angle is physical or field-period normalized. |
| Branch identity | Stable/unstable, side, orientation, orbit phase, generation, seed label. |
| Wall distance | Sign, units, dimensionality, admissible region, and nonsmooth locations. |
| Connection length | Physical arclength, launch convention, terminal event, and maximum trace domain. |

A symbol such as `Bphi` must not ambiguously denote a physical cylindrical component, a covariant component, and a coordinate velocity.

### 4.2 Fixed-toroidal-angle tracing

For physical cylindrical components and a valid toroidal-angle parameterization, use

$$
\frac{dR}{d\phi}=\frac{R B_R}{B_\phi},
\qquad
\frac{dZ}{d\phi}=\frac{R B_Z}{B_\phi}.
$$

These equations follow from the cylindrical coordinate velocity along a field line. Their domain requires a usable, nonvanishing toroidal component along the requested trajectory. Specify whether `Bphi` thresholds are absolute, relative to field strength, or both.

Do not silently clip the denominator or continue a trace through a coordinate singularity. An arclength-parameterized fallback would be a separately tested capability, not an implicit change to the current numerical problem.

### 4.3 Map span and rotational transform

For a section map covering toroidal angle $\Delta\phi$, an invariant-circle parameterization obeys

$$
P_c(K(\theta))=K(\theta+\omega).
$$

If $\theta$ is a $2\pi$-periodic poloidal angle and $\omega$ is its continuously lifted advance in radians, then

$$
\iota=\frac{\omega}{\Delta\phi}.
$$

For one field period, $\Delta\phi=2\pi/N_{\mathrm{fp}}$. Do not lose the field-period factor or infer the winding from a shift reduced modulo $2\pi$ without tracking its lift.

### 4.4 Flux preservation is coordinate dependent

For a divergence-free field mapped between transverse sections, the conserved measure is magnetic flux. On a constant-$\phi$ section with physical cylindrical field components, it is proportional to $B_\phi\,dR\,dZ$.

Accordingly, a useful numerical check is

$$
B_\phi(P(x),\phi_1)\det DP(x)
\approx B_\phi(x,\phi_0),
$$

with consistent orientation and equivalent periodic sections where appropriate. Do not require $\det DP=1$ everywhere in raw $(R,Z)$ coordinates. At an exactly periodic point on equivalent sections, the density cancels and the familiar unit-determinant check is recovered under these assumptions.

Include field interpolation error and divergence error when interpreting this diagnostic.

### 4.5 Surface labels are part of the problem

Distinguish fixed toroidal flux, fixed rotational transform, fixed enclosed volume, and continuation from a reference surface. They define different derivatives.

The identity

$$
\delta r=-\frac{\delta\iota}{\iota'}
$$

is appropriate only under a matching convention, such as compensating a transform change by moving along a sheared family while preserving a selected transform value. It is not a universal physical radial displacement at fixed flux. Zero or small shear requires separate treatment.

---

## 5. Mathematical work packages

### 5.1 MATH-01: audit rotational-transform and torus-deformation formulas

**Owner:** pyna.  
**Blocks:** physical transform objectives and Demonstration B.

The existing non-resonant module contains both spectral deformation helpers and the disputed transform helper. Use its documented Fourier convention when constructing tests. [S01]

**Required work**

- [ ] Derive the intended transform quantity and the held-fixed surface label from the governing field-line equation.
- [ ] Audit nonzero harmonics, the zero mode, conjugate-pair normalization, angular units, and field-period factors.
- [ ] Separate finite-turn angular increments from asymptotic rotational transform.
- [ ] Audit the assumptions behind first-order mean radial displacement and any second-order expression before using them in an objective.
- [ ] Correct the API or rename the finite-time quantity; provide a migration note for changed semantics.
- [ ] Mark unsupported resonant/shearless cases explicitly rather than hiding them in a regularized result.

**Analytic regression: finite-time versus long-time quantity**

Consider the scalar flow, with $\iota_0>0$ and $|\epsilon|<1$,

$$
\frac{d\theta}{d\phi}=\iota_0(1+\epsilon\cos\theta).
$$

One poloidal turn takes

$$
T_\phi=\int_0^{2\pi}\frac{d\theta}{\iota_0(1+\epsilon\cos\theta)}
=\frac{2\pi}{\iota_0\sqrt{1-\epsilon^2}},
$$

so

$$
\iota_{\mathrm{rot}}=\iota_0\sqrt{1-\epsilon^2}
=\iota_0-\tfrac12\iota_0\epsilon^2+O(\epsilon^4).
$$

The linear variation is zero. The finite-one-turn harmonic average underlying the review's interpretation instead gives

$$
\delta\iota_{\mathrm{finite}}=
\frac{\epsilon\sin(2\pi\iota_0)}{2\pi}
$$

for a cosine launched at zero phase. This derivation distinguishes two quantities; it is not by itself a proof that every physical assumption of the module maps to this scalar flow. Add that mapping explicitly before changing the implementation.

**Exit gate:** analytic tests establish what each exported quantity means; no physical objective uses an unresolved definition.

### 5.2 MATH-02: periodic-orbit sensitivity and conditioning

**Owner:** pyna.  
**Blocks:** trustworthy manifold and wall-hit derivatives.

For the tracked root

$$
F(x_*,c)=P_c^k(x_*)-x_*=0,
$$

the local implicit derivative satisfies

$$
(D_xP_c^k-I)\frac{dx_*}{dc}=-D_cP_c^k.
$$

The current implicit-root approach is a foundation to retain. [S08]

**Required work**

- [ ] Return primal root residual, scaled linear-system conditioning, root identity, and derivative validity separately.
- [ ] Use linear solves, not explicit matrix inversion, in implementations.
- [ ] Account for both direct parameter dependence and the moving root when differentiating the monodromy.
- [ ] Test primitive period versus map power; detect convergence to a different orbit or phase.
- [ ] Support forward and backward tracing conventions consistently, including numerical non-exactness of inverse maps.
- [ ] Propagate failure without fabricating a zero gradient or treating NaNs as ordinary objective values.

When $A=D_xP_c^k-I$ is poorly conditioned, a small residual $r$ can imply a much larger state error, approximately $A^{-1}r$. Acceptance tolerances must account for this amplification.

**Exit gate:** analytic and finite-difference checks agree away from singular regimes; near-singular or unconverged cases return explicit invalid/ill-conditioned states.

### 5.3 MATH-03: distinguish derivative levels for manifolds

**Owner:** pyna, exposed through ESSOS.

Adopt explicit derivative modes:

| Mode | What changes continuously | Appropriate description |
|---|---|---|
| `frozen_reference` | Anchor and traced maps; selected seed geometry remains fixed | Derivative of the frozen-reference surrogate. |
| `moving_linear_seed` | Anchor, simple eigendirection, and any continuously defined seed coordinates | Derivative of a moving linearized local-manifold approximation. |
| `moving_parameterized_manifold` | A corrected higher-order local manifold and its propagated images | Derivative of the converged discretized manifold representation. |

Discrete identities remain fixed in all three modes within a local solve. None differentiates through branch discovery or an identity switch.

For a linear seed

$$
u_0(c)=x_*(c)+\epsilon(c)v(c),
$$

its derivative is

$$
\frac{du_0}{dc}=\frac{dx_*}{dc}
+\epsilon\frac{dv}{dc}
+v\frac{d\epsilon}{dc}.
$$

Whether $\epsilon$ varies is a problem-definition decision. A fixed physical seed distance and a continuously defined fundamental-segment coordinate are different conventions.

For a simple, normalized real eigenpair $Mv=\lambda v$, one local sensitivity construction is

$$
\begin{bmatrix}M-\lambda I&-v\\v^\mathsf{T}&0\end{bmatrix}
\begin{bmatrix}\delta v\\\delta\lambda\end{bmatrix}
=
\begin{bmatrix}-(\delta M)v\\0\end{bmatrix}.
$$

Use orientation continuation to fix the sign, and diagnose eigenvalue separation and solve conditioning. Here $\delta M$ includes the motion of the periodic point.

**Required tests**

- [ ] A manufactured hyperbolic map with a rotating eigendirection.
- [ ] Moving anchors and changing eigenvalues, independently and together.
- [ ] Both stable and unstable branches, both sides, and negative-eigenvalue parity.
- [ ] Weak hyperbolicity and eigenpair near-degeneracy with explicit rejection.
- [ ] Seed-size convergence; quantify local linearization error after propagation.
- [ ] Derivative comparisons with independently reconstructed, orientation-aligned geometry.

**Important limit:** differentiating the eigendirection does not make a finite-offset straight seed segment an exact nonlinear invariant manifold. Compare local invariance residuals and seed-size refinement. Introduce a higher-order parameterization only if this error limits the scientific result.

For the return map $Q_c=P_c^k$ that fixes the selected periodic anchor, a possible higher-order local equation is $Q_c(W(s))=W(\lambda s)$ with $W(0)=x_*$ and a normalization for $W'(0)$. Here $\lambda$ is the eigenvalue of the same return map; use its backward counterpart consistently for the selected stable construction. Domain restrictions must keep both arguments inside the parameterized neighborhood.

**Exit gate:** the selected mode passes its own derivative tests and its physical representation error is reported separately.

### 5.4 MATH-04: moving wall events

**Owner:** pyna.  
**Blocks:** physical wall-hit optimization.

Let a trajectory $X(\phi,c)$ intersect a wall level set $h(X,c)=0$ at $\phi=\tau(c)$. Differentiating the event equation gives

$$
\frac{d\tau}{dc}
=-\frac{h_x X_c+h_c}{h_x X_\phi},
\qquad
\frac{dX_{\mathrm{hit}}}{dc}=X_c+X_\phi\frac{d\tau}{dc}.
$$

The derivative is local to a transverse event with stable identity. The existing implicit wall-event machinery should be retained and strengthened through this contract. [S09]

**Required work**

- [ ] Test analytic plane, curved-wall, and moving-launch cases.
- [ ] Distinguish frozen local-plane models from a wall's actual geometric representation.
- [ ] Track unwrapped phase and verify the same global first hit after candidate refresh.
- [ ] Diagnose grazing using a physically interpretable, scaled transversality metric; raw level-set scaling must not change validity arbitrarily.
- [ ] Test competing intersections, surface seams, and targets crossed after multiple turns.
- [ ] Treat absent hits and trace limits as explicit statuses, not arbitrary huge connection lengths with invented derivatives.

A tangent plane can reproduce a smooth wall's normal at a reference hit, but finite candidate motion, patch changes, and wall curvature still require validation against the actual wall. Do not interpret local-plane convergence as global event validation.

**Exit gate:** same-discretization and refreshed-wall derivative checks pass away from switching/grazing, and adversarial event cases reject correctly.

### 5.5 MATH-05: invariant-circle predictor, corrector, and sensitivity

**Owner:** pyna solver; ESSOS target integration.  
**Blocks:** Demonstration B.

Use the section invariance equation

$$
\mathcal R(K,\omega,c)(\theta)=P_c(K(\theta))-K(\theta+\omega)=0.
$$

**Formulation tasks**

- [ ] Choose the Fourier/collocation representation, map span, and winding lift.
- [ ] Remove phase freedom with an explicit gauge, such as an alignment condition against a fixed reference parameterization.
- [ ] Choose one physical surface-label convention and determine which scalar quantities are unknowns or constraints.
- [ ] Derive a dimensionally consistent, well-posed bordered system; do not append flux and transform constraints without counting unknowns and solvability conditions.
- [ ] Recognize that an exact torus may not exist for every requested label; report unavailable labels rather than forcing a nominally exact solve.
- [ ] Validate injectivity, regularity, nesting, and dense off-grid invariance, not only collocation residuals.

**Predictor tasks**

Use the non-resonant deformation theory as a predictor or preconditioner. Include the shear term when the desired quantity is the invariant-torus conjugacy. [S01]

- [ ] Derive the zero-mode treatment and nonzero-mode denominators under the published convention.
- [ ] Diagnose small divisors and the validity of the perturbative approximation.
- [ ] For a converged baseline and smooth non-resonant perturbations, test reduction of the leading residual from $O(\epsilon)$ to $O(\epsilon^2)$ over a resolved amplitude range.
- [ ] Separate perturbation-truncation error from Fourier, quadrature, and tracing errors.

**Corrector and derivative tasks**

Denote the complete gauged system by $G(z,c)=0$, where $z$ includes the torus coefficients and any scalar unknowns. Then

$$
G_z\,\delta z=-G_c\,\delta c.
$$

Provide JVP and VJP operations using the corresponding linear or transpose solve.

If the solver instead returns an approximate surface by minimizing a nonzero residual, differentiate the **stationarity equations of that optimization problem**, including its constraints. Do not silently differentiate $\mathcal R=0$ when it is not satisfied. A Gauss–Newton sensitivity that drops residual-Hessian terms is an approximation and must be labelled and tested.

**Exit gate:** well-posedness, predictor order, corrected residual convergence, and implicit sensitivities are demonstrated on manufactured examples and one actual coil field.

### 5.6 MATH-06: general open-line bundles

**Owner:** pyna, integrated by ESSOS.

Open field lines are broader than stable/unstable manifolds. Define a launch-bundle interface supporting fixed physical launches, flux-labelled launches, and manifold-generated launches without requiring an artificial hyperbolic orbit in every problem.

- [ ] State whether the launch surface and quadrature weights move with the field.
- [ ] Differentiate moving launches where the physical definition requires them.
- [ ] Keep topology labels and launch quadrature independent from plotting resampling.
- [ ] Verify connection-length and wall-hit derivatives on an arbitrary labelled bundle.

**Exit gate:** at least one non-manifold open-line test works through the same objective/validation interfaces.


## 6. Optimization and acceptance algorithm

### 6.1 Start with a constrained, scaled local optimizer

Represent normalized design variables by $q$, with physical parameters $c=c_{\mathrm{ref}}+Dq$ or another explicitly documented transform. Define an objective and constraints such as

$$
\min_q J(q)=\tfrac12\|r(q)\|^2+J_{\mathrm{reg}}(q),
\qquad a(q)=0,\quad g(q)\leq0,\quad q_{\min}\leq q\leq q_{\max}.
$$

Residuals must be dimensionless or have explicit weights with documented units. Preserve engineering limits as limits: a smaller weighted objective must not conceal a violated hard clearance constraint.

Begin with one well-tested trust-region or constrained local-solver path that fits the residual/JVP/VJP interfaces. Avoid supporting many optimizers before one complete driver is validated. Benchmark optimizer changes separately from derivative changes.

### 6.2 Separate three acceptance questions

A candidate must pass:

1. **Numerical validity:** converged roots, finite trajectories, acceptable conditioning, valid events, and sufficient discretization accuracy.
2. **Topology and physical feasibility:** permitted identities, branch orientation, wall first hits, protected regions, and engineering constraints.
3. **Optimization acceptance:** sufficient improvement according to the declared merit, filter, or trust-region rule.

The topology-only helper remains useful inside this process. It must not be renamed or advertised as a complete optimization acceptance algorithm without adding the missing criterion. [S05]

For the first driver, require a numerically valid starting design satisfying the declared hard engineering constraints. If a starting design is infeasible, use a separately specified feasibility-restoration phase with explicit admissible relaxations and stopping criteria. Such intermediate states are not validated feasible designs. Do not silently relax a safety/geometry limit or confuse objective targets with hard constraints to make progress possible.

### 6.3 Two-level trial evaluation

At accepted parameters $q_k$, retain an accepted tracking snapshot $T_k$ and a refreshed physical evaluation. Build a local model from the declared derivative mode.

For each proposed trial:

```text
1. Build q_trial from the accepted state and the current step/trust radius.
2. Apply cheap parameter-bound and geometry screens.
3. Evaluate the differentiable local model using the accepted tracking identities.
4. Reject invalid local roots/events or clearly unacceptable local merit.
5. Reconstruct the candidate's tracked physical objects without mutating T_k.
6. Run production topology, event, clearance, and engineering validation.
7. Evaluate the refreshed candidate objective under the same physical definition.
8. Compare refreshed merit improvement with the local model prediction.
9. Accept atomically, or retain the accepted state and adjust proposal controls.
10. Record the trial, including all rejection reasons and costs.
```

Cheap screening can avoid production work for obviously poor candidates. In the first reliable implementation, every **accepted** candidate receives production validation. Any later relaxation of refresh frequency requires its own ablation and final full validation.

### 6.4 Trust-region agreement and objective consistency

For a merit function $\Phi$ and local model $m_k$, one useful agreement ratio is

$$
\rho_k=
\frac{\Phi(q_k,T_k)-\Phi(q_{\mathrm{trial}},T_{\mathrm{trial}})}
{m_k(0)-m_k(p_k)}.
$$

Evaluate numerator terms using the same physical targets, sampling measure, objective scaling, and comparable validation accuracy. Do not compare a refreshed candidate against an old value defined on a different target or quadrature.

If the predicted decrease is nonpositive or below numerical resolution, do not divide by it and accept an arbitrary ratio. Shrink, refine, or terminate with an explicit reason. Keep merit weights fixed during an individual acceptance decision.

A topology refresh can update representations while preserving the underlying physical functional. When it changes the actual functional—such as adding a target, changing penalty weights, or changing which launches are counted—start a new optimization stage and reset or deliberately transport optimizer history.

### 6.5 Rejection and state semantics

A rejected candidate must not alter accepted parameters, accepted topology, or quasi-Newton/momentum information associated with a successful step. It may legitimately update trial-control metadata such as the trust radius, rejection counter, and diagnostic log.

Acceptance must atomically commit:

- Parameters and field identity.
- Refreshed topology and event identities.
- Objective and numerical-configuration versions.
- Valid optimizer history for the accepted step.
- A checkpoint and its validation provenance.

Exceptions caused by programming, configuration, or backend failures should remain distinguishable from ordinary physical rejection. Avoid an all-purpose exception handler that reports every crash as an infeasible coil design.

### 6.6 Topology-regime changes

The default optimizer stays within a tracked smooth regime. A branch switch, lost torus, or changed first hit is not automatically another differentiable trial.

When progress stalls at a regime boundary:

- Diagnose the limiting validity condition.
- Reduce the step or change the continuation stage where scientifically justified.
- Optionally perform an explicit outer rediscovery, create a new snapshot, and restart the local model.
- Preserve the history as a regime transition; do not splice gradients across it as though the problem had remained smooth.

### 6.7 Stopping conditions

Report separate reasons for successful convergence, feasible stagnation, infeasible stagnation, topology boundary, insufficient numerical accuracy, iteration/evaluation budget, and backend failure.

Do not declare convergence from a small surrogate gradient alone. Require a refreshed feasible state and a stationarity/progress criterion appropriate to the constrained problem and its numerical noise floor.

---

## 7. Wall geometry, clearance, and physical objectives

### 7.1 Wall representations

Use one authoritative physical wall definition and document how each backend approximates it. Local differentiable surfaces, signed-distance approximations, and triangulated production walls must be related by a measured geometric error.

For a non-axisymmetric wall, the closest point constrained to the same toroidal section is not generally the closest point in three dimensions. Since the section is a subset of the wall,

$$
d_{\mathrm{3D}}(x,\mathcal W)\leq d_{\mathrm{section}}(x,\mathcal W\cap\{\phi=\phi_x\}).
$$

Therefore a large section distance does not prove a corresponding 3-D clearance.

**Required tasks**

- [ ] Name every clearance metric by dimension, sign convention, and approximation.
- [ ] Add a 3-D closest-point or signed-distance validation path for engineering-clearance claims.
- [ ] Handle nonunique closest points, wall corners, and patch boundaries explicitly.
- [ ] Test production-wall and local-model convergence independently.
- [ ] Preserve an analytic wall fixture for exact-event and geometry regression tests.

### 7.2 Whole-leg versus sampled clearance

A trajectory checked only at discrete samples has a **sampled minimum clearance**, not automatically a certified continuous minimum.

For distance to a fixed closed set, the distance function is 1-Lipschitz. If a resolved curve segment has a conservative arclength bound $\ell$ between its endpoints, then

$$
d(x(s),\mathcal W)\geq\min(d_i,d_{i+1})-\frac{\ell}{2}
$$

provides one simple conservative lower bound using the nearer endpoint. It is only a certificate when the segment-length bound and trajectory/geometry errors are themselves controlled. Endpoint chord length is not an upper bound on curve arclength.

Implement adaptive sampling first, with convergence evidence. Add certified bounds only when their assumptions can be supplied and verified. Subtract known trajectory and wall-approximation uncertainties from any claimed lower bound.

### 7.3 Target approach exemptions

A trajectory intended to strike a target cannot maintain positive wall distance at its endpoint. Define the permitted target approach geometrically: for example, an authorized wall patch and a specified physical final arclength region.

The existing angle-fraction exclusion is a numerical convention, not a universal physical length. [S11]

Do not allow an exemption to hide a prior collision with another patch or an excursion into a forbidden region. Validate the entire first-hit ordering and distinguish target hardware from unrelated wall structures.

### 7.4 Candidate objective families

| Objective | Required definition | Independent validation |
|---|---|---|
| Footprint position | Labelled wall coordinates or a target-patch distance with a stated metric | Actual 3-D first hits, not only local planes. |
| Footprint shape/spread | A curve or distribution metric with fixed sampling measure | Held-out launches and denser wall sampling. |
| Connection length | Arclength from a declared launch to a declared event | Independently integrated length with converged events. |
| Protected-core condition | Explicit surface, orbit, or region constraints | Dense field-line/surface checks at refined settings. |
| Coil regularization | Length, curvature, spacing, current limits, and optional manufacturability terms | Independent geometry checks and physical units. |
| Rotational transform | Corrected-surface label, winding lift, and transform definition | Long-trace or independent surface-based estimate. |

Hard first-hit switches, nearest-patch choices, and exact maxima may be nonsmooth. Smooth surrogates can be used inside optimization, but report hard-metric validation and smoothing bias separately.

### 7.5 Heat deposition is optional and explicitly model dependent

Do not equate a geometric footprint map with a transport prediction. A prescribed-power model must state its launch measure, bundle weights, total power, deposition kernel, wall-area measure, incidence factors, and unresolved loss channels.

Power conservation should be tested under the chosen model:

$$
\sum_j P_j + P_{\mathrm{unresolved}} = P_{\mathrm{launched}}
$$

within stated numerical error. Do not artificially lower peak deposition by silently dropping missing trajectories or changing launch weights during optimization.

Heat deposition remains an optional follow-on objective. It does not gate the first open-line geometry result.

---

## 8. Gated implementation milestones

### Milestone 0 — Reproduce and freeze the baseline

**Owners:** both repositories; ESSOS coordinates integration.  
**Dependencies:** none.  
**Gate:** G0.

- [ ] **BASE-01:** Resolve the abbreviated commits to full hashes and record clean/dirty working-tree status without modifying the baseline.
- [ ] **BASE-02:** Create a reproducible CPU environment; record Python, numerical-library, compiler, backend, and precision settings.
- [ ] **BASE-03:** Inventory relevant tests and examples in both repositories. Record executed, skipped, failed, and unavailable cases separately.
- [ ] **BASE-04:** Reproduce the existing analytic, QA, and tokamak-like checkpoints where dependencies are available.
- [ ] **BASE-05:** Capture baseline objective values, trace differences, test outputs, and timing breakdowns without interpreting a regression fixture as a benchmark result.
- [ ] **BASE-06:** Publish the conventions document, compatibility manifest, and issue dispositions from Section 2.

**Deliverables:** baseline report, exact environment/commit manifest, fixture inventory, initial numerical-error measurements.  
**G0 exit:** the reproducible baseline and any missing prerequisites are explicit; required integration tests cannot disappear through silent skips.

### Milestone 1 — Harden interfaces and accepted-state integrity

**Owners:** pyna for topology contracts; ESSOS for state orchestration.  
**Dependencies:** G0.  
**Gate:** G1.

- [ ] **ARCH-01:** Separate physical targets, tracking state, and candidate evaluation.
- [ ] **ARCH-02:** Add versioned snapshot serialization, owned arrays, and mutation/aliasing tests.
- [ ] **ARCH-03:** Allow torus-only and general open-line problems without an artificial manifold target.
- [ ] **ARCH-04:** Define field capabilities, units, parameter scales, and numerical validity statuses.
- [ ] **ARCH-05:** Preserve compatibility wrappers and ordinary ESSOS use without required pyna imports.
- [ ] **ARCH-06:** Add a minimal shared-candidate evaluation context; defer elaborate cache optimization.

**Deliverables:** documented contracts, migration notes, compatibility tests, snapshot tests.  
**G1 exit:** independent problem types construct cleanly; rejected and serialized states retain their meaning and contents.

### Milestone 2 — Establish mathematical and derivative correctness

**Owners:** pyna, with ESSOS field-parameter integration tests.  
**Dependencies:** G0; integrate with G1 contracts.  
**Gates:** G2-open and G2-surface.

- [ ] **DERIV-01:** Complete MATH-01 formula audit and tests.
- [ ] **DERIV-02:** Complete periodic-root conditioning and implicit-derivative tests.
- [ ] **DERIV-03:** Expose frozen and moving-linear-seed modes; add rotating-eigendirection tests.
- [ ] **DERIV-04:** Validate curved-wall events, same-first-hit continuation, and transversality checks.
- [ ] **DERIV-05:** Implement and validate the invariant-circle formulation, predictor, corrector, and sensitivity.
- [ ] **DERIV-06:** Validate one general open-line bundle and its connection-length sensitivity.

**Deliverables:** mathematical specification, manufactured-case suite, derivative convergence report.  
**G2-open exit:** periodic, branch, general-launch, and wall-event derivatives pass their stated semantics.  
**G2-surface exit:** corrected torus geometry and transform sensitivities pass their stated label/gauge semantics.

The two gates are deliberately separate: the open-line benchmark need not wait for the complete torus solver when it can use independently validated core-protection constraints.

### Milestone 3 — Complete constrained optimization and physical validation

**Owner:** ESSOS; pyna supplies numerical/physical validation.  
**Dependencies:** G1 and G2-open; surface integration also requires G2-surface.  
**Gate:** G3.

- [ ] **OPT-01:** Implement one scaled multi-variable optimizer with explicit hard constraints or a declared constrained merit/filter strategy.
- [ ] **OPT-02:** Add refreshed-objective acceptance, predicted-versus-actual improvement logging, and numerical-noise handling.
- [ ] **OPT-03:** Implement atomic commit, restart, objective-version changes, and deliberate history resets.
- [ ] **OPT-04:** Add rejection-reason taxonomy and topology-regime transition handling.
- [ ] **OPT-05:** Integrate 3-D wall validation, target exemptions, and sampled-clearance convergence checks.
- [ ] **OPT-06:** Solve small multi-variable manufactured problems through repeated accepted refreshes, not just one favorable step.

**Deliverables:** runnable optimizer, reproducible checkpoints, end-to-end integration tests.  
**G3 exit:** feasible accepted steps satisfy the declared optimization rule; invalid/rejected trials leave physical accepted state intact; restart reproduces continuation behavior within numerical tolerance.

### Milestone 4 — Complete Demonstration A

**Owner:** ESSOS benchmark, supported by pyna.  
**Dependencies:** G3 and the relevant benchmark qualification checks.  
**Gate:** G4.

- [ ] **OPEN-01:** Freeze the wall, targets, coil controls, constraints, training/held-out launches, and scoring protocol before comparative optimization.
- [ ] **OPEN-02:** Establish current-only control authority, then selected shape controls where justified.
- [ ] **OPEN-03:** Run matched finite-difference and derivative-mode comparisons.
- [ ] **OPEN-04:** Run repeated starts, held-out launches, adverse topology cases, and refined-tracer validation.
- [ ] **OPEN-05:** Report physical gains, rejection behavior, timing, memory, and uncertainty.

**G4 exit:** a reproducible multi-variable open-line result meets the predeclared physical criteria and supports only the comparative claims actually measured.

### Milestone 5 — Complete Demonstration B

**Owner:** ESSOS benchmark and pyna torus solver.  
**Dependencies:** G3, G2-surface.  
**Gate:** G5.

- [ ] **SURF-01:** Qualify a vacuum field with resolved nested surfaces and suitable non-resonant labels.
- [ ] **SURF-02:** Freeze geometry/transform goals and constraints under a well-posed surface-label convention.
- [ ] **SURF-03:** Run predictor-order and implicit-sensitivity experiments.
- [ ] **SURF-04:** Compare predictor/corrector continuation against the same corrector without prediction.
- [ ] **SURF-05:** Compare against a strong direct-surface optimization baseline with matched physical evaluation.
- [ ] **SURF-06:** Study deliberate approaches to resonance and document valid-domain limits.

**G5 exit:** geometric and transform improvements survive independent surface/tracing checks, and predictor/implementation advantages are measured rather than inferred.

### Milestone 6 — Package and release the vacuum capability

**Owners:** both repositories.  
**Dependencies:** G4 and G5.  
**Gate:** G6.

- [ ] **REL-01:** Stabilize documented APIs, optional dependencies, installed-wheel tests, and compatibility pins.
- [ ] **REL-02:** Produce benchmark reproduction commands, data manifests, and machine-readable results.
- [ ] **REL-03:** Document limitations, invalid regimes, surrogate modes, and known failures.
- [ ] **REL-04:** Demonstrate a tokamak-like prescribed-background verification path through the same interfaces.
- [ ] **REL-05:** Review the manuscript/technical claims against the evidence matrix.

**G6 exit:** a fresh supported environment can reproduce the reduced verification suite and both benchmark workflows from recorded inputs.

### Milestone 7 — Add self-consistent finite-beta response

**Owners:** equilibrium/field adapter, ESSOS, and pyna integration.  
**Dependencies:** G6 for a production extension; a lightweight adapter design may proceed earlier.  
**Gate:** G7.

- [ ] **BETA-01:** Select and document the equilibrium model, boundary formulation, profiles, and controlled parameters.
- [ ] **BETA-02:** Implement equilibrium residual/response interfaces and validate total field sensitivities.
- [ ] **BETA-03:** Verify vacuum and prescribed-background limits before pressure continuation.
- [ ] **BETA-04:** Demonstrate constrained optimization with refreshed equilibrium and topology validation.

**G7 exit:** accepted gradients include the relevant plasma response; the calculation is no longer merely a frozen-background perturbation problem.

---

## 9. Demonstration A: vacuum open-field-line control

### 9.1 Scientific question and working hypothesis

Can tracked-manifold and open-line sensitivities produce useful, constrained changes to divertor footprints and connection lengths in a vacuum stellarator field, while preserving a protected core?

**Hypothesis A:** within a well-conditioned tracked regime, moving-geometry derivatives predict refreshed physical changes more faithfully than the frozen-reference derivative, and enable a useful multi-variable optimization at competitive end-to-end cost.

The result must test both parts. Better derivative agreement alone is not a completed divertor-design demonstration.

### 9.2 Starting configuration and qualification

Use the existing QA period-five integration fixture as a development seed, not as the entire scientific experiment. Its wall is explicitly a frozen regression geometry. [S07]

Before optimization:

- Qualify the baseline orbit, branches, wall geometry, first hits, core region, and field validity with refined tracing.
- Determine which trim-current and shape controls exist; adding a new control is a documented design change, not an assumed repository capability.
- Compute a small local controllability study using scaled Jacobian singular values or directional responses.
- Remove exactly redundant controls, fix current/field normalization conventions, and document weakly controlled targets.
- Select targets based on an independent physical design specification. Do not generate the target by applying the same known one-variable perturbation that the optimizer will recover.

A preliminary feasibility study may adjust the target before the benchmark is frozen. Archive that selection process so a conveniently solvable final case is not presented as a blind test.

### 9.3 Design variables and staged problem sizes

Begin with bounded currents for fixed coil geometry. Release selected low-order coil-shape coefficients only after field and current sensitivities pass their tests.

Proposed scaling sizes are **8, 32, and 128 active variables**, where independent and physically meaningful controls are available. These are experimental targets, not claims of present capability. A smaller physically meaningful basis is preferable to padding the benchmark with ineffective variables.

Use the same parameter basis and bounds for matched derivative comparisons. Include coil spacing, coil-to-plasma spacing, current limits, and geometric regularization appropriate to the chosen fixture. Numerical values must come from the actual design specification, not arbitrary generic clearance defaults.

### 9.4 Launches, labels, and targets

Include multiple seeds and both relevant branch sides. Add a general non-manifold bundle if that is needed to test the stated open-field-line scope.

Freeze optimization launches and their weights. Create held-out launches before tuning the optimizer, and keep their results out of the objective. Define targets on fixed physical wall patches, with a clear treatment of wall coordinates and patch seams.

For footprint residuals, an example is

$$
r_{\mathrm{hit},j}=\frac{d_{\mathcal W}(X_{\mathrm{hit},j},\mathcal T_j)}{\ell_{\mathrm{target}}},
$$

where the wall metric and target set $\mathcal T_j$ are specified. A patch-distance residual is not identical to a prescribed-point residual; select the formulation before comparison.

Connection-length goals may use a target interval rather than an exact value when that better represents the design intent. Keep the launch convention fixed so changing a numerical manifold seed distance cannot masquerade as improved connection length.

### 9.5 Experiment ladder

| Stage | Purpose | Required comparison |
|---|---|---|
| A0: manufactured cases | Isolate derivative and event correctness | Analytic derivatives and same-discretization finite differences. |
| A1: current-only QA case | Establish a complete constrained workflow | Frozen-reference, moving geometry, and refreshed finite differences. |
| A2: selected shape controls | Test meaningful coil-design freedom | Matched basis, constraints, physical targets, and optimizer settings. |
| A3: robustness/refinement | Test scientific relevance | Held-out launches, repeated starts, wall/field/trace refinement. |
| A4: scaling | Measure implementation advantage | End-to-end costs across variable counts and trace lengths. |

The conventional coil-fit baseline can show that explicit edge objectives deliver additional control. It is not a substitute for the same-objective finite-difference comparison, because it optimizes a different loss.

### 9.6 Metrics and qualification criteria

Report at least:

- Physical footprint error and target-patch coverage, on optimized and held-out launches.
- Connection-length statistics and unresolved/missing-hit counts.
- Minimum validated 3-D clearance outside authorized target-approach regions.
- Protected-core diagnostics and all engineering-constraint margins.
- Root/eigenpair conditioning, event transversality, topology changes, and rejection reasons.
- Same-discretization and refreshed-geometry derivative errors.
- End-to-end time, peak memory, field evaluations, rejected-trial costs, and convergence reliability.

**Success gate:** the predeclared physical targets are met within numerical uncertainty, the result remains feasible after refinement, and any advantage exceeds the observed variability/error floor. Do not claim success solely because a weighted objective falls or one selected wall hit moves correctly.

---

## 10. Demonstration B: vacuum closed-surface control

### 10.1 Scientific question and working hypothesis

Can a non-resonant geometric predictor plus a corrected invariant-surface sensitivity enable reliable coil optimization of closed-surface geometry and rotational-transform characteristics?

**Hypothesis B:** in its stated non-resonant regime, the predictor has the expected perturbative accuracy and reduces the work needed to continue corrected surfaces; the resulting optimizer achieves validated physical goals at competitive cost relative to a strong direct-surface baseline.

### 10.2 Starting configuration

Choose a coil-only QA field with independently resolved nested surfaces. Use a parameter regime appropriate to closed-surface control rather than assuming the edge-manifold regime is also suitable.

Select several usable surface labels. Establish initial invariance residuals, rotational transforms, Fourier convergence, regularity, and separation from relevant low-order resonances. Include at least one held-out surface or a denser set of radial diagnostics not directly optimized.

Retain both current and geometry normalization constraints to prevent trivial scaling or relabelling from appearing as a design improvement.

### 10.3 Physical target and surface selection

Decide whether to control geometry at selected flux labels, maintain selected rotational transforms while changing geometry, or pursue another well-posed continuation convention. Do not impose incompatible fixed labels and frequencies on a generic field.

Use gauge-invariant geometric metrics where possible: physical signed distances, shape descriptors, or explicitly aligned curve/surface comparisons. Raw Fourier-coefficient differences are meaningful only after parameterization, phase, and coordinate conventions are fixed.

If quasi-symmetry is included, define the diagnostic and coordinates independently. Do not add it merely because the seed configuration is called QA; the first closed-surface result can focus on geometry and transform.

### 10.4 Experiment ladder

| Stage | Purpose | Required evidence |
|---|---|---|
| B0: manufactured integrable map/flow | Verify gauge, label, and frequency normalization | Known invariant circles and analytic sensitivities. |
| B1: non-resonant amplitude sweep | Verify first-order predictor | Resolved $O(\epsilon^2)$ post-predictor residual range. |
| B2: nonlinear correction | Verify actual surface solve | Collocation, off-grid, Fourier, and tracer convergence. |
| B3: constrained multi-variable coil optimization | Establish design capability | Refreshed surfaces, physical target improvements, feasible coils. |
| B4: baseline comparisons | Establish relative merit | Same corrector without predictor; matched finite differences; direct-surface baseline. |
| B5: resonance approach | Characterize limits | Conditioning growth, loss of validity, and explicit fallback behavior. |

### 10.5 Strong comparator and scope of the comparison

Giuliani and collaborators' direct vacuum coil optimization uses an inner magnetic-surface formulation that can also represent approximate surfaces in island/chaotic fields. It is a stronger comparator than a normal-field fit alone. [S14]

There are two fair comparison regimes:

**Common valid domain:** compare methods on well-resolved nested surfaces, matched controls, and shared physical metrics.

**Capability boundary:** investigate approaching resonance or lost surfaces, while clearly stating that the methods optimize different mathematical objects when one continues with approximate surfaces. Do not label an approximate-surface result a failed exact-torus solve, or an unavailable exact torus an implementation defect by default.

Implementing or adapting the baseline requires its own validation against the original method and examples. Reserve part of the experiment work for baseline competence rather than treating it as an afterthought.

### 10.6 Metrics and success gate

Measure corrected invariance residuals, off-grid errors, surface regularity/nesting, physical shape error, transform error, protected/held-out surface behavior, correction iterations, linear-solver work, total field evaluation cost, time, and memory.

**Success gate:** the predicted perturbative order is observed before the numerical floor; corrected sensitivities pass independent checks; optimized physical outcomes survive refinement; any speed/design advantage is supported at matched accuracy and with complete costs.

---

## 11. Comparative experiments and claim criteria

### 11.1 Separate claims and comparisons

| Proposed claim | Appropriate experiment | What does not establish it |
|---|---|---|
| More accurate derivative | Analytic cases plus same-objective and refreshed-object checks | Differentiating a simpler surrogate and comparing only to its own finite differences. |
| Better geometric model | Seed/parameterization convergence against reconstructed manifolds or corrected surfaces | Adding eigendirection derivatives without checking local manifold error. |
| Better optimization capability | Physical outcomes under matched controls and engineering limits | Lower values of two differently defined losses. |
| Faster implementation | Matched-accuracy end-to-end runs including validation and rejected work | A single JIT-warmed kernel timing. |
| Better scaling | Variable-count, trace-length, and surface-resolution sweeps | One small problem or an unfavorable baseline implementation. |
| Broader valid domain | Deliberate tests at difficult regimes with honest object definitions | Suppressing invalidity checks or relabelling approximate objects as exact. |

### 11.2 Baseline matrix

Use the following comparisons where applicable:

1. **Finite differences of the same fixed discretization:** differentiation correctness and cost.
2. **Finite differences with corrected/refreshed geometry:** physical derivative semantics.
3. **Frozen-reference versus moving geometry:** value and cost of the additional geometry response.
4. **Predictor plus corrector versus the identical corrector alone:** value of the analytic predictor.
5. **Conventional field-fit objective versus topology-aware design:** added design capability, evaluated with shared external metrics.
6. **Strong surface/island optimization methods where relevant:** scientific context and capability comparison, not a straw-man benchmark.

Geraldini, Landreman, and Paul already demonstrated adjoint island-size/residue sensitivities and gradient-based reduction of stochasticity in vacuum-field examples. Do not claim the broad invention of gradient-based magnetic-topology optimization. [S13]

The contribution to investigate is a unified, tracked treatment of open and closed objects, their moving geometry and events, explicit derivative-validity conditions, and robust cross-repository implementation. A literature update before manuscript submission is required; the two cited papers are starting comparators, not a complete priority survey.

### 11.3 Experimental controls

Freeze the following before comparative production runs:

- Physical geometry, initial designs, parameter basis, targets, constraints, and normalization.
- Optimizer budget and stopping criteria; provide both matched-evaluation and matched-wall-time views where useful.
- Numerical accuracy targets and the process used to attain them in each method.
- Held-out launches/surfaces, initial perturbation seeds, and failure accounting.
- Hardware class, precision, compilation policy, and CPU/GPU timing conventions.

Use several deterministic initial perturbations; a proposed minimum is five when run costs permit. Report every preregistered case, not only successful or visually compelling runs. For timing, separate run-to-run timing variability from optimization variability across starting points.

### 11.4 Thresholds for claims

Define physical success thresholds from the design problem. Define derivative and residual tolerances from the error budget. Choose required effect sizes before final comparisons; do not set them after seeing which method wins.

A measured difference is not persuasive when comparable to field interpolation bias, validation uncertainty, or start-to-start variability. Report uncertainty or ranges alongside effect sizes. Do not imply that a small number of numerical experiments proves universal superiority.


## 12. Verification and numerical error budgets

### 12.1 Test hierarchy

Maintain four levels: fast unit tests, cross-repository integration tests, physical validation tests, and reproducible benchmark runs. A passing benchmark must not replace unit tests for corner cases; a green unit suite must not be presented as physical benchmark validation.

### 12.2 Minimum verification matrix

The test IDs below are proposed requirements, not assertions that corresponding files already exist.

| Test ID | Component | Required property |
|---|---|---|
| V01 | Coordinates/units | Cartesian/cylindrical round trip and correct physical field components. |
| V02 | Map conventions | Correct span, field-period factor, angular lift, and map power. |
| V03 | Magnetic-flux measure | Convergence of flux-weighted map preservation in a solenoidal manufactured field. |
| V04 | Fixed-angle validity | Explicit rejection near unusable toroidal field, without denominator clipping. |
| V05 | Periodic root | Correct root and implicit sensitivity for a manufactured map. |
| V06 | Root conditioning | Small residual but large sensitivity is diagnosed near a singular root. |
| V07 | Monodromy | Total derivative includes motion of the root as well as direct field dependence. |
| V08 | Rotating eigendirection | Moving-geometry mode agrees with analytic/reference response; frozen mode is identified as different. |
| V09 | Branch identity | Stable/unstable, sign, parity, phase, and multi-generation labels remain consistent. |
| V10 | Local manifold accuracy | Seed-size or parameterization-order convergence separates model error from derivative error. |
| V11 | Wall event | Analytic plane and curved-wall first-hit sensitivities agree under refinement. |
| V12 | Event failure | Grazing, competing hits, missing hits, and unwrapped-phase switches reject or change regime explicitly. |
| V13 | Clearance geometry | A non-axisymmetric example distinguishes section distance from true 3-D distance. |
| V14 | Whole-leg checking | A between-sample near-collision is found by refinement or a valid conservative bound. |
| V15 | Target exemption | No unauthorized earlier wall encounter is hidden by the terminal exclusion. |
| V16 | Transform definition | Finite-time angular increments, zero modes, and long-time transform are distinguished. |
| V17 | Torus gauge/label | Phase shifts do not alter physical output; different label conventions yield correctly different derivatives. |
| V18 | Torus predictor | Expected perturbative residual order appears above the numerical floor. |
| V19 | Torus corrector | Dense off-grid, Fourier, and tracing refinement confirm invariance and regularity. |
| V20 | Torus sensitivity | Corrected-solution JVP/VJP agrees with directional finite differences. |
| V21 | Approximate surface | Nonzero-residual sensitivities differentiate the declared stationarity problem. |
| V22 | General open-line bundle | Moving/fixed launch conventions and connection-length sensitivities work without a manifold target. |
| V23 | Snapshot integrity | Trial mutation and serialization do not alter the accepted state. |
| V24 | Acceptance | A topology-valid but merit-worsening trial is rejected under the chosen rule. |
| V25 | Refresh mismatch | A locally improved surrogate but degraded refreshed physical objective is not silently accepted. |
| V26 | Rollback/restart | Rejection preserves physical state; restart restores objective and optimizer semantics. |
| V27 | Cache identity | Current, shape, wall, numerical-config, and topology changes invalidate the correct cached data. |
| V28 | Integration/package | Required backends run; installed-wheel imports and resources work outside repository roots. |
| V29 | Current-only field cache | Cached linear combination matches fresh field values and current derivatives. |
| V30 | Finite-beta adapter | Total response matches re-solved equilibrium finite differences and appropriate limiting cases. |

### 12.3 Derivative verification procedure

For a normalized direction $v$ in design space, compare the implemented directional derivative against

$$
D_hJ(q)[v]=\frac{J(q+hv)-J(q-hv)}{2h}.
$$

Use a logarithmic step sweep rather than a single favorable $h$. Plot or tabulate the convergence region, truncation regime, and roundoff/root-error regime.

Run two distinct procedures:

**Discretization check:** use the same numerical configuration and exactly the same frozen quantities as the implemented derivative.

**Physical-object check:** reconstruct the moving orbit, eigendirection/manifold, wall event, or torus on each side, preserving discrete identity and aligning gauge/orientation. No nearest-branch substitution is allowed to make the finite difference look smooth.

For vector residuals, compare JVPs directly. Check VJP consistency with the dot-product identity

$$
u^\mathsf{T}(Jv)\approx (J^\mathsf{T}u)^\mathsf{T}v.
$$

Use scaled absolute errors and relative errors with an explicit derivative floor. A near-zero exact derivative should be tested by an absolute criterion; division by an arbitrary unit floor must not hide a bad nonzero derivative.

### 12.4 Proposed starting tolerances

These are **initial validation targets**, not measured capabilities or universally appropriate physical thresholds:

| Test family | Starting target | Qualification |
|---|---|---|
| Well-conditioned float64 manufactured derivatives of order one | Relative directional error around `1e-7` or better over a resolved step range | Require the analytic problem and solver tolerances to support it; report the full sweep. |
| Physical-coil directional derivatives | Relative agreement around `1e-3` or better for resolved nonzero derivatives | Separate model, field, trace, root, and event error; use an absolute criterion near zero. |
| First-order predictor | Log-log slope approaching two over a resolved amplitude range | Do not fit points dominated by numerical floor or near-resonant failure. |
| Physical outcome uncertainty | Clearly smaller than the claimed improvement and active feasibility margins | Set case-specific distances, transform errors, and tolerances before final runs. |

A failed target triggers diagnosis or an explicit, scientifically justified specification revision. It does not justify silently loosening tolerance until a test passes.

### 12.5 Error-budget components

Track at least:

- Coil quadrature and field evaluation error.
- Production-grid interpolation and divergence error.
- Field-line integration and dense-output error.
- Periodic-root residual and conditioning amplification.
- Local manifold representation and seed-size error.
- Wall geometry, event location, and first-hit identity uncertainty.
- Torus Fourier/collocation truncation, correction residual, and gauge conditioning.
- Optimizer stopping tolerance and surrogate-to-refreshed mismatch.

Refine individual components before refining everything simultaneously. This identifies the dominant limitation and avoids claiming unnecessary precision from one tightly solved subproblem.

For a physical margin or improvement $\Delta$, the uncertainty bound or demonstrated refinement variation must be substantially smaller than $\Delta$. Report absolute physical accuracy separately from accurate differential cancellation.

### 12.6 Failure-status vocabulary

Adopt structured codes such as `root_not_converged`, `root_ill_conditioned`, `eigenpair_not_separated`, `field_parameterization_invalid`, `event_grazing`, `first_hit_changed`, `trace_domain_exceeded`, `surface_not_available`, `surface_resonant`, `clearance_violated`, `merit_rejected`, and `backend_error`.

The exact names are provisional. The requirement is machine-readable, distinct failure semantics that survive serialization and support aggregate benchmark reporting.

---

## 13. Performance engineering

### 13.1 Profile before restructuring kernels

Instrument complete candidate evaluations before making broad speed claims. Attribute time and peak memory to field construction, host/device transfer, grid sampling, root solves, map tracing, derivative evaluation, manifold/surface correction, wall events, production validation, and rejected proposals.

Report cold-start compilation separately from warmed operation. GPU timings must synchronize completed work; asynchronous dispatch time is not solver runtime.

### 13.2 Shared candidate evaluation

Compute shared roots, monodromy, launch reconstruction, trajectories, and wall hits once per candidate where the requested outputs permit reuse. Keep a dependency-aware request mechanism so a torus-only objective does not trace unrelated manifolds.

Test cache-disabled and cache-enabled modes for identical scientific results within the same numerical configuration. Record cache hit rates and memory costs rather than assuming caching is beneficial.

### 13.3 Current-only production-field acceleration

For fixed coil geometry and independently linear current controls,

$$
B(x;c)=B_{\mathrm{fixed}}(x)+\sum_j c_j B_j(x).
$$

Cache or tile unit-current contributions to update a production field grid without reevaluating every coil integral at every accepted step. The field cache depends on geometry, grid, quadrature, normalization, and the control basis.

Do not retain a full basis grid when its memory footprint is excessive. Use chunked evaluation, grouping, or recomputation based on measured time-memory tradeoffs. Invalidate geometry-dependent data when shape changes.

### 13.4 Derivative scaling

Expose JVP/VJP operations and reuse factorizations where mathematically valid. Compare forward and reverse modes using the actual number of controls and outputs. Do not assume reverse mode wins for every residual bundle.

Consider checkpointing for long trajectories and multiple shooting when single-shooting sensitivities become numerically problematic. These are follow-on engineering options, not substitutes for documenting genuine physical conditioning.

Higher-order local manifold representations should be introduced because they reduce a measured error/cost bottleneck, not because they make the architecture look more complete.

### 13.5 Performance experiment matrix

Measure scaling against active-variable count, number of launches, branch generations/connection length, torus modes, and production-grid size. Use the same declared accuracy target throughout each comparison.

Provide at least an end-to-end CPU reference. Add GPU results on authorized hardware when available; compare both hardware and algorithm fairly. Do not treat access to faster hardware as mathematical superiority.

---

## 14. Continuous integration, packaging, and compatibility

### 14.1 Required test lanes

| Lane | Required behavior |
|---|---|
| ESSOS core | Ordinary installation and unrelated core features remain usable without pyna. |
| pyna analytic/JAX | Run manufactured topology and derivative tests with explicit float64 configuration where required. |
| ESSOS–pyna pinned integration | Install the compatible pair and required backend; fail on missing required imports or collected tests. |
| Production tracing | Exercise actual production tracing, wall events, and cross-backend validation; a stub-only lane is insufficient. |
| Packaging | Build/install distributions and test from outside the source tree. |
| Physical regression | Run reduced QA and prescribed-background tokamak-like checkpoints. |
| Full benchmark | Run through an explicit, reproducible workflow on suitable resources; archive full results. |

Optional local tests may still use skip behavior. A required integration lane must not pass merely because all integration tests were skipped. Report expected test identifiers or counts and enforce required execution.

### 14.2 Cross-repository change management

Every integration-sensitive pull request must state the tested ESSOS and pyna hashes. Land reusable pyna primitives and tests first, then the ESSOS adapter, then the benchmark that depends on them.

Maintain an exact pair for reproducibility and a separately identified forward-compatibility lane when feasible. Do not confuse the latter's branch-tip results with validation of the pinned scientific baseline.

### 14.3 Compatibility and installation

Prefer targeted module extraction and compatibility wrappers over immediate deep package relocation. If subpackages or package data are added, verify that built distributions contain them and that examples do not depend on the current working directory. [S12]

Document supported dependency versions from actual CI results rather than guessing new version bounds. Record native-backend build requirements and fail with actionable dependency errors.

Keep accelerated backends optional for unrelated use, but explicit and mandatory for workflows that claim to validate them.

---

## 15. Reproducibility, artifacts, and execution safeguards

### 15.1 Proposed artifact organization

These are proposed paths and schemas to create during implementation, not files claimed to exist in either baseline repository.

```text
ESSOS/
  docs/manifold_optimization/
    plan.md
    conventions.md
    mathematical_contracts.md
    validation_protocol.md
    benchmark_protocol.md
    limitations.md
  benchmarks/manifold_optimization/
    cases/
    configs/
    run_case.py
    summarize_results.py

pyna/
  tests/                       # extend the established test layout
  docs/                        # mathematical primitives and validity contracts

run_artifacts/<case>/<run_id>/
  manifest.json
  resolved_config.yaml
  initial_state/
  accepted_states/
  iterations.jsonl
  metrics.json
  timings.json
  validation/
  figures/
```

Keep reusable mathematical specifications in pyna where appropriate and link from ESSOS; avoid maintaining conflicting copies. Store large outputs outside source control according to the repositories' existing policies.

### 15.2 Run manifest

Every benchmark run must record repository hashes, working-tree state, input checksums, dependency lock/environment, hardware, precision, command/configuration, random seeds, field/wall model, physical units, optimizer settings, derivative mode, validity thresholds, and the exact benchmark protocol version.

Accepted-step logs must include objective components, constraint margins, residual/conditioning diagnostics, first-hit status, numerical settings, step size, trust controls, predicted/actual decrease, refresh counts, and timing. Rejected steps must remain visible in cost and reliability statistics.

A figure should be regenerable from saved numerical outputs. Avoid manually edited result tables or figures that cannot be traced to a run identifier.

### 15.3 Illustrative configuration contract

The following is a **proposed schema example**, not an executable interface at the pinned commits. `null` values mark required case-specific inputs; validation must reject a production benchmark until they are supplied.

```yaml
schema_version: 1
case_id: vacuum_open_qa
field_mode: vacuum_coils
repositories:
  essos_commit: 3c1dc4f  # expand to the full resolved hash in the run manifest
  pyna_commit: 4f032d7
numerics:
  precision: float64
  optimization_config: null
  validation_config: null
  error_budget: null
derivatives:
  manifold_mode: moving_linear_seed
  event_geometry: physical_wall
  differentiate_launch_geometry: true
problem:
  coil_geometry: null
  design_variable_basis: null
  bounds_and_scales: null
  wall_geometry: null
  physical_targets: null
  engineering_constraints: null
  optimization_launches: null
  held_out_launches: null
acceptance:
  production_validation_for_every_accepted_step: true
  refreshed_merit_required: true
  same_first_hit_required: true
benchmark:
  initial_perturbation_seeds: null
  physical_success_thresholds: null
  comparative_effect_thresholds: null
  stopping_and_resource_budgets: null
```

Separate optimization and validation numerics, but require a documented convergence relationship between them. Add an analogous surface configuration with explicit label, gauge, winding, resolution, and resonance policy.

### 15.4 Execution safeguards

Honor each repository's existing contributor and agent instructions. Work in explicitly authorized project worktrees; treat files outside those project workspaces as read-only unless separately authorized.

Do not expose secrets, private notebooks/data, unpublished material, host details, or sensitive logs through web searches, telemetry, public issues, or new sharing destinations. Use only necessary reviewed files for authorized remote compute, with authenticated encrypted transfer and no destructive synchronization.

Use authorized GPU resources for development. On NERSC, use an interactive allocation with the required GPU count for interactive development/testing, subject to site policy and existing project instructions. Record hardware and resource settings in benchmark manifests.

Do not install software globally, modify unrelated environments, rewrite shared history, or publish results merely because this plan proposes an experiment. Repository changes, remote execution, and publication must follow the actual workspace permissions and project policies.

### 15.5 Deliverable standards

Every scientific result should include its input/configuration, data, reproduction command, validation report, and limitations. Never fabricate successful runs, replace unavailable dependencies with unreported stubs, or present planned performance as measured performance.

---

## 16. Finite-beta extension

### 16.1 Three explicit field modes

| Mode | Model | Permitted description |
|---|---|---|
| Vacuum coils | Field generated by declared external currents in the plasma region | Coil-only vacuum optimization. |
| Prescribed plasma background plus coils | Plasma/background field is supplied and held fixed under selected coil changes | Frozen-background or vacuum-perturbation optimization. |
| Self-consistent equilibrium response | Plasma/equilibrium state changes according to a specified equilibrium problem | Finite-beta equilibrium-aware optimization, within that model's validity. |

Zero pressure does not imply zero plasma current. The distinction must appear in configurations, benchmark names, and scientific claims.

### 16.2 Equilibrium-response contract

Let an equilibrium state $z$ satisfy

$$
\mathcal E(z,c,\beta)=0,
\qquad B=B(x;z,c,\beta).
$$

For fixed $\beta$ and a nonsingular, appropriately gauged equilibrium linearization,

$$
\frac{dz}{dc}=-\mathcal E_z^{-1}\mathcal E_c,
\qquad
\frac{dB}{dc}=B_c+B_z\frac{dz}{dc}.
$$

Implement this through linear/adjoint solves or validated response operators rather than explicit inversion. Add the appropriate pressure/profile derivatives when those quantities are controls.

The adapter must define which currents, profiles, boundary quantities, constraints, and normalization choices remain fixed. Plasma-response sensitivity is not determined by the word “finite beta” alone.

### 16.3 Adapter design tasks

- [ ] Specify fixed-boundary versus free-boundary assumptions and how coil changes enter the equilibrium problem.
- [ ] Define primal equilibrium state, residual, convergence, and conditioning diagnostics.
- [ ] Provide field values plus JVP/VJP operations for total field response.
- [ ] Permit an external non-JAX equilibrium solver through a tested response interface; do not require an immediate rewrite of that solver.
- [ ] Verify independent re-solved-equilibrium finite differences, including input/profile normalization.
- [ ] Extend candidate acceptance to require both equilibrium and topology validity.

A nested-surface equilibrium model and an edge/open-field-line model may have different domains. Document their coupling and boundary assumptions instead of assuming every equilibrium backend supplies a self-consistent open-field-region model.

### 16.4 Extension ladder

1. Recover the vacuum limit and the existing vacuum benchmarks where the selected equilibrium model permits that limit.
2. Verify a prescribed-background problem as an intermediate integration test without calling it self-consistent.
3. Add small-pressure or small-response continuation with independent field-response checks.
4. Increase complexity only after equilibrium and topology sensitivities converge separately.
5. Repeat a constrained open- or closed-object design experiment with re-solved equilibrium at accepted candidates.

Keep an explicit failure policy near equilibrium bifurcations, singular responses, or incompatible topology assumptions. Do not regularize a singular response and report it as an exact derivative without qualifying the approximation.

---

## 17. Pull-request sequence and dependency map

### 17.1 Suggested reviewable pull requests

Each pull request should contain tests and documentation for its own change. Avoid combining formula corrections, module moves, and performance rewrites in one patch.

| PR | Repository | Scope | Dependencies / evidence |
|---|---|---|---|
| PR-01 | Both / integration configuration | Baseline manifest, required test lane, fixture inventory | G0 evidence; do not change scientific algorithms. |
| PR-02 | pyna + ESSOS docs | Shared conventions and derivative-mode semantics | Review before API changes. |
| PR-03 | pyna | Transform-helper audit, analytic regressions, correction/rename | V16 and authorized theory reconciliation. |
| PR-04 | pyna | Immutable snapshots, status schema, serialization | V23; no unwanted public-API break. |
| PR-05 | ESSOS | Separate targets, topology state, and candidate evaluation | PR-02/04; torus-only/open-line construction tests. |
| PR-06 | pyna | Root/monodromy diagnostics and sensitivities | V05–V07. |
| PR-07 | pyna | Moving-linear-seed derivative and representation tests | PR-06; V08–V10. |
| PR-08 | pyna | Curved-wall event validation and event semantics | V11–V12; phase and first-hit tests. |
| PR-09 | pyna + ESSOS adapters | General open-line bundle and connection length | V22; no artificial manifold dependency. |
| PR-10 | pyna | 3-D clearance and target-exemption validation | V13–V15. |
| PR-11 | ESSOS | Complete acceptance, atomic state commit, restart | PR-05–10; V24–V26. |
| PR-12 | pyna | Gauged torus formulation and nonlinear corrector | PR-02/03; V17/V19. |
| PR-13 | pyna | Torus predictor and implicit sensitivity | PR-12; V18/V20/V21. |
| PR-14 | ESSOS | Closed-surface objective/constraint integration | PR-05/11/13. |
| PR-15 | ESSOS | Demonstration A configuration and measured results | G3; preregistered A protocol. |
| PR-16 | ESSOS | Demonstration B configuration and measured results | PR-14; preregistered B protocol. |
| PR-17 | Both as needed | Profile-guided caching, JVP/VJP, and memory improvements | V27/V29 and matched-accuracy benchmark evidence. |
| PR-18 | Both | Packaging, compatibility, reproducibility, limitation docs | G4/G5 plus V28; vacuum release gate. |
| PR-19 | Adapter + both | Finite-beta response integration | G6 and V30; separate review of physical model. |

Several rows may be split into smaller patches. Parallel work on torus mathematics is appropriate once conventions are agreed; the dependency order is not a requirement to serialize unrelated development.

### 17.2 Critical path

```text
G0 baseline
  |
  +--> G1 contracts/state integrity
  |      |
  |      +--> G2-open --> G3 optimizer/validation --> G4 open demonstration
  |      |
  |      +--> G2-surface --> closed-surface integration --> G5 surface demonstration
  |
  +--> benchmark protocol and validated baseline implementations

G4 + G5 + compatibility/reproducibility --> G6 vacuum release
G6 + independently validated equilibrium response --> G7 finite-beta release
```

Profile and compatibility checks run throughout. Large performance refactors should follow correctness evidence so that a speed change does not conceal a changed scientific problem.

---

## 18. Risk register and fallback decisions

| Risk | Detection | Response |
|---|---|---|
| Intended transform definition differs from the review's interpretation | Reconcile governing equations, internal theory, and analytic tests | Correct the review/API interpretation as warranted; do not preserve an accusation at the expense of the definition. |
| Weak hyperbolicity amplifies root/manifold errors | Scaled condition estimates and sensitivity convergence | Restrict steps, choose a better-conditioned case, or explicitly study the limit. |
| Linear seeds fail after long propagation | Seed-size and local-invariance residual tests | Use smaller controlled seeds or a higher-order parameterization; report any changed launch convention. |
| First-hit switches make objectives nonsmooth | Independent full first-hit tracing and phase identity | Reject/localize the boundary; outer rediscovery and model restart if allowed. |
| Closed torus is unavailable at a requested label | Corrector failure plus resonance and independent tracing diagnostics | Change a scientifically permissible label/goal or use an explicitly approximate-surface formulation. |
| Gauge or relabelling creates false improvement | Gauge-invariance tests and physical-space validation | Fix the parameterization convention and recompute the metric. |
| Production grid introduces false topology | Grid/field divergence and tracer-refinement studies | Refine representation or use a direct-field reference; do not privilege the production result without convergence. |
| Clearance surrogate misses a 3-D collision | Independent 3-D and between-sample validation | Reject the design and refine geometry/trajectory checking. |
| Target lacks control authority | Scaled local controllability analysis and feasibility pilot | Revise controls/targets before protocol freeze; report limits afterward. |
| Full response is not faster than a frozen surrogate | Complete cost and optimization-quality measurements | Report a correctness/design tradeoff rather than claiming a universal speedup. |
| Strong baseline wins or gives comparable results | Matched experiments and repeated starts | Narrow the contribution to demonstrated capability or robustness; publish the measured comparison honestly. |
| Validation dominates runtime | End-to-end profiling | Current-basis caching, batching, shared evaluations, or validated multi-fidelity screening. |
| Cross-repository drift breaks integration | Exact compatibility pair and required CI | Pin, fix upstream/downstream in small patches, and preserve regression coverage. |
| Finite-beta response is unreliable | Equilibrium residual, response finite differences, limiting cases | Keep the feature experimental and retain the validated vacuum/prescribed modes. |

Fallbacks may change the scientific question. Record that change rather than using the fallback's result to support the original claim unchanged.

---

## 19. First implementation backlog

Begin with this bounded set before adding new physics or large abstractions:

- [ ] **NEXT-01:** Create authorized worktrees at the pinned baselines; resolve full hashes and capture environment/build information.
- [ ] **NEXT-02:** Run and inventory the current relevant tests; establish an integration lane that cannot silently skip the feature.
- [ ] **NEXT-03:** Add a shared conventions note covering field components, map span, transform, surface label, and launch semantics.
- [ ] **NEXT-04:** Add the finite-time/long-time transform regression and reconcile the helper's intended definition.
- [ ] **NEXT-05:** Add a rotating-eigendirection manufactured test, initially documenting the expected difference between frozen and moving geometry.
- [ ] **NEXT-06:** Add mutation/aliasing tests for accepted snapshots before refactoring them.
- [ ] **NEXT-07:** Add an acceptance regression where topology is valid but the refreshed merit worsens.
- [ ] **NEXT-08:** Establish baseline timing and numerical-error breakdowns for one reduced QA trace and one manufactured torus problem.
- [ ] **NEXT-09:** Implement the smallest contract changes needed to make those tests and diagnostics reliable.
- [ ] **NEXT-10:** Freeze the two benchmark qualification protocols, leaving physical limits unset until their fixtures are measured and specified.

**First tangible milestone:** one clean, reproducible cross-repository environment plus a diagnostic test suite that exposes unresolved mathematical/acceptance issues explicitly. A larger plot gallery is not the first milestone.

---

## 20. Release definition of done

### Vacuum capability

- [ ] Exact compatible repository versions and environment requirements are recorded.
- [ ] Required integration tests exercise both differentiable and production backends.
- [ ] Physical targets, surface labels, coordinate conventions, and derivative modes are documented.
- [ ] Transform semantics and analytic counterexamples are resolved.
- [ ] Periodic, manifold, general-open-line, wall-event, and torus derivatives pass the appropriate tests.
- [ ] Invalid or ill-conditioned regimes return explicit statuses.
- [ ] Accepted state is isolated, serializable, restartable, and updated atomically.
- [ ] Refreshed physical feasibility and optimization acceptance are both enforced.
- [ ] Clearance and deposition claims are limited to what the geometry/model actually validates.
- [ ] Demonstrations A and B use independent physical specifications, multiple controls, and held-out/refined validation.
- [ ] Baselines are competent and comparisons use matched quantities or clearly identified different objectives.
- [ ] Runtime includes compilation policy, field preparation, correction, validation, and rejected work.
- [ ] Scientific claims match measured evidence and uncertainty.
- [ ] A prescribed-background tokamak-like test verifies portability without mislabelling it as coil-only vacuum equilibrium.
- [ ] Reproduction artifacts and known limitations are available within authorized project storage.

### Finite-beta capability

- [ ] Equilibrium model, profiles, constraints, boundary assumptions, and controlled parameters are explicit.
- [ ] Total field sensitivities include the relevant plasma response.
- [ ] Independent re-solved-equilibrium derivative checks and appropriate limiting cases pass.
- [ ] Accepted designs pass equilibrium and topology validation together.
- [ ] Frozen-background and self-consistent results are labelled separately.

**Final standard:** a new contributor can understand what is optimized, reproduce the relevant result, identify where its derivative is valid, and distinguish demonstrated performance from a proposed improvement.

---

## 21. Source register

These primary sources establish the pinned-code observations and two initial literature comparators. Proposed interfaces, task IDs, thresholds, benchmark designs, and derivations in this plan are not claims that those sources already implement or prove them.

The source review performed for this document is selective. No source below establishes that a test was executed or that a benchmark succeeded during preparation of this plan.

| ID | Source | Used for |
|---|---|---|
| [S01] | pyna `torus_deformation.py` at `4f032d7` | Fourier convention, deformation helpers, transform helper, and shear-aware predictor context. |
| [S02] | ESSOS `manifold.py` at `3c1dc4f` | Field adapter ownership and existing differentiable topology wrappers. |
| [S03] | pyna `manifold_correspondence.py` at `4f032d7` | Labels, orientation, snapshot contents, and mutability observation. |
| [S04] | pyna `jax_manifold.py` at `4f032d7` | Frozen-reference derivative semantics and backend split. |
| [S05] | ESSOS `manifold_driver.py` at `3c1dc4f` | Topology-validation backtracking semantics. |
| [S06] | ESSOS `.github/workflows/build_test.yml` at `3c1dc4f` | Inspected CI triggers and installation steps. |
| [S07] | ESSOS `tests/test_qa_manifold_coils.py` at `3c1dc4f` | QA regression fixture, optional imports, and period-five integration context. |
| [S08] | pyna `jax_periodic.py` at `4f032d7` | Implicit periodic-point implementation. |
| [S09] | pyna `jax_strike.py` at `4f032d7` | Implicit local wall-event implementation. |
| [S10] | ESSOS `manifold_optimization.py` at `3c1dc4f` | Existing target and continuation-state organization. |
| [S11] | pyna `manifold_leg_clearance.py` at `4f032d7` | Production leg validation and terminal-exclusion convention. |
| [S12] | ESSOS `pyproject.toml` at `3c1dc4f` | Package configuration to audit if introducing subpackages. |
| [S13] | A. Geraldini, M. Landreman, E. Paul, *An adjoint method for determining the sensitivity of island size to magnetic field variations*, arXiv:2102.04497 | Prior island/residue sensitivity and gradient-based vacuum-field optimization. |
| [S14] | A. Giuliani, F. Wechsung, A. Cerfon, M. Landreman, G. Stadler, *Direct stellarator coil optimization for nested magnetic surfaces with precise quasi-symmetry*, arXiv:2210.03248 | Strong direct-surface optimization comparator and approximate-surface distinction. |

[S01]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/toroidal/torus_deformation.py
[S02]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/essos/manifold.py
[S03]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/topo/manifold_correspondence.py
[S04]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/topo/jax_manifold.py
[S05]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/essos/manifold_driver.py
[S06]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/.github/workflows/build_test.yml
[S07]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/tests/test_qa_manifold_coils.py
[S08]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/topo/jax_periodic.py
[S09]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/topo/jax_strike.py
[S10]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/essos/manifold_optimization.py
[S11]: https://raw.githubusercontent.com/WenyinWei/pyna/4f032d7/pyna/topo/manifold_leg_clearance.py
[S12]: https://raw.githubusercontent.com/hongkelu/ESSOS/3c1dc4f/pyproject.toml
[S13]: https://arxiv.org/abs/2102.04497
[S14]: https://arxiv.org/abs/2210.03248
