# Q-MAML on the LHCO (m_X, m_Y) signal grid: feasibility notes (2026-10-16)

Working notes for the round-3 reply to Junyong: the feasibility checks and pilot behind `qmaml_hep.experiments.lhco_routes`.

## Why this task family

Q-MAML maps a physical task descriptor to circuit parameters. The Hamiltonian tracks failed against warm starts
because nearby tasks were interchangeable: the nearest training task's optimum was almost as good a start. A
classification family only helps if (i) the descriptor is physical, (ii) a classifier tuned for one task transfers
badly to its neighbours, and (iii) a strong conditioned baseline exists to compare against.

The LHCO 2020 R&D signal grid satisfies all three:

- **Signal:** W′(3.5 TeV) → X(qq) Y(qq), (m_X, m_Y) on a 12 × 12 grid from 50 to 600 GeV, about 100k events per point.
  Zenodo [11188685](https://zenodo.org/records/11188685).
- **QCD:** the LHCO R&D background, high-level features in [3733787](https://zenodo.org/records/3733787) (1M events).
  The larger Pythia sample in [22685072](https://zenodo.org/records/22685072) has 22.5M events, 2.74M of them in the
  signal region.
- **Descriptor:** (m_X, m_Y). This is a physics parameter, exactly as in the paper's Heisenberg task (J_x, J_y, J_z).
- **Use case:** searches simulate signal on a coarse mass grid and need classifiers in between. The standard
  classical answer is a parameterised network (Baldi et al. 2016). PAWS (2024) uses this exact dataset.

## Setup

- **Features:** lighter jet mass, mass difference, and τ₂₁ and τ₃₂ of the lighter and heavier jet. The τ ratios
  cancel the normalisation differences between the files.
- **Signal region:** 3.3 < m_jj < 3.7 TeV.
- **Metric:** background rejection R = 1/ε_b at ε_s = 0.3. AUC is saturated, so it can't separate methods.
- **Reference:** a BDT trained per grid point on 20k signal events ("oracle").
- **Data hygiene:**
  - (50, 750) is a stray off-grid point with 2.4k events; excluded.
  - (550, 50) and (600, 50) have fewer than 30k signal-region events.
  - Tasks are close to symmetric under m_X ↔ m_Y, so splits must hold out mirror pairs together. The grids
    below do.

## Results

### 1. Data checks

| Check | Result |
|---|---|
| Grid point (500, 100) vs the original R&D signal | KS ≤ 0.031 on every feature |
| 22.5M-event QCD vs the original 1M-event QCD, signal region | KS ≤ 0.005 on every feature; every event has label 0 |

### 2. Saturation (1M QCD test events)

| Quantity | Range | Median |
|---|---|---|
| Oracle AUC | 0.962–0.994 | 0.985 |
| Oracle rejection | 939–15,873 | 5,587 |

AUC can't separate methods; rejection can.

### 3. Nearest-mass transfer is expensive

Round 1, 1M-event QCD. Values are the median log₁₀ R difference from the oracle.

| Offset | Loss | Rejection lost more than 2× on |
|---|---|---|
| 50 GeV | −0.90 decades | 92% of points |
| 100 GeV | −1.59 decades | 100% |
| 200 GeV | −2.03 decades | 100% |

This is the property the Hamiltonian tracks lacked.

### 4. Conditioning on the masses works classically, given a smooth model

Large QCD sample. Held-out grid points, log₁₀ R relative to the oracle, median with 95% bootstrap CI over points:

| Training grid | Parameterised MLP (3×128) | Parameterised BDT | Nearest-mass | MLP beats nearest on |
|---|---|---|---|---|
| 100 GeV spacing (36 points; 108 held out) | −0.03 [−0.05, −0.02] | −0.66 | −0.82 | 100% |
| 150 GeV spacing (16 points; 128 held out) | −0.29 [−0.41, −0.21] | −1.08 | −1.00 | 98% |

Trees handle the masses as step functions; the MLP interpolates smoothly. **Classical headroom is small at 100 GeV
spacing (about 7%) and about 2× at 150 GeV.** Any claim for Q-MAML has to be made against this baseline, and against
its quantum counterpart (masses as circuit inputs).

### 5. A 6-qubit circuit has enough capacity

Large QCD sample, 2 seeds averaged. The circuit is a 6→6 linear projection, then RY(π·tanh) encoding,
StronglyEntanglingLayers and a linear head on ⟨Z⟩.

| Point | Oracle | Circuit d4 (128 params) | Circuit d2 (92) | Circuit removed (56) | MLP (1,346) | Logistic regression |
|---|---|---|---|---|---|---|
| (100, 100) | 3,378 | 2,828 | 2,615 | 3,066 | 3,584 | 245 |
| (300, 300) | 15,873 | 13,763 | 11,580 | 12,020 | 14,955 | 198 |
| (150, 450) | 7,519 | 7,189 | 6,516 | 6,132 | 7,725 | 142 |
| (500, 200) | 6,667 | 5,320 | 4,688 | 4,556 | 6,830 | 123 |

The depth-4 circuit reaches 80–96% of the oracle and 20–50× logistic regression. A good classifier needs mass
windows, which a linear model can't represent. The circuit roughly ties its no-circuit twin, so the circuit itself
adds nothing; that matches round 2. The Q-MAML question below doesn't depend on that, because it compares
sources of θ inside the same circuit.

### 6. Few-shot per point

Round 1, 1M-event QCD, 36 points. Median log₁₀ R relative to the oracle:

| Model | K = 64 | K = 512 |
|---|---|---|
| BDT | −0.53 | −0.21 |
| Logistic regression | ≈ −1.5 | ≈ −1.5 |

### 7. Q-MAML pilot

#### Setup

All arms use the same classifier: features → (front end) → 6 encoding angles → FastPQCModel (6 qubits, depth 2,
θ) → linear head on ⟨Z⟩. The arms differ only in how the masses enter and where θ comes from.

- **Training:** the 36-point grid at 100 GeV spacing. Every arm gets the same budget: 3000 steps × 8 tasks × 128 + 128
  events, Adam at 3e-3, untuned.
- **Evaluation:** zero-shot on 108 held-out points. 2 seeds (1 for joint-deep). QCD test pool of 100k events.
- **Learner:** the paper's architecture, 2 → 256 → 256 → 36, float64.

| Arm | Mass information enters via | Feature front end | Conditioning params |
|---|---|---|---|
| joint | none | Linear 6→6 | 0 |
| param | Linear(8→6) on [x, m] | Linear | ~50 |
| **qmaml** | θ = Learner(m) | Linear | 76k |
| hyper-enc | MLP(m) generates the Linear(6→6) encoding | Linear | 77k |
| param-deep | MLP([x, m]) → angles | MLP 8→256→256→6 | 70k (shared with the features) |
| **qmaml-deep** | θ = Learner(m) | MLP 6→256→256→6 | 76k + 68k front end |
| joint-deep | none | MLP 6→256→256→6 | 0 |
| nearest | a circuit trained on the nearest training point | Linear | — |
| pertask | a circuit trained on the target point (16 points) | Linear | — |

#### Zero-shot results

Held-out points; paired log₁₀ R differences, median and 95% bootstrap CI over points.

| Comparison | Difference | Q-MAML better on |
|---|---|---|
| qmaml − param | +0.38 [+0.33, +0.47] | 81% |
| qmaml − nearest | +0.27 [+0.15, +0.35] | 76% |
| qmaml − joint | +0.93 [+0.89, +1.02] | 100% |
| qmaml − pertask | −0.15 [−0.27, −0.10] | — |
| **qmaml − hyper-enc (matched capacity)** | **−0.04 [−0.08, +0.01]** | 43% |
| **qmaml − param-deep** | **−0.29 [−0.32, −0.27]** | **0%** |
| qmaml − classical parameterised MLP (check 4) | −0.39 [−0.42, −0.34] | — |
| qmaml − oracle BDT | −0.43 [−0.46, −0.41] | — |

**Same deep feature network for both routes.** This isolates where the masses enter.

| Comparison (held-out) | Difference | First better on |
|---|---|---|
| **qmaml-deep − param-deep** | **−0.30 [−0.34, −0.25]** | **2%** |
| qmaml-deep − joint-deep (no masses) | +0.92 [+0.83, +1.07] | 100% |
| qmaml-deep − qmaml (linear front end) | +0.04 [+0.00, +0.11] | 60% |

On the training points, qmaml-deep − param-deep is −0.15 [−0.20, −0.11].

So the deep feature network is not what makes param-deep better. Feeding the masses in with the features beats
generating θ from them, by 2× at equal capacity. The masses themselves are essential: without them the same
network loses 0.9 decades.

#### Other observations

- **Seeds:** the gains over param and nearest hold for both seeds (+0.31 / +0.49 and +0.25 / +0.39).
- **Distance:** the gain over nearest grows with distance to the nearest training point: +0.24 at ≤ 60 GeV,
  +0.63 at 60–100 GeV.
- **Extrapolation edge** (m = 600 GeV, outside the training range): the gain over nearest shrinks to +0.09.
- **Few-shot (Algorithm 2):** θ-only adaptation, 50 steps, lr 0.01, untuned, on 16 points. It does not improve
  Q-MAML's zero-shot output: −0.11 at K = 16, +0.02 at K = 64. At K = 64 the adapted nearest-mass circuit ties
  Q-MAML (−0.01 [−0.15, +0.22]).

#### Reading

1. Q-MAML clearly beats the two baselines that sank it on the Hamiltonian tracks: nearest-task transfer and an
   unconditioned shared model. On this family that is a real result.
2. Against a capacity-matched alternative, the advantage disappears. Q-MAML ties a hypernetwork that conditions
   the encoding instead of θ. It loses about 2× to a parameterised network with a circuit head, which is itself at
   the classical parameterised-MLP level, so the circuit adds nothing. The loss remains when Q-MAML gets the same
   deep feature network: **the route matters**. Conditioning the features beats conditioning θ. A plausible
   mechanism: a new mass hypothesis moves the jet-mass windows, which is a feature-space operation. θ acts after a
   fixed encoding and can only reshape the angular response.
3. Pilot caveats: 2 seeds, no tuning, one grid spacing, an output scale of 1, and few-shot on only 16 points.
   Nothing here is paper-grade yet.

## Proposed question for the paper

> With signal simulated only on a coarse (m_X, m_Y) grid, does a Learner that maps the masses to circuit angles
> (Q-MAML) give a better classifier at unsimulated masses than the same circuit given the masses as inputs, than
> transfer from the nearest simulated mass, and than the classical parameterised network?

**Axes:** grid spacing (100, 150 and 200 GeV), zero-shot vs few-shot (K = 16, 64, 512), and the extrapolation
corner.

**Protocol:** round 2's, unchanged. Paired per-point differences, hierarchical bootstrap over seeds and points,
Holm correction, tuning on validation points.

## Files

- `lhco_feasibility.py`: round 1 (normalisation, saturation, few-shot, transfer, parameterised BDT).
  Output: `lhco_feasibility.json`.
- `lhco_check2.py`: parameterised MLP and circuit capacity on the large QCD sample. Output: `check2_big.json`.
- `lhco_qmaml_pilot.py`: Q-MAML pilot (joint, param, qmaml, nearest, pertask, few-shot). Output: `pilot.json`.
- `lhco_pilot_matched.py`: capacity-matched arms. Output: `pilot_matched.json` (hyper-enc, param-deep) and
  `pilot_deep.json` (qmaml-deep, joint-deep).
- `lhco_summarize.py`: prints the tables above from the JSON files. `summary_output.txt` is its output.
- `email_draft.md`: draft reply to Junyong. Private: gitignored and kept local only.
- Data lives in `data/lhco/` (gitignored), with checksums verified against Zenodo.
  `qcd_big_sr_feats.npy` caches the signal-region features of the large QCD sample.
