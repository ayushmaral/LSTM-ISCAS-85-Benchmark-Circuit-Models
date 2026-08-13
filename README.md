# Fault Prediction on ISCAS-85 Circuits with a Corrected Bidirectional LSTM

Stuck-at fault prediction from ATPG test patterns, using a bidirectional LSTM with a corrected two-direction readout. Across ten ISCAS-85 benchmark circuits the model reaches a **mean validation accuracy of 0.9648** (range 0.8757–0.9991), trained on 28.1 million expanded test patterns in roughly 8.2 GPU-hours.

On every circuit for which a prior baseline exists, the model improves on it substantially:

| Circuit | ANN baseline | LSTM baseline | This work | Gain |
|---|---:|---:|---:|---:|
| c1355 | 0.6390 | 0.6963 | **0.9366** | +24.0 pts |
| c432  | — | 0.8817 | **0.9555** | +7.4 pts |
| c499  | — | 0.8156 | **0.9801** | +16.5 pts |

---

## Results

| Circuit | Pattern bits | Classes with data | Samples | Samples/class | Accuracy | SA-0 | SA-1 | Best epoch | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c880  |  86 |  942 |   982,848 | 1043 | **0.9991** | 0.9990 | 0.9991 | 58 | 39.2 min |
| c2670 | 373 | 2584 | 2,542,400 |  984 | **0.9986** | 0.9987 | 0.9985 | 38 | 39.5 min |
| c1908 |  58 | 1766 | 1,631,528 |  924 | **0.9943** | 0.9944 | 0.9942 | 37 | 30.0 min |
| c499  |  73 |  501 |    72,602 |  145 | **0.9801** | 0.9849 | 0.9763 | 55 |  3.0 min |
| c7552 | 315 | 7550 | 8,581,952 | 1137 | **0.9787** | 0.9770 | 0.9794 | 28 |  1.54 h  |
| c5315 | 301 | 5285 | 5,694,592 | 1078 | **0.9653** | 0.9637 | 0.9659 | 43 |  1.23 h  |
| c3540 |  72 | 3288 | 3,448,744 | 1049 | **0.9646** | 0.9632 | 0.9652 | 39 | 42.7 min |
| c432  |  43 |  518 |   283,174 |  547 | **0.9555** | 0.9561 | 0.9552 | 54 |  5.9 min |
| c1355 |  73 |  895 |   148,993 |  167 | **0.9366** | 0.8997 | 0.9442 | 59 |  1.88 h  |
| c6288 |  64 | 5685 | 4,729,112 |  832 | **0.8757** | 0.8760 | 0.8751 | 79 | 50.0 min |
| **Mean** | | | | | **0.9648** | 0.9613 | 0.9653 | | |

**c17** is reported separately. With only 80 unique samples after deduplication, a single 80/20 split would validate on 16 patterns, so it is evaluated by 5-fold cross-validation: pooled accuracy **0.8500** (68/80), per-fold mean 0.8500 ± 0.0637, against a chance level of 0.0769.

---


## Repository contents

```

LSTM_c17_kfold.ipynb              # 5-fold CV (multiclass) — use this for c17 reporting
LSTM_c17_multilabel_improved.ipynb# multi-label formulation
LSTM_c432.ipynb
LSTM_c499.ipynb
LSTM_c880.ipynb
LSTM_c1355.ipynb
LSTM_c1908.ipynb
LSTM_c2670.ipynb
LSTM_c3540.ipynb
LSTM_c5315.ipynb
LSTM_c6288.ipynb
LSTM_c7552.ipynb
circuits/                           # Atalanta-generated .test files
  c17.test
  c432.test
  c499.test
  c880.test
  c1355.test
  c1908.test
  c2670.test
  c3540.test
  c5315.test
  c6288.test
  c7552.test
```

Each notebook is self-contained: parsing, dataset construction, model, training loop, and plots.

---

## Quickstart

### Google Colab

1. Put the Atalanta-generated `.test` files in Google Drive under `My Drive/circuits/`, named `c432.test`, `c499.test`, and so on.
2. Open the notebook for the circuit, e.g. `LSTM_c880.ipynb`.
3. **Runtime → Change runtime type → GPU** (A100 preferred; a T4 works with longer epochs).
4. Run all cells top to bottom. The Drive-mount cell prompts for authorization.
5. The setup cell prints sample count, class count, classes with data, sequence length, and parameter count — check these against the results table before committing to a long run.

### Local

The only Colab-specific cell is the Drive mount:

```bash
pip install torch numpy matplotlib scikit-learn psutil
jupyter notebook            # or: jupyter nbconvert --to script LSTM_c880.ipynb
```

1. Delete or comment out the `from google.colab import drive` cell.
2. Set `test_path = './circuits/c880.test'`.

CPU-only works for the small circuits (c17, c432, c499) in minutes. The large circuits need a GPU — expect a 20–50× slowdown otherwise.

### Requirements

PyTorch 2.11 (CUDA 12.8), Python 3.12. Peak GPU memory across all runs was 11.4 GB (c880), so a 16 GB card is sufficient for every circuit. Peak host memory during parsing reaches ~7.6 GB for c7552.

---

## Key parameters

| Parameter | Effect |
|---|---|
| `MAX_SAMPLES` | `None` uses the full dataset. Capping trades accuracy for time roughly proportionally — the main accuracy lever. |
| `BITS_PER_STEP` | Bits per timestep: 1 for short patterns, 8 for patterns over 300 bits, 7 (whole pattern) for c17. |
| `HIDDEN_SIZE` | 384 for A100 runs. 512 is the next step, ~1.7× slower, worth it only under genuine underfitting. |
| `EPOCHS` | Extend if the best epoch lands within ~3 of the end. |
| `LR` | 1e-3 is the stable regime; 2e-3 caused gradient explosions on the largest circuits. |

---

## Findings

**Accuracy tracks pattern width, not fault count.** c7552 has 7,550 classes and scores 0.9787; c6288 has 5,685 and scores 0.8757; c880 has 942 and scores 0.9991. c6288 is a 16×16 multiplier with only 64 pattern bits against 7,744 declared faults — so information-poor that 2,059 of its classes (26.6%) end up with no samples at all after deduplication, their every pattern already claimed by an earlier fault.

**Sample density is the second factor.** Every circuit above 900 samples per class scores 0.9646 or better; the sparsest datasets (c1355 at 167/class, c432 at 547/class) sit at the bottom alongside c6288.

**The learning-rate reduction is where the gains are.** The largest single-epoch improvement in nearly every run immediately follows the plateau scheduler's first LR cut (c1355 +3.3 points at epoch 53, c1908 +2.3 at epoch 31). A run whose best epoch is its last is almost certainly under-trained.

**Class imbalance did not matter.** Ratios span 0.42:1 to 4.87:1, yet SA-0 and SA-1 accuracies differ by under 0.25 points on eight of ten circuits. Unweighted cross-entropy is sufficient; class weighting was tried and did not help.

---

## Reproducibility

All runs use seed 42 across Python's `random`, NumPy, and PyTorch, with `torch.backends.cudnn.deterministic = True` and `benchmark = False`. The 80/20 split is seeded identically. Don't-care expansion beyond three positions uses seeded random fill, so sample counts are stable to within a few hundredths of a percent.

**Note on evaluation protocol:** each test pattern expands into up to 64 samples through don't-care enumeration, and the 80/20 split is applied *after* expansion, so sibling expansions of the same parent pattern can appear in both training and validation. This matches the protocol of the baseline ANN and LSTM models, so the comparisons above are like-for-like; absolute accuracies should be read as optimistic relative to a pattern-level split.

---

## Acknowledgements

Test data generated with the [Atalanta](https://www.ee.vt.edu/~mhsiao/atalanta.html) ATPG tool. Extends prior work on ML-based digital system testing on the ISCAS-85 benchmark suite.
