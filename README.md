# Cluster recruitment

A small pipeline based on `analysis.ipynb`: load MATLAB radial intensity profiles,
plot them, calculate recruitment, and compare two selected datasets.
The implementation is one module, `recruitment.py`.

## Installation

```bash
conda env create -f environmental.yaml
conda activate cluster-recruitment
python -m pip install -e . --no-deps --no-build-isolation
```

Use `cluster-recruitment` or `python recruitment.py` with identical arguments.
SciPy reads MATLAB through v7.2; v7.3 uses `mat73`, included in the environment.

## Input mapping: one channel per file pair

**Every position describes one dataset:** a cluster file, a random file, a label,
and one MATLAB field/channel. The channel at position 1 is selected from BOTH
files at position 1, and likewise for position 2. All filenames are explicit;
there is no automatic random-file discovery.

For example:

| Position | Cluster file | Random file | Channel | Label |
|---|---|---|---|---|
| 1 | Stimulated.mat | Stimulated_random.mat | IF_IRF | stimulated |
| 2 | Not_stimulated.mat | Not_stimulated_random.mat | IF_IRF | unstimulated |

Each selected field must contain a list of MATLAB structs with an `intensity`
vector, matching your notebook:

```python
data['IF_IRF'] = [
    {'intensity': array_of_54_values},  # Region 0
    {'intensity': array_of_54_values},  # Region 1
    # ...
]
```

Rows are regions, columns are distances. Counts may differ between independent
conditions (136 versus 32) and between cluster and random datasets. The distance
grid must match; code checks bin counts, not physical calibration. It does not
find centers, segment images, or process image pixels. Only requested fields are
loaded by SciPy, skipping large unrelated image arrays.

## Tutorial 1: POI in stimulated versus unstimulated cells

```bash
cluster-recruitment \
  --cluster-files Stimulated.mat Not_stimulated.mat \
  --random-files Stimulated_random.mat Not_stimulated_random.mat \
  --labels stimulated unstimulated \
  --channels IF_IRF IF_IRF \
  --inside 0 5 --outside 30 54 \
  --out results_poi
```

This compares IF_IRF in the first condition with IF_IRF in the second. Each is
calibrated against its own IF_IRF random profiles. To compare the reference across
conditions, use `--channels FISH_IRF FISH_IRF` instead.

The terminal prints both the score summary and the statistical comparisons.
For the stimulated-versus-unstimulated p-value, use the **`recruitment_change`
row of `comparisons.csv`, column `p_conditional`**. `cluster_vs_random` rows test
each condition separately; `random_control_change` tests the random baselines.
`q_BY` is a multiple-testing-adjusted value, not the original p-value.

The summary's `empirical_p_floor` (previously called `min_empirical_p`) is only
`1/(n_random + 1)`, the smallest attainable per-cluster empirical p-value.
For 190 and 274 random regions it is 0.005236 and 0.003636. These numbers are
neither measured per-cluster minima nor between-condition p-values. A
`detected_fraction` of zero can coexist with a significant group comparison.


The direction of every effect is **first dataset minus second dataset**.
Inner bins 0-4 and outer bins 30-53 are the defaults (stops are exclusive).
Choose windows from reference cluster dimensions and imaging resolution before
inspecting POI scores/p-values, and keep them consistent across the comparison.

## Tutorial 2: reference versus POI in the same stimulated cell

Supply the same cluster/random files twice, selecting a different field each time:

```bash
cluster-recruitment \
  --cluster-files Stimulated.mat Stimulated.mat \
  --random-files Stimulated_random.mat Stimulated_random.mat \
  --labels reference protein_of_interest \
  --channels FISH_IRF IF_IRF \
  --out results_reference_vs_poi
```

Here the effect is **reference minus POI**, corrected by the corresponding random
channels. Both inputs are stimulated, so labels `reference` and
`protein_of_interest` are clearer than `stimulated` and `unstimulated`.
Labels describe the selected datasets; they do not select files or statistical tests.

Repeated file paths automatically preserve pairing by row: the cluster-channel
comparison uses within-cluster differences, and the random-channel comparison
uses within-random-region differences. Matching row order across the two fields
is essential. A file shared in only one part preserves pairing for that part only.
Different files are treated as independent unless `--paired` explicitly asserts
matching cluster AND random-region order across them. Do not use `--paired` (i.e. for
136-versus-32 unmatched clusters).

These are comparisons of spatial enrichment ratios, not absolute fluorescence
or molecule counts across channels.

## Tutorial 3: analyze a single dataset

```bash
cluster-recruitment \
  --cluster-files Stimulated.mat \
  --random-files Stimulated_random.mat \
  --labels stimulated_POI \
  --channels IF_IRF \
  --out results_single
```

One dataset produces plots, per-cluster recruitment scores and a cluster-versus-
random aggregate comparison. Two datasets additionally produce recruitment and
random-control comparisons. To select two channels from one condition, repeat
its files as in tutorial 2; `--channels` is not a list applied to every file.

## Tutorial 4: functions in a notebook

### Load, plot and calculate recruitment

```python
import matplotlib.pyplot as plt
from recruitment import load_profiles, proces_data, recruitment, compare_conditions, run

cluster = load_profiles('Stimulated.mat', ['IF_IRF', 'FISH_IRF'])
random = load_profiles('Stimulated_random.mat', ['IF_IRF', 'FISH_IRF'])

df, fig = proces_data(cluster['IF_IRF'], 'Stimulated | IF_IRF')
plt.show()
fig.savefig('profile.png', dpi=160, bbox_inches='tight')

inside, outside = slice(0, 5), slice(30, 54)
scores = recruitment(cluster['IF_IRF'], random['IF_IRF'], inside, outside)
print(scores.head())
print(scores['score'].mean(), scores['detected'].mean())
```

### Compare the same-cell reference and POI, preserving pairing

```python
comparison = compare_conditions(
    cluster['FISH_IRF'], random['FISH_IRF'],
    cluster['IF_IRF'], random['IF_IRF'],
    inside, outside, paired=True,
)
print(comparison)
```

For independent conditions, load the second cluster/random files and pass their
arrays as the third/fourth arguments; leave `paired=False` (the default).
For the complete automatic workflow:

```python
summary, comparisons = run(
    cluster_files=['Stimulated.mat', 'Not_stimulated.mat'],
    random_files=['Stimulated_random.mat', 'Not_stimulated_random.mat'],
    channels=['IF_IRF', 'IF_IRF'],
    labels=['stimulated', 'unstimulated'], out_dir='results_python',
)
```

### Optional auxiliary reference normalization

For the POI between-condition command, adding `--reference FISH_IRF` also computes
`relative_reference_change`: the difference between POI and reference recruitment
changes. The selected channels remain IF_IRF for each dataset. This auxiliary
normalization is optional and is not needed for tutorial 2's direct channel
comparison. Protein/reference rows must match within each cluster and random file.
The Python `compare_conditions` equivalent is `reference=[ref_cluster_A,
ref_random_A, ref_cluster_B, ref_random_B]`.

## Scores and significance

For each profile, `E = log2(mean_inner / mean_outer)`. For each selected dataset,
`score = E_cluster - mean(E_random)`. A score of 1 means the enrichment ratio is
twice the geometric-mean random ratio, not twice the protein abundance.
The code averages radial bins equally; annular means need pixel/area weights for
pixel-wise region averages. Use quantitative, unsaturated data with appropriate
instrumental background correction and positive region means; no pseudocounts
are added.

Per-cluster empirical p-values are `(1 + count(E_random >= E_cluster))/(M+1)`.
Their resolution is limited by M. `q` is Benjamini-Yekutieli correction within
each selected dataset; `detected` requires `q < 0.05`, positive score and positive E.
Nondetection does not establish absence of recruitment.

The between-dataset effect is:

```text
difference = (mean(E_cluster_A) - mean(E_random_A))
           - (mean(E_cluster_B) - mean(E_random_B))
```

`ratio = 2**difference`; 0.129 log2 units corresponds to approximately 9.4%
higher random-corrected enrichment in A. Independent inputs use a four-group
Welch-Satterthwaite contrast. For paired channels, form `E_cluster_A - E_cluster_B`
and `E_random_A - E_random_B` first, then compare their means with Welch's test.
The paired random-control test is a one-sample t-test of the random differences.
Both approaches include random-baseline uncertainty. CIs are in log2 units.
The independent four-group calculation matches the original notebook's
`compare_conditions` formula. Testing only the two corrected score vectors
with `ttest_ind` omits uncertainty in the estimated random baselines and can
give a different p-value. Identical four input matrices and windows reproduce
the notebook calculation; matching mean scores alone does not determine p.

`p_conditional` and CIs assume approximately independent sampled regions after
any channel pairing. Pairing fixes within-region channel dependence, not spatial
dependence among clusters. With one cell per condition these are conditional
comparisons of the measured cells; biological generalization needs independent
replication. Use `--scores-only` to omit aggregate p-values/CIs if this assumption
is inappropriate. Empirical per-cluster tests also require representative,
comparably sampled random controls. Reference enrichment is selection-dependent
when the reference defined cluster centers. Nonsignificance does not prove
equivalence, and a group mean difference can be significant even when no individual
cluster passes its q threshold.

## Output files

| File | Contents |
|---|---|
| `summary.csv` | One row per input position: label, selected channel, counts, mean scores, detected fraction |
| `comparisons.csv` | Effect direction, channel_A/B, pairing flags, ratio, conditional p/CI and aggregate q_BY |
| `group1_clusters.csv`, `group2_clusters.csv` | Per-cluster E, score, p, q and detection; optional auxiliary reference score |
| `group1_random.csv`, `group2_random.csv` | Uncalibrated random E values |
| `group1_cluster.png`, `group1_random.png` (and group2 equivalents) | Radial intensity boxplots |
| `recruitment_scores.png` | Recruitment distributions for the selected datasets |
| `run.json` | Code version/location, Python executable, explicit inputs, channel/label mapping, windows and pairing settings |

Region IDs are zero-based MATLAB row indices. Aggregate `q_BY` corrects all
available aggregate comparisons in the run, separately from per-cluster q.
Define study-wide correction separately if running many commands. Use a fresh
empty output directory for each run. Insufficient sample size or zero estimated
variance produces NaN p-values/CIs. Peak offsets and decay widths are not estimated.

## Check the installation

```bash
python -m unittest discover -s tests -v
cluster-recruitment --help
```

Synthetic tests cover MATLAB structs, unequal counts, explicit filenames,
per-input channel selection, same-file pairing, scores/statistics, CLI output
and input errors. Experimental `.mat` files are not bundled.

Methods: [SciPy MAT loading](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html),
[mat73](https://github.com/skjerns/mat7.3),
[FDR correction](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).
