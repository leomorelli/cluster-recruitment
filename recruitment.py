"""Small radial-profile pipeline based on analysis.ipynb. See README.md."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import false_discovery_control, t

__version__ = '0.3.1'


def load_profiles(path, channels):
    """Read selected MATLAB fields: each is a list of structs with intensity."""
    try:
        data = loadmat(path, simplify_cells=True, variable_names=list(channels))
    except NotImplementedError:  # MATLAB v7.3 uses HDF5.
        import mat73
        data = mat73.loadmat(str(path))
    result = {}
    for key in channels:
        if key not in data:
            raise ValueError(f"{path}: missing MATLAB field {key!r}.")
        records = data[key]
        if isinstance(records, dict):  # Single-region struct after squeezing.
            records = [records]
        try:
            result[key] = np.stack([
                np.asarray(row['intensity'], dtype=float).reshape(-1)
                for row in np.asarray(records, dtype=object).reshape(-1)
            ])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}/{key}: need equal-length intensity vectors, one per region.") from error
        if result[key].shape[1] == 0 or not np.isfinite(result[key]).all():
            raise ValueError(f"{path}/{key}: profiles must be nonempty and finite.")
    return result


def enrichment(x, inside=slice(0, 5), outside=slice(30, 54)):
    """Log2(inner radial-bin mean / outer radial-bin mean), one score per row."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("Need a finite (n_regions, n_distances) matrix.")
    for window in (inside, outside):
        if (window.step not in (None, 1) or window.start is None or window.stop is None
                or not 0 <= window.start < window.stop <= x.shape[1]):
            raise ValueError(f"Invalid bin slice {window} for {x.shape[1]} columns.")
    if max(inside.start, outside.start) < min(inside.stop, outside.stop):
        raise ValueError("Inside and outside windows must not overlap.")
    a, b = x[:, inside].mean(axis=1), x[:, outside].mean(axis=1)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Inner and outer means must be positive; check background correction.")
    return np.log2(a / b)


def recruitment(profiles, random_profiles, inside=slice(0, 5), outside=slice(30, 54)):
    """Notebook score and empirical p/q-values, using same-cell random controls."""
    if np.shape(profiles)[1:] != np.shape(random_profiles)[1:]:
        raise ValueError("Cluster and random profiles need the same distance bins.")
    observed, null = [enrichment(x, inside, outside) for x in (profiles, random_profiles)]
    score = observed - null.mean()
    # Count upper-tail ties conservatively without allocating an n_clusters x M array.
    p = (1 + len(null) - np.searchsorted(np.sort(null), observed, side='left')) / (len(null) + 1)
    q = false_discovery_control(p, method='by')
    return pd.DataFrame(dict(enrichment=observed, score=score, p=p, q=q,
                             detected=(q < .05) & (score > 0) & (observed > 0)))


def _contrast(groups, coefficients, scores_only=False):
    """Welch-Satterthwaite contrast; conditional on independent sampled regions."""
    n = np.array([len(g) for g in groups])
    delta = float(np.dot(coefficients, [np.mean(g) for g in groups]))
    result = dict(difference=delta, ratio=2.0 ** delta, p_conditional=np.nan,
                  ci95_low=np.nan, ci95_high=np.nan)
    if scores_only or np.any(n < 2):
        return result
    v = np.array([c*c*np.var(g, ddof=1)/len(g) for c, g in zip(coefficients, groups)])
    if v.sum() == 0:
        return result
    se = np.sqrt(v.sum())
    df = v.sum()**2 / np.sum(v**2/(n-1))
    margin = t.ppf(.975, df) * se
    result.update(p_conditional=float(2*t.sf(abs(delta/se), df)),
                  ci95_low=float(delta-margin), ci95_high=float(delta+margin))
    return result


def compare_conditions(stim, random_stim, unstim, random_unstim,
                       inside=slice(0, 5), outside=slice(30, 54), reference=None,
                       scores_only=False, paired=False):
    """Condition A minus B, correcting both with their own random controls.

    Optional reference: four arrays in the same order, paired by region within
    each array. paired=True preserves row pairing between groups; a two-item
    tuple independently controls cluster pairing and random-region pairing.
    """
    arrays = [stim, random_stim, unstim, random_unstim]
    if len({np.shape(x)[1:] for x in arrays}) != 1:
        raise ValueError("All arrays must use the same radial bins.")
    groups = [enrichment(x, inside, outside) for x in arrays]
    if reference is not None:
        if len(reference) != 4 or any(np.shape(a) != np.shape(b) for a, b in zip(arrays, reference)):
            raise ValueError("Reference needs four paired arrays matching the protein shapes.")
        groups = [g - enrichment(ref, inside, outside) for g, ref in zip(groups, reference)]
    flags = (paired, paired) if isinstance(paired, bool) else tuple(paired)
    if len(flags) != 2:
        raise ValueError("paired must be a bool or (clusters_paired, random_paired).")
    terms, coefficients = [], []
    for a, b, sign, pair in zip(groups[:2], groups[2:], [1, -1], flags):
        if pair:
            if a.shape != b.shape:
                raise ValueError("Paired groups need equal counts and matching region order.")
            terms.append(a - b)
            coefficients.append(sign)
        else:
            terms.extend([a, b])
            coefficients.extend([sign, -sign])
    return _contrast(terms, coefficients, scores_only)


def proces_data(profiles, title='Radial profiles'):
    """Vectorized version of the notebook plotter; return long data and figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    x = np.asarray(profiles)
    df = pd.DataFrame(x).rename_axis('region').reset_index().melt(
        id_vars='region', var_name='dist', value_name='intensity')
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    sns.boxplot(data=df, x='dist', y='intensity', color='#5A9AB5', fliersize=2, ax=ax)
    ticks = np.arange(0, x.shape[1], max(1, x.shape[1] // 9))
    ax.set(xticks=ticks, xticklabels=ticks, xlabel='Distance bin', title=title)
    return df, fig


def run(cluster_files, random_files, channels=None, labels=None, reference=None,
        inside=slice(0, 5), outside=slice(30, 54), out_dir='results', scores_only=False,
        paired=False):
    """One selected field for EACH cluster/random file pair, in input order."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    files, randoms = [[Path(p).resolve() for p in paths] for paths in (cluster_files, random_files)]
    if len(files) not in (1, 2):
        raise ValueError("Supply one or two cluster .mat files.")
    channels = ['IF_IRF'] * len(files) if channels is None else list(channels)
    if len(channels) != len(files):
        raise ValueError("Supply exactly one channel per cluster/random file pair, in input order.")
    labels = [f'{p.stem}:{key}' for p, key in zip(files, channels)] if labels is None else list(labels)
    if len(randoms) != len(files) or len(labels) != len(files) or len(set(labels)) != len(labels):
        raise ValueError("Supply one random file and one unique label per cluster file.")
    # Read selected fields once per path, including when a file is supplied twice.
    requests = {}
    for f, r, key in zip(files, randoms, channels):
        if f == r:
            raise ValueError("Cluster and random measurements must be in separate files.")
        for path in (f, r):
            if not path.is_file():
                raise ValueError(f"File not found: {path}.")
            requests.setdefault(path, set()).update([key] + ([reference] if reference else []))
    cache = {path: load_profiles(path, sorted(keys)) for path, keys in requests.items()}
    loaded = [(cache[f], cache[r]) for f, r in zip(files, randoms)]
    if len({group[key].shape[1] for pair, key in zip(loaded, channels) for group in pair}) != 1:
        raise ValueError("All selected profiles must use the same radial bins.")
    tables, nulls, summary, comparisons = [], [], [], []
    for i, ((cluster, random), key) in enumerate(zip(loaded, channels)):
        table = recruitment(cluster[key], random[key], inside, outside)
        null = enrichment(random[key], inside, outside)
        if reference:
            if any(group[key].shape != group[reference].shape for group in (cluster, random)):
                raise ValueError("Reference and protein rows must be paired in both files.")
            ref = recruitment(cluster[reference], random[reference], inside, outside)
            table['relative_reference'] = table.score - ref.score
        tables.append(table)
        nulls.append(null)
        summary.append(dict(condition=labels[i], channel=key, n_clusters=len(table), n_random=len(null),
                            mean_enrichment=table.enrichment.mean(), mean_random_enrichment=null.mean(),
                            mean_score=table.score.mean(), detected_fraction=table.detected.mean(),
                            empirical_p_floor=1/(len(null)+1)))
        comparisons.append(dict(comparison='cluster_vs_random', condition_A=labels[i], condition_B='',
                                channel_A=key, channel_B=key, paired_clusters=False, paired_random=False,
                                **_contrast([table.enrichment, null], [1, -1], scores_only)))
    pair_flags = (False, False)
    if len(files) == 2:
        pair_flags = (paired or files[0] == files[1], paired or randoms[0] == randoms[1])
        arrays = [group[key] for pair, key in zip(loaded, channels) for group in pair]
        common = dict(condition_A=labels[0], condition_B=labels[1], channel_A=channels[0],
                      channel_B=channels[1], paired_clusters=pair_flags[0], paired_random=pair_flags[1])
        comparisons.append(dict(comparison='recruitment_change', **common,
                                **compare_conditions(*arrays, inside, outside, scores_only=scores_only, paired=pair_flags)))
        if pair_flags[1] and nulls[0].shape != nulls[1].shape:
            raise ValueError("Paired random groups need equal counts and matching region order.")
        random_groups = [nulls[0] - nulls[1]] if pair_flags[1] else nulls
        comparisons.append(dict(comparison='random_control_change', **common,
                                **_contrast(random_groups, [1] if pair_flags[1] else [1, -1], scores_only)))
        if reference:
            refs = [group[reference] for pair in loaded for group in pair]
            comparisons.append(dict(comparison='relative_reference_change', **common,
                                    **compare_conditions(*arrays, inside, outside, reference=refs,
                                                         scores_only=scores_only, paired=pair_flags)))
    summary, comparisons = pd.DataFrame(summary), pd.DataFrame(comparisons)
    comparisons['q_BY'] = np.nan
    valid = comparisons.p_conditional.notna()
    if valid.any():
        comparisons.loc[valid, 'q_BY'] = false_discovery_control(comparisons.loc[valid, 'p_conditional'].to_numpy(), method='by')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --out directory.")
    out.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for i, ((cluster, random), key) in enumerate(zip(loaded, channels)):
        prefix = f'group{i+1}'
        tables[i].to_csv(out / f'{prefix}_clusters.csv', index_label='region')
        pd.DataFrame({'enrichment': nulls[i]}).to_csv(out / f'{prefix}_random.csv', index_label='region')
        for group, profiles in [('cluster', cluster[key]), ('random', random[key])]:
            _, fig = proces_data(profiles, f'{labels[i]} | {key} | {group}')
            fig.savefig(out / f'{prefix}_{group}.png', dpi=160)
            plt.close(fig)
        all_scores.append(tables[i].assign(condition=labels[i], channel=key))
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    sns.boxplot(data=pd.concat(all_scores), x='condition', y='score', hue='channel', ax=ax)
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_ylabel('Random-corrected enrichment (log2)')
    fig.savefig(out / 'recruitment_scores.png', dpi=160)
    plt.close(fig)
    summary.to_csv(out / 'summary.csv', index=False)
    comparisons.to_csv(out / 'comparisons.csv', index=False)
    config = dict(version=__version__, source_file=str(Path(__file__).resolve()),
                  python_executable=sys.executable,
                  cluster_files=list(map(str, files)), random_files=list(map(str, randoms)),
                  labels=labels, channels=channels, reference=reference, inside=[inside.start, inside.stop],
                  outside=[outside.start, outside.stop], scores_only=scores_only,
                  paired_clusters=pair_flags[0], paired_random=pair_flags[1],
                  scope='conditional region-level tests; not biological replication',
                  file_mapping={f'group{i+1}': dict(condition=label, channel=key)
                                for i, (label, key) in enumerate(zip(labels, channels))})
    (out / 'run.json').write_text(json.dumps(config, indent=2))
    return summary, comparisons


def main():
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__} | {Path(__file__).resolve()}')
    parser.add_argument('--cluster-files', nargs='+', required=True, help='One or two cluster .mat files; first dataset minus second.')
    parser.add_argument('--channels', nargs='+', required=True, help='One MATLAB field per input, in matching order; e.g. IF_IRF IF_IRF or FISH_IRF IF_IRF.')
    parser.add_argument('--random-files', nargs='+', required=True, help='Separate random-region .mat files, in matching dataset order.')
    parser.add_argument('--labels', nargs='+', help='Dataset labels, e.g. stimulated unstimulated or reference POI; defaults to file stem:channel.')
    parser.add_argument('--reference', help='Optional paired reference field, e.g. FISH_IRF.')
    parser.add_argument('--paired', action='store_true', help='Assert matching region order across different files; repeated files are paired automatically.')
    parser.add_argument('--inside', nargs=2, type=int, default=[0, 5], metavar=('START', 'STOP'))
    parser.add_argument('--outside', nargs=2, type=int, default=[30, 54], metavar=('START', 'STOP'))
    parser.add_argument('--out', default='results')
    parser.add_argument('--scores-only', action='store_true', help='Omit aggregate p-values/CIs if region independence is inappropriate.')
    args = parser.parse_args()
    try:
        summary, comparisons = run(args.cluster_files, args.random_files, channels=args.channels,
                                   labels=args.labels, reference=args.reference, inside=slice(*args.inside),
                                   outside=slice(*args.outside), out_dir=args.out, scores_only=args.scores_only,
                                   paired=args.paired)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(summary.to_string(index=False))
    print('\nComparisons (first dataset minus second; p-values in comparisons.csv):')
    print(comparisons[['comparison', 'condition_A', 'condition_B', 'channel_A', 'channel_B',
                       'difference', 'p_conditional', 'q_BY']].to_string(index=False))
    print(f'\nResults: {Path(args.out).resolve()}')
    print('empirical_p_floor: per-cluster p-value resolution, NOT a comparison p-value.')
    print('p_conditional: assumes independent regions within the measured cells.')


if __name__ == '__main__':
    main()
