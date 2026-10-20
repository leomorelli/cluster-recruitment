"""Synthetic scientific/data-format checks; no experimental data bundled."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from scipy.io import savemat
from scipy.stats import t, ttest_ind, ttest_rel

import recruitment as rp


def matfile(path, poi, ref):
    # This yields the notebook's list-of-dictionaries after simplify_cells=True.
    def structs(x):
        return np.array([{'intensity': row} for row in x], dtype=object)
    savemat(path, {'IF_IRF': structs(poi), 'FISH_IRF': structs(ref),
                   'input_params': {'test': 1}, 'image_data_set': np.zeros((2, 2))})


def make_files(tmp_path):
    rng = np.random.default_rng(7)
    r = np.arange(54)
    paths = []
    for label, n, amplitude in [('Stimulated', 136, .8), ('Not_stimulated', 32, .05)]:
        cluster = 100 + rng.normal(0, 2, (n, 54))
        cluster *= 1 + amplitude*np.exp(-.5*(r/4)**2)
        reference = 100 + rng.normal(0, 2, (n, 54))
        reference *= 1 + 2*amplitude*np.exp(-.5*(r/4)**2)
        nr = 190 if label == 'Stimulated' else 274
        random = 100 + rng.normal(0, 3, (nr, 54))
        random_ref = 100 + rng.normal(0, 3, (nr, 54))
        path = tmp_path / f'{label}.mat'
        matfile(path, cluster, reference)
        matfile(tmp_path / f'{label}_random.mat', random, random_ref)
        paths.append(path)
    return paths


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = make_files(self.root)

    def test_mat_loading_and_singleton(self):
        tmp_path, files = self.root, self.files
        loaded = rp.load_profiles(files[0], ['IF_IRF', 'FISH_IRF'])
        assert loaded['IF_IRF'].shape == (136, 54)
        assert set(loaded) == {'IF_IRF', 'FISH_IRF'}
        p = tmp_path / 'single.mat'
        matfile(p, np.ones((1, 54)), np.ones((1, 54)))
        assert rp.load_profiles(p, ['IF_IRF'])['IF_IRF'].shape == (1, 54)
        with self.assertRaisesRegex(ValueError, 'missing MATLAB field'):
            rp.load_profiles(p, ['MISSING'])

    def test_exact_score_and_empirical_ties(self):
        tmp_path, files = self.root, self.files
        null = np.full((3, 54), 100.)
        observed = null.copy()
        observed[:, :5] *= 2
        result = rp.recruitment(observed, null)
        assert np.allclose(result.score, 1.)
        assert np.allclose(result.p, .25)  # M=3: no fabricated p=0.
        assert np.allclose(rp.recruitment(null, null).p, 1.)

    def test_welch_and_reference_pairing(self):
        tmp_path, files = self.root, self.files
        rng = np.random.default_rng(4)
        a, b = rng.normal(1, 2, 136), rng.normal(0, 3, 32)
        test = rp._contrast([a, b], [1, -1])
        expected = ttest_ind(a, b, equal_var=False)
        assert np.isclose(test['p_conditional'], expected.pvalue)
        assert np.allclose([test['ci95_low'], test['ci95_high']], expected.confidence_interval())
        assert np.isnan(rp._contrast([a, b], [1, -1], scores_only=True)['p_conditional'])
        g = [np.full((n, 54), 100.) for n in (136, 100, 32, 80)]
        g[0][:, :5] *= 2
        result = rp.compare_conditions(*g)
        assert np.isclose(result['difference'], 1.)
        assert np.isclose(rp.compare_conditions(*g, reference=g)['difference'], 0.)
        with self.assertRaisesRegex(ValueError, 'four paired arrays'):
            rp.compare_conditions(*g, reference=[g[0]])

    def test_invalid_windows_and_signals(self):
        tmp_path, files = self.root, self.files
        a = np.ones((2, 54))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            rp.enrichment(a, slice(0, 5), slice(3, 54))
        with self.assertRaisesRegex(ValueError, 'Invalid bin slice'):
            rp.enrichment(a, outside=slice(30, 55))
        with self.assertRaisesRegex(ValueError, 'positive'):
            rp.enrichment(np.zeros((2, 54)))
        a[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            rp.enrichment(a)

    def test_v73_dispatch(self):
        tmp_path, files = self.root, self.files
        def unsupported(*args, **kwargs):
            raise NotImplementedError('MATLAB 7.3')
        fake = SimpleNamespace(loadmat=lambda path: {'IF_IRF': [{'intensity': np.ones(54)}]})
        with patch.object(rp, 'loadmat', unsupported), patch.dict(sys.modules, {'mat73': fake}):
            assert rp.load_profiles(tmp_path/'example.mat', ['IF_IRF'])['IF_IRF'].shape == (1, 54)

    def test_one_condition_workflow_and_safe_outputs(self):
        tmp_path, files = self.root, self.files
        out = tmp_path/'one'
        randoms = [tmp_path/'Stimulated_random.mat']
        summary, comparison = rp.run(files[:1], randoms, channels=['IF_IRF'], out_dir=out)
        assert summary.n_clusters.tolist() == [136]
        assert comparison.comparison.tolist() == ['cluster_vs_random']
        assert (out/'group1_cluster.png').stat().st_size > 1000
        with self.assertRaisesRegex(ValueError, 'not empty'):
            rp.run(files[:1], randoms, channels=['IF_IRF'], out_dir=out)

    def test_two_condition_cli(self):
        tmp_path, files = self.root, self.files
        out = tmp_path/'two'
        # Deliberately unrelated random filenames: no auto-discovery is involved.
        randoms = [tmp_path/'background_A.mat', tmp_path/'other_control.mat']
        (tmp_path/'Stimulated_random.mat').rename(randoms[0])
        (tmp_path/'Not_stimulated_random.mat').rename(randoms[1])
        command = [sys.executable, str(Path(rp.__file__)), '--cluster-files', *map(str, files),
                   '--random-files', *map(str, randoms), '--channels',
                   'IF_IRF', 'IF_IRF', '--labels', 'stim', 'unstim', '--out', str(out)]
        proc = subprocess.run(command, capture_output=True, text=True,
                              env={**os.environ, 'MPLBACKEND': 'Agg'})
        assert proc.returncode == 0, proc.stderr
        summary = rp.pd.read_csv(out/'summary.csv')
        assert summary.n_clusters.tolist() == [136, 32]
        assert summary.channel.tolist() == ['IF_IRF', 'IF_IRF']
        assert summary.n_random.tolist() == [190, 274]
        assert np.allclose(summary.empirical_p_floor, [1/191, 1/275])
        assert 'FISH_IRF' not in (out/'summary.csv').read_text()
        assert 'FISH_IRF' not in proc.stdout
        assert 'recruitment_change' in proc.stdout and 'p_conditional' in proc.stdout
        compared = rp.pd.read_csv(out/'comparisons.csv')
        assert set(compared.comparison) == {'cluster_vs_random', 'recruitment_change',
                                           'random_control_change'}
        poi = compared.query("comparison == 'recruitment_change' and channel_A == 'IF_IRF'").iloc[0]
        assert poi.difference > 0 and poi.p_conditional < .001
        # Independent reference calculation copied from the original notebook.
        # All four variances contribute, including the two random baselines.
        groups = [rp.enrichment(rp.load_profiles(p, ['IF_IRF'])['IF_IRF'])
                  for p in (files[0], randoms[0], files[1], randoms[1])]
        n = np.array([len(g) for g in groups])
        delta = np.dot([1, -1, -1, 1], [g.mean() for g in groups])
        variance = np.array([g.var(ddof=1)/len(g) for g in groups])
        df = variance.sum()**2 / np.sum(variance**2/(n-1))
        p_notebook = 2*t.sf(abs(delta / np.sqrt(variance.sum())), df)
        assert np.isclose(poi.difference, delta, atol=1e-14)
        assert np.isclose(poi.p_conditional, p_notebook, rtol=1e-12, atol=0)
        assert np.all(compared.q_BY >= compared.p_conditional)
        config = json.loads((out/'run.json').read_text())
        assert config['cluster_files'] == [str(f.resolve()) for f in files]
        assert config['random_files'] == [str(f.resolve()) for f in randoms]
        assert config['file_mapping']['group2']['condition'] == 'unstim'
        assert config['channels'] == ['IF_IRF', 'IF_IRF']
        assert config['version'] == rp.__version__
        assert config['source_file'] == str(Path(rp.__file__).resolve())
        assert config['python_executable'] == sys.executable
        assert not config['paired_clusters'] and not config['paired_random']
        assert len(list(out.glob('*.png'))) == 5
        assert len(rp.pd.read_csv(out/'group2_clusters.csv')) == 32

    def test_auxiliary_reference_does_not_add_summary_channels(self):
        randoms = [self.root/'Stimulated_random.mat', self.root/'Not_stimulated_random.mat']
        summary, compared = rp.run(self.files, randoms, channels=['IF_IRF', 'IF_IRF'],
                                   reference='FISH_IRF', out_dir=self.root/'auxiliary')
        assert summary.channel.tolist() == ['IF_IRF', 'IF_IRF']
        assert 'relative_reference_change' in compared.comparison.tolist()
        # Selecting only IF_IRF must not require FISH_IRF to exist in the file.
        def only_poi(path, channels):
            assert channels == ['IF_IRF']
            return original(path, channels)
        original = rp.load_profiles
        with patch.object(rp, 'load_profiles', side_effect=only_poi):
            summary, _ = rp.run(self.files, randoms, channels=['IF_IRF', 'IF_IRF'],
                                out_dir=self.root/'poi_only')
        assert summary.channel.tolist() == ['IF_IRF', 'IF_IRF']

    def test_same_condition_different_channels_are_paired(self):
        cluster = self.files[0]
        random = self.root/'Stimulated_random.mat'
        summary, compared = rp.run([cluster, cluster], [random, random],
                                   channels=['FISH_IRF', 'IF_IRF'], labels=['reference', 'POI'],
                                   out_dir=self.root/'paired')
        assert summary.channel.tolist() == ['FISH_IRF', 'IF_IRF']
        assert summary.n_clusters.tolist() == [136, 136]
        c, r = [rp.load_profiles(p, ['FISH_IRF', 'IF_IRF']) for p in (cluster, random)]
        dc = rp.enrichment(c['FISH_IRF']) - rp.enrichment(c['IF_IRF'])
        dr = rp.enrichment(r['FISH_IRF']) - rp.enrichment(r['IF_IRF'])
        result = compared.query("comparison == 'recruitment_change'").iloc[0]
        assert result.paired_clusters and result.paired_random
        assert np.isclose(result.difference, dc.mean()-dr.mean())
        assert np.isclose(result.p_conditional, ttest_ind(dc, dr, equal_var=False).pvalue)
        control = compared.query("comparison == 'random_control_change'").iloc[0]
        assert np.isclose(control.p_conditional, ttest_rel(rp.enrichment(r['FISH_IRF']), rp.enrichment(r['IF_IRF'])).pvalue)

    def test_channel_count_and_paired_count_validation(self):
        randoms = [self.root/'Stimulated_random.mat', self.root/'Not_stimulated_random.mat']
        with self.assertRaisesRegex(ValueError, 'one channel per'):
            rp.run(self.files, randoms, channels=['IF_IRF'], out_dir=self.root/'bad_channels')
        with self.assertRaisesRegex(ValueError, 'Paired groups'):
            rp.compare_conditions(np.ones((136,54)), np.ones((100,54)),
                                  np.ones((32,54)), np.ones((100,54)), paired=True)

    def test_shared_random_file_preserves_its_pairing(self):
        random = self.root/'Stimulated_random.mat'
        _, compared = rp.run(self.files, [random, random], channels=['IF_IRF', 'IF_IRF'],
                             out_dir=self.root/'shared_random')
        a, b = [rp.enrichment(rp.load_profiles(p, ['IF_IRF'])['IF_IRF']) for p in self.files]
        result = compared.query("comparison == 'recruitment_change'").iloc[0]
        assert not result.paired_clusters and result.paired_random
        assert np.isclose(result.p_conditional, ttest_ind(a, b, equal_var=False).pvalue)

    def test_explicit_file_counts_required(self):
        with self.assertRaisesRegex(ValueError, 'one random file'):
            rp.run(self.files, [self.root/'Stimulated_random.mat'], out_dir=self.root/'invalid')
        proc = subprocess.run([sys.executable, str(Path(rp.__file__)), '--cluster-files',
                               str(self.files[0])], capture_output=True, text=True)
        assert proc.returncode == 2 and '--random-files' in proc.stderr

    def test_missing_file_and_reference_mismatch(self):
        tmp_path, files = self.root, self.files
        with self.assertRaisesRegex(ValueError, 'File not found'):
            rp.run(files[:1], random_files=[tmp_path/'absent.mat'], out_dir=tmp_path/'missing')
        cluster = tmp_path/'mismatch.mat'
        matfile(cluster, np.ones((2, 54)), np.ones((3, 54)))
        matfile(tmp_path/'mismatch_random.mat', np.ones((4, 54)), np.ones((4, 54)))
        with self.assertRaisesRegex(ValueError, 'paired'):
            rp.run([cluster], [tmp_path/'mismatch_random.mat'], reference='FISH_IRF', out_dir=tmp_path/'mismatch_out')

if __name__ == "__main__":
    unittest.main()
