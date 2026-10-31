"""Prepare matched saved frozen-policy evaluation traces, grouped by seed."""
import csv
import hashlib
import json
from pathlib import Path
import render_common

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'examples/results/non_spatial_commons/ppo_cooperative_h1000'
NAMES = [f'{selection}_{scenario}' for selection in ['deterministic', 'stochastic'] for scenario in ['default', 'recovery']]


def render(name):
    if name not in NAMES:
        raise ValueError(name)
    selection, scenario = name.split('_')
    manifest = json.loads((RUN / 'training_manifest.json').read_text())
    assert manifest['training']['episode_length'] == 1000
    assert manifest['training_shield'] is False
    assert manifest['seeds'] == [0, 1, 2]
    combined = json.loads((RUN / 'evaluation_summary.json').read_text())
    assert combined['evaluation_weights_frozen'] and combined['horizon'] == 1000
    assert len(combined['episodes']) == 144
    for rel, expected in json.loads((RUN / 'checkpoint_sha256_before_evaluation.json').read_text()).items():
        assert hashlib.sha256((RUN / rel).read_bytes()).hexdigest() == expected
    folder = RUN / ('showcase_' + name)
    folder.mkdir(exist_ok=True)
    summaries, records, source_hashes = [], [], {}
    for seed in range(3):
        directory = RUN / 'cooperative' / f'seed_{seed}' / 'evaluation'
        summary_path, trace_path = directory / 'summary.json', directory / 'trajectories.csv'
        data = json.loads(summary_path.read_text())
        for path in [summary_path, trace_path]:
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in data['episodes']:
            if item['selection'] == selection and item['scenario'] == scenario:
                summaries.append(dict(item, population=f'seed_{seed}'))
        with trace_path.open() as stream:
            for item in csv.DictReader(stream):
                if item['population'] == name:
                    records.append(dict(item, population=f'seed_{seed}'))
    assert len(summaries) == 6
    with (folder / 'trajectories.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {key: data[key] for key in ['config', 'critical_escapement', 'protected_escapement', 'msy', 'msy_stock']}
    payload['episodes'] = summaries
    (folder / 'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    render_common.HERE = HERE
    render_common.SPECS[name] = (
        str(folder.relative_to(render_common.DATA)),
        [f'seed_{i}' for i in range(3)],
        [f'Training Seed {i}' for i in range(3)],
        0 if scenario == 'default' else 1, 1000,
        f'Cooperative PPO: {"Default" if scenario == "default" else "Depleted"} Start',
        f'Trained With 1,000-Round Episodes; {selection.title()} Actions; One Episode per Seed',
    )
    render_common.render(name)
    provenance = HERE / (name + '.provenance.json')
    metadata = json.loads(provenance.read_text())
    metadata['original_evaluation_sha256'] = source_hashes
    metadata['cumulative_return_definition'] = 'Undiscounted sum of executed harvest across all four agents; zero future reward after termination.'
    metadata['training_episode_length'] = 1000
    metadata['training_transitions_per_seed'] = 200704
    provenance.write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    for name in NAMES:
        render(name)
