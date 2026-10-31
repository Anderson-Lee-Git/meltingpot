"""Recreate saved experiment figures in matplotlib and independent pgfplots."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.agents/skills/paper-figure/scaffold'))
from paper_style import apply_style, PALETTE
apply_style()
DATA = ROOT / 'examples/results/non_spatial_commons'
EXPERIMENTS = ['individual_maintenance', 'individual_msy', 'cooperative']
PPO_TITLES = ['Individual: Maintenance Partners', 'Individual: MSY Partners', 'Cooperative Joint Policy']
SPECS = {
    'fixed_default': ('default', ['greedy', 'abstain', 'mixed'], ['Greedy', 'Abstain', 'Mixed'], 0, 100, 'Fixed Policies: Default Start', 'Single Episode (Seed 0); First 100 Rounds'),
    'fixed_recovery': ('recovery', ['msy_share', 'reserve_target', 'mixed'], ['MSY Share', 'Reserve Target', 'Mixed'], 0, 100, 'Fixed Policies: Depleted Start', 'Single Episode (Seed 0); First 100 Rounds'),
    'ppo_default': ('ppo/showcase_default', EXPERIMENTS, PPO_TITLES, 0, 1000, 'Frozen PPO Policies: Default Start', 'Single Episode per Policy; Training Seed 0; Deterministic Actions'),
    'ppo_recovery': ('ppo/showcase_recovery', EXPERIMENTS, PPO_TITLES, 1, 1000, 'Frozen PPO Policies: Depleted Start', 'Single Episode per Policy; Training Seed 0; Deterministic Actions'),
}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_table(name, values):
    path = HERE / 'data' / (name + '.dat')
    path.parent.mkdir(exist_ok=True)
    np.savetxt(path, values, header='x y low high', comments='', fmt='%.12g')
    return 'data/' + path.name


def escape(value):
    return value.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def render(name):
    training = name == 'ppo_training'
    nrows = 3 if training else 4
    panels = [[[] for _ in range(3)] for _ in range(nrows)]
    sources = []
    if training:
        title = 'PPO Training Without Shielding'
        subtitle = 'Mean and Min–Max Range Across Three Training Seeds'
        footer = 'Recent 100 Completed Episodes; Return: Agent 0 (Individual), Total Harvest (Cooperative)'
        titles = PPO_TITLES
        labels = ['Episode Harvest Return', 'Episode Length', 'Failure Fraction']
        xmax, xlabel = 200.704, 'Environment Transitions (Thousands)'
        references = [None, 256, 0]
        for col, experiment in enumerate(EXPERIMENTS):
            paths = sorted((DATA / 'ppo' / experiment).glob('seed_*/training.csv'))
            assert len(paths) == 3
            sources.extend(paths)
            runs = [read_csv(path) for path in paths]
            assert all([row['steps'] for row in run] == [row['steps'] for row in runs[0]] for run in runs)
            x = np.array([float(row['steps']) / 1000 for row in runs[0]])
            for r, key in enumerate(['recent_mean_return', 'recent_mean_length', 'recent_failure_fraction']):
                values = np.array([[float(row[key]) if int(row['completed_episodes']) else np.nan for row in run] for run in runs])
                valid = np.any(np.isfinite(values), axis=0)
                values = values[:, valid]
                table = np.column_stack([x[valid], np.nanmean(values, axis=0), np.nanmin(values, axis=0), np.nanmax(values, axis=0)])
                panels[r][col].append((save_table(f'{name}_{r}_{col}', table), table, False, False))
    else:
        folder, populations, titles, episode, xmax, title, subtitle = SPECS[name]
        footer = 'Dotted Lines: MSY Stock / MSY Harvest and Zero Energy; × Marks Failure'
        xlabel = 'Round'
        labels = ['Resource Stock', 'Minimum Agent Energy', 'Aggregate Harvest', 'Cumulative Return']
        summary_path, trace_path = DATA / folder / 'summary.json', DATA / folder / 'trajectories.csv'
        sources.extend([summary_path, trace_path])
        summary = json.loads(summary_path.read_text())
        traces = read_csv(trace_path)
        references = [summary['msy_stock'], 0, summary['msy'], None]
        for col, pop in enumerate(populations):
            for shield in [True, False]:
                records = [row for row in traces if row['population'] == pop and int(row['episode']) == episode and (row['shield'] == 'True') == shield and int(row['round']) <= xmax]
                initial = next(row for row in summary['episodes'] if row['population'] == pop and row['episode'] == episode and row['shield'] == shield)
                for r, key in enumerate(['resource_after', 'minimum_energy', 'harvest', 'harvest']):
                    x = [int(row['round']) for row in records]
                    y = [float(row[key]) for row in records]
                    if r < 2:
                        x.insert(0, 0)
                        y.insert(0, initial['initial_resource'] if r == 0 else min(initial['initial_energies']))
                    if r == 3:
                        y = [0.0, *np.cumsum(y).tolist()]
                        x.insert(0, 0)
                    failed = bool(records and records[-1]['failed'] == 'True')
                    failure_point = (x[-1], y[-1]) if failed else None
                    if r == 3 and failed and x[-1] < xmax:
                        # No reward accrues after termination: retain final return.
                        x.append(xmax)
                        y.append(y[-1])
                    table = np.column_stack([x, y, y, y])
                    panels[r][col].append((save_table(f'{name}_{r}_{col}_{int(shield)}', table), table, not shield, failure_point))
    fig, axes = plt.subplots(nrows, 3, figsize=(12.8, 11.7 if nrows == 4 else 9.2))
    fig.subplots_adjust(left=.09, right=.99, bottom=.09, top=.87, hspace=.26, wspace=.23)
    fig.suptitle(title, y=.985, fontsize=18, weight='bold')
    fig.text(.54, .943, subtitle, ha='center', fontsize=13)
    fig.text(.54, .022, footer, ha='center', fontsize=11)
    if training:
        handles = [Line2D([], [], color=PALETTE[0], label='Seed Mean'), Line2D([], [], color=PALETTE[0], alpha=.2, linewidth=9, label='Seed Min–Max')]
    else:
        handles = [Line2D([], [], color=PALETTE[1], linestyle='--', label='Unshielded'), Line2D([], [], color=PALETTE[0], label='Shielded')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54,.918), ncol=2)
    tex = [r'\begin{tikzpicture}', r'\begin{groupplot}[group style={group size=3 by ' + str(nrows) + r',horizontal sep=1.35cm,vertical sep=0.65cm},width=6.5cm,height=4.25cm,scale only axis,axis x line*=bottom,axis y line*=left,axis line style={gray!60},grid=major,grid style={gray!18},tick align=outside,tick label style={font=\normalsize},label style={font=\large},title style={font=\large},scaled ticks=false,/pgf/number format/1000 sep={},unbounded coords=jump]']
    for r in range(nrows):
        # Common scales permit comparison of trajectories; individual and team returns differ in definition.
        all_y = np.concatenate([item[1][:, 2:4].ravel() for col in range(3) for item in panels[r][col]])
        for col in range(3):
            ax = axes[r,col]
            vals = np.concatenate([item[1][:,2:4].ravel() for item in panels[r][col]]) if training and r == 0 else all_y
            lo, hi = float(min(vals.min(), references[r] if references[r] is not None else vals.min())), float(max(vals.max(), references[r] if references[r] is not None else vals.max()))
            pad = max(.06 * (hi - lo), .04 if r == 2 and training else .5)
            ymin, ymax = lo - pad, hi + pad
            if training and r == 2: ymin, ymax = -.04, 1.04
            opts = [f'xmin=0,xmax={xmax},ymin={ymin},ymax={ymax}']
            if r == 0: opts.append('title={' + escape(titles[col]) + '}')
            if r == nrows - 1: opts.append('xlabel={' + xlabel + '}')
            else: opts.append('xticklabels={}')
            if col == 0: opts.append('ylabel={' + labels[r] + '}')
            tex.append(r'\nextgroupplot[' + ','.join(opts) + ']')
            if references[r] is not None:
                ax.axhline(references[r], color='.45', linestyle=':', linewidth=1.3, zorder=1)
                tex.append(r'\addplot[gray,densely dotted,thick] coordinates {(0,' + str(references[r]) + ') (' + str(xmax) + ',' + str(references[r]) + ')};')
            for index, (file, table, raw, failed) in enumerate(panels[r][col]):
                color = PALETTE[1] if raw else PALETTE[0]
                if training:
                    ax.fill_between(table[:,0],table[:,2],table[:,3],color=color,alpha=.18,linewidth=0)
                    tex.extend([r'\addplot[name path=lower,draw=none] table[x=x,y=low] {'+file+'};', r'\addplot[name path=upper,draw=none] table[x=x,y=high] {'+file+'};', r'\addplot[c0,fill opacity=0.18,draw=none] fill between[of=lower and upper];'])
                ax.plot(table[:,0], table[:,1], color=color, linestyle='--' if raw else '-', linewidth=1.8, zorder=3 if raw else 2)
                tex.append(r'\addplot[' + ('c1,dashed' if raw else 'c0') + r',line width=1pt,no marks] table[x=x,y=y] {'+file+'};')
                if failed:
                    ax.plot(failed[0],failed[1],marker='x',color=color,markersize=7,markeredgewidth=1.5)
                    tex.append(r'\addplot[c1,only marks,mark=x,mark size=3pt,thick] coordinates {('+str(failed[0])+','+str(failed[1])+')};')
            ax.set(xlim=(0,xmax), ylim=(ymin,ymax))
            ax.grid(True)
            ax.tick_params(labelsize=11)
            if r == 0: ax.set_title(titles[col], fontsize=13)
            if r == nrows - 1: ax.set_xlabel(xlabel, fontsize=12)
            else: ax.tick_params(labelbottom=False)
            if col == 0: ax.set_ylabel(labels[r], fontsize=13)
    tex.append(r'\end{groupplot}')
    tex.append(r'\node[font=\LARGE\bfseries,anchor=south] at ($(group c2r1.north)+(0,2.5cm)$) {'+escape(title)+'};')
    tex.append(r'\node[font=\large,anchor=south] at ($(group c2r1.north)+(0,1.9cm)$) {'+escape(subtitle.replace('–','--'))+'};')
    legend = [('c0','solid','Seed Mean'),('c0','solid','Seed Min--Max')] if training else [('c1','dashed','Unshielded'),('c0','solid','Shielded')]
    for i,(color,style,label) in enumerate(legend):
        x = -3.2 + i*4.1
        tex.append(r'\draw['+color+','+style+',line width='+('5pt,opacity=.2' if training and i else '1pt')+r'] ($(group c2r1.north)+('+str(x)+r'cm,1.4cm)$) -- ++(.65cm,0) node[right,black,opacity=1,font=\large] {'+label+'};')
    tex.append(r'\node[font=\normalsize,anchor=north] at ($(group c2r' + str(nrows) + r'.south)+(0,-1.15cm)$) {'+escape(footer.replace('×',r'$\times$'))+'};')
    tex.append(r'\end{tikzpicture}')
    (HERE/(name+'.tex')).write_text('\n'.join(tex)+'\n')
    fig.savefig(HERE/(name+'.png'), dpi=220)
    plt.close(fig)
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (HERE/(name+'.provenance.json')).write_text(json.dumps({'source_sha256':provenance,'title':title,'subtitle':subtitle,'notes':footer},indent=2)+'\n')
    print(f'Rendered {name}')

if __name__ == '__main__':
    for name in sys.argv[1:] or [*SPECS,'ppo_training']:
        render(name)
