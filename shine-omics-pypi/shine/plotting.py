"""Compact SHINE notebook panels, using measured values and original partitions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

COLORS = ['#6AAB98', '#D58B76', '#AE99B9', '#82A6C0', '#B5CFA2',
          '#AA8179', '#9EB6AC', '#DAC879', '#C8D0DC', '#8C7AA5',
          '#A9AFB0', '#A3C7CF', '#C18AA9', '#C3AF98', '#C87551']
REGION_COLORS = dict(ACB='#82A6C0', CP_intact='#6AAB98', CP_lesioned='#D58B76',
                     CTX='#AE99B9', LSX='#B5CFA2', MSC='#AA8179', OT='#9EB6AC',
                     VL='#A9AFB0', aco='#DAC879', cc='#C8D0DC',
                     SNr_intact='#21877B', SNr_lesioned='#C45F41',
                     MB='#A796BA', HPF='#83A7BD', TH='#DBC06F', Unlabelled='#E3E6E7')
REGION_COLORS.update({'VTA-SNc_intact': '#62B09D', 'VTA-SNc_lesioned': '#E8A180'})


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def _map(ax, xy, colors, title):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=9, linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    ax.margins(.02); ax.set_title(title, pad=5)


def partition_colors(section, labels):
    mask, reference = section.endpoints()['Whole section']
    names = sorted(np.unique(reference[mask]))
    color = {n: REGION_COLORS.get(n, COLORS[i % len(COLORS)]) for i, n in enumerate(names)}
    clusters = np.unique(labels)
    counts = np.array([[np.sum(mask & (labels == c) & (reference == n))
                       for n in names] for c in clusters])
    rows, cols = linear_sum_assignment(-counts)
    matched = {clusters[i]: color[names[j]] for i, j in zip(rows, cols)}
    prediction = [matched.get(c, '#747E87') for c in labels]
    annotation = [color.get(n, '#E3E6E7') if valid else '#E3E6E7'
                  for n, valid in zip(reference, mask)]
    return annotation, prediction, color


def spatial_results(section, labels, umap=None):
    """Reference and SHINE, with optional visualization-only UMAP.

    Hungarian matching changes colours only; metrics use original cluster IDs.
    Every paired point is plotted; no unmeasured points are filled or imputed.
    """
    style()
    ann, pred, colors = partition_colors(section, labels)
    fig, axes = plt.subplots(1, 3 if umap is not None else 2,
                             figsize=(10.5 if umap is not None else 7.2, 4.1))
    _map(axes[0], section.xy, ann, 'Reference annotation')
    _map(axes[1], section.xy, pred, 'SHINE · seed 123 · K = ' + str(section.config['K']))
    if umap is not None:
        ax = axes[2]
        ax.scatter(umap[:, 0], umap[:, 1], c=pred, s=5, lw=0, rasterized=True)
        low, high = np.min(umap, axis=0), np.max(umap, axis=0)
        span = np.maximum(high-low, 1e-6)
        # Reserve empty coordinate space for the shared-origin axis arrows;
        # never paint over or remove embedded points to make room.
        ax.set_xlim(low[0]-.24*span[0], high[0]+.04*span[0])
        ax.set_ylim(low[1]-.25*span[1], high[1]+.04*span[1])
        ax.axis('off'); ax.set_title('SHINE UMAP')
        for end in [(.25, .06), (.06, .25)]:
            ax.annotate('', xy=end, xytext=(.06, .06), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color='#344148', lw=1,
                                        shrinkA=0, shrinkB=0))
        ax.text(.27, .04, 'UMAP1', transform=ax.transAxes, fontsize=8)
        ax.text(.015, .28, 'UMAP2', transform=ax.transAxes, fontsize=8, rotation=90)
    handles = [Line2D([], [], marker='o', color=c, ls='none', markersize=4, label=n.replace('_', ' ')) for n, c in colors.items()]
    for name, color, present in [('Unlabelled', '#E3E6E7', '#E3E6E7' in ann),
                                 ('Unmatched cluster', '#747E87', '#747E87' in pred)]:
        if present and name not in colors:
            handles.append(Line2D([], [], marker='o', color=color, ls='none', markersize=4, label=name))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8)
    fig.subplots_adjust(left=.01, right=.99, top=.89, bottom=.23, wspace=.14)
    return fig


def regional_results(section, labels):
    style()
    regions = list(section.endpoints())[1:]
    if not regions:
        raise ValueError('No regional endpoints for this section.')
    _, pred, _ = partition_colors(section, labels)
    pred = np.asarray(pred)
    fig, axes = plt.subplots(1, len(regions), squeeze=False, figsize=(4*len(regions), 3.5))
    for ax, region in zip(axes[0], regions):
        mask, side = section.endpoints()[region]
        colors = np.full(len(labels), '#E6E8EA', dtype=object); colors[mask] = pred[mask]
        _map(ax, section.xy, colors, region + ' · SHINE global clusters')
        for name, c in [('intact', '#268577'), ('lesioned', '#C66E54')]:
            sel = mask & (side == name)
            if sel.any():
                center = section.xy[sel].mean(0)
                ax.annotate(name, xy=center, xytext=(0, -25 if name=='intact' else 25),
                            textcoords='offset points', ha='center', color=c, fontsize=9,
                            bbox=dict(fc='white', ec='none', alpha=.8, pad=1),
                            arrowprops=dict(arrowstyle='-', color=c, lw=.8))
    fig.tight_layout()
    return fig


def molecular_maps(section, values):
    style()
    genes, peaks = section.config['display_genes'], section.config['display_peaks']
    ncols = 6 if max(len(genes), len(peaks)) <= 6 else 4
    rows_gene, rows_peak = int(np.ceil(len(genes)/ncols)), int(np.ceil(len(peaks)/ncols))
    fig, axes = plt.subplots(rows_gene+rows_peak, ncols, figsize=(12, 2.25*(rows_gene+rows_peak)), squeeze=False)
    cmap = LinearSegmentedColormap.from_list('measured_intensity', ['#ADD2AC', '#FFFFFF', '#92008C'])
    for rowgroup, features, start in [('RNA', genes, 0), ('MSI m/z', peaks, rows_gene)]:
        for j, feature in enumerate(features):
            ax = axes[start+j//ncols, j%ncols]
            x = values[feature].to_numpy()
            lo, hi = np.percentile(x, [2, 98])
            if hi <= lo:
                lo, hi = float(x.min()), float(x.max())
            if hi <= lo:
                hi = lo+1
            title = feature if rowgroup == 'RNA' else '%.5f' % float(feature.replace('mz_', ''))
            _map(ax, section.xy, cmap(np.clip((x-lo)/(hi-lo), 0, 1)), title)
            if rowgroup == 'RNA': ax.title.set_fontstyle('italic')
        axes[start, 0].text(-.12, .5, rowgroup, transform=axes[start, 0].transAxes,
                            rotation=90, va='center', fontsize=10, weight='bold')
    for ax in axes.flat:
        if not ax.collections: ax.axis('off')
    fig.subplots_adjust(left=.04, right=.99, bottom=.12, top=.93, hspace=.22, wspace=.12)
    cax = fig.add_axes([.73, .045, .22, .025])
    import matplotlib as mpl
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High']); cb.outline.set_visible(False)
    fig.text(.04, .045, 'Measured RNA / paired MSI; P2–P98 per feature (min–max if tied).', fontsize=9)
    return fig


def attention_plot(model):
    style(); fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key in zip(axes, ['rna', 'msi']):
        ax.hist(model.attention_[key+'_raw'][:, 0], bins=np.linspace(0, 1, 31),
                histtype='step', color='#8D969C', label='Learned weight')
        ax.hist(model.attention_[key][:, 0], bins=np.linspace(0, 1, 31),
                histtype='step', color='#498D91', label='Effective weight')
        ax.set(xlim=(0, 1), title=key.upper(), xlabel='Within-modality spatial weight', ylabel='Spots')
    axes[1].legend(fontsize=8); fig.tight_layout(); return fig


def save_panel(fig, stem):
    """Save editable vectors and a notebook-scale preview (not a submission plate)."""
    from pathlib import Path
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + '.svg', bbox_inches='tight')
    fig.savefig(str(stem) + '.pdf', bbox_inches='tight')
    fig.savefig(str(stem) + '.png', dpi=300, bbox_inches='tight')
