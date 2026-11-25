"""Compact SHINE notebook panels, using measured values and original partitions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment

# Figure-specific palettes, in the same matching order as the manuscript.
STRIATUM = dict(ACB='#82A7C5', CP_intact='#6CAA99', CP_lesioned='#D68D78',
                CTX='#B09AB9', LSX='#B5D2A5', MSC='#A57B70', OT='#9DB6AD',
                VL='#A7AAA7', aco='#DCCC83', cc='#C7D1DC')
NIGRA = {'SNr_intact':'#297F7C', 'SNr_lesioned':'#C6634D',
         'VTA-SNc_intact':'#76AEC7', 'VTA-SNc_lesioned':'#DEAD54',
         'MB':'#A999BF','HPF':'#8CA9C7','Isocortex':'#C5AA91','TH':'#D9BE77',
         'CA1-CA2':'#95B79F','fibers':'#A6A7A8','DG':'#BED39D','PAG':'#AEC8D0',
         'CA3':'#C590A9','ulcer':'#916F6A','Unlabelled':'#DFE2E2'}
CANCER = {'Airway':'#9B8BAB','Blood vessel':'#B59E65','Lymphocytes':'#82AA8D',
          'Normal cells':'#9CBCD0','Tumor':'#D88473','Unlabelled':'#DFE2E2',
          'CAFs':'#789EBA','Cancer Basal/Her2 SC':'#DCA472','Cancer Cycling':'#7AA991',
          'Cancer LumA/B SC':'#CE7C7D','Immune':'#A28CB5','Normal epithelial':'#B38D7A',
          'Other (stroma region)':'#C596B7','Other (tumor region)':'#B8BC84',
          'Vascular':'#B4CBDC','Non-tumor':'#76AAA5'}

def palette(section):
    return {'striatum': (STRIATUM, '#8A79A5', '#E4E6E8'),
            'nigra': (NIGRA, '#707981', '#DFE2E2'),
            'cancer': (CANCER, '#747E87', '#DFE2E2')}[section.config['kind']]


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def display_cells(all_xy, measured_xy, pitch):
    """Figure 2 display raster: nearest measured spot, clipped to local tissue.

    Pixel indices refer to existing rows, never additional observations.
    This is the same step and support cutoff used in Figure 2a/c.
    """
    from scipy.spatial import cKDTree
    step = pitch / 10
    lower, upper = all_xy.min(0) - pitch, all_xy.max(0) + pitch
    xs = np.arange(lower[0], upper[0] + step, step)
    ys = np.arange(lower[1], upper[1] + step, step)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    tissue_distance = cKDTree(all_xy).query(points)[0]
    distance, idx = cKDTree(measured_xy).query(points)
    support = (tissue_distance <= .60 * pitch) & (distance <= 1.50 * pitch)
    grid = np.where(support, idx, -1).reshape(xx.shape)
    cols = np.rint((measured_xy[:, 0] - xs[0]) / step).astype(int)
    rows = np.rint((measured_xy[:, 1] - ys[0]) / step).astype(int)
    if not np.array_equal(grid[rows, cols], np.arange(len(measured_xy))):
        raise ValueError('Display cells do not preserve measured spot centres.')
    return grid, xs, ys, step


def spatial_geometry(section):
    """Construct geometry once; keep display context separate from model data."""
    if hasattr(section, '_notebook_geometry'):
        return section._notebook_geometry
    import pandas as pd
    from scipy.spatial import cKDTree
    xy = section.xy
    kind = section.config['kind']
    if kind in ('striatum', 'nigra'):
        section.verify('display_context')
        context = pd.read_csv(section.path('display_context')).set_index('spot', verify_integrity=True)
        positions = context.index.get_indexer(section.spots.spot)
        if (positions < 0).any() or not np.allclose(context.iloc[positions][['x', 'y']], xy, atol=1e-3):
            raise ValueError('Display context is not aligned to the measured cohort.')
    if kind == 'striatum':
        grid, xs, ys, step = display_cells(context[['x', 'y']].to_numpy(), xy, section.config['display']['pitch'])
        geo = dict(grid=grid, extent=(xs[0]-step/2, xs[-1]+step/2, ys[0]-step/2, ys[-1]+step/2),
                   low=xy.min(0), high=xy.max(0))
    elif kind == 'nigra':
        design = np.c_[context[['array_col', 'array_row']], np.ones(len(context))]
        affine = np.linalg.lstsq(design, context[['x', 'y']].to_numpy(), rcond=None)[0]
        centres = design @ affine
        if np.linalg.norm(centres-context[['x','y']].to_numpy(), axis=1).max() >= 2:
            raise ValueError('Native tissue lattice fit differs from Figure 3.')
        offsets = np.array([[1,1/3],[0,2/3],[-1,1/3],[-1,-1/3],[0,-2/3],[1,-1/3]]) @ affine[:2]
        geo = dict(vertices=centres[:, None, :] + offsets[None, :, :], context=context,
                   positions=positions, low=centres.min(0), high=centres.max(0))
    else:
        pitch = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
        angles = np.arange(6)*np.pi/3 + np.pi/6
        offsets = np.c_[np.cos(angles), np.sin(angles)] * pitch * .56
        geo = dict(vertices=xy[:, None, :] + offsets, low=xy.min(0)-pitch, high=xy.max(0)+pitch)
    section._notebook_geometry = geo
    return geo


def _map(ax, section, colors, title, annotation=False):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba_array
    geo = spatial_geometry(section)
    colors = to_rgba_array(colors)
    if len(colors) != len(section.xy):
        raise ValueError('One color per measured paired spot is required.')
    kind = section.config['kind']
    if kind == 'striatum':
        rgba = np.vstack([colors, [0, 0, 0, 0]])[geo['grid']]
        ax.imshow(rgba, origin='lower', interpolation='nearest', extent=geo['extent'])
    else:
        if kind == 'nigra':
            context = geo['context']
            face = to_rgba_array(context.annotation.map(NIGRA).fillna('#DFE2E2') if annotation
                                 else ['#B9C2C5'] * len(context))
            if annotation and not np.allclose(face[geo['positions']], colors):
                raise ValueError('Paired and full-tissue annotation colors disagree.')
            face[geo['positions']] = colors
        else:
            face = colors
        ax.add_collection(PolyCollection(geo['vertices'], facecolors=face, edgecolors='none',
                                         antialiaseds=False, rasterized=True))
    pad = (geo['high']-geo['low']) * (.025 if kind == 'striatum' else .023 if kind == 'nigra' else 0)
    low, high = geo['low']-pad, geo['high']+pad
    ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), aspect='equal')
    ax.axis('off'); ax.set_title(title, pad=5)


def partition_colors(section, labels):
    mask, reference = section.endpoints()['Whole section']
    pal, extra, grey = palette(section)
    names = list(pal) if section.config['kind'] == 'striatum' else sorted(np.unique(reference[mask]))
    clusters = np.unique(labels)
    counts = np.array([[np.sum(mask & (labels == c) & (reference == n))
                       for n in names] for c in clusters])
    rows, cols = linear_sum_assignment(-counts)
    matched = {clusters[i]: pal[names[j]] for i, j in zip(rows, cols)
               if section.config['kind'] != 'nigra' or counts[i, j] > 0}
    prediction = [matched.get(c, extra) for c in labels]
    annotation = [pal.get(n, grey) if valid else grey for n, valid in zip(reference, mask)]
    color = {n: pal[n] for n in pal if n in reference}
    return annotation, prediction, color


def spatial_results(section, labels, umap=None):
    """Reference and SHINE, with optional visualization-only UMAP.

    Hungarian matching changes colours only; metrics use original cluster IDs.
    Filled display cells follow the manuscript; metrics retain measured spots only.
    """
    style()
    ann, pred, colors = partition_colors(section, labels)
    fig, axes = plt.subplots(1, 3 if umap is not None else 2,
                             figsize=(10.5 if umap is not None else 7.2, 4.1))
    _map(axes[0], section, ann, 'Reference annotation', annotation=True)
    _map(axes[1], section, pred, 'SHINE · seed 123 · K = ' + str(section.config['K']))
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
    _, extra, grey = palette(section)
    for name, color, present in [('Unlabelled', grey, grey in ann),
                                 ('Unmatched cluster', extra, extra in pred),
                                 ('Unpaired tissue', '#B9C2C5', section.config['kind'] == 'nigra')]:
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
        _map(ax, section, colors, region + ' · SHINE global clusters')
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
    cmap = LinearSegmentedColormap.from_list('measured_intensity', ['#C0DDBD', '#F8F6F8', '#C88CC7', '#790783'])
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
            _map(ax, section, cmap(np.clip((x-lo)/(hi-lo), 0, 1)), title)
            if rowgroup == 'RNA': ax.title.set_fontstyle('italic')
        axes[start, 0].text(-.12, .5, rowgroup, transform=axes[start, 0].transAxes,
                            rotation=90, va='center', fontsize=10, weight='bold')
    for ax in axes.flat:
        if not ax.collections and not ax.images: ax.axis('off')
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
