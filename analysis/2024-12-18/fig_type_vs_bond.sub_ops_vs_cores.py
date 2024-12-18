import matplotlib.legend
import numpy as np
from seaborn import color_palette
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

plt.style.use('slack.mplstyle')

files = ['tbdb.h5']


tbdb_values = {
    'mode': ['cutensor','tblis',  'eigen1','cyclops', 'xtensor'],
    'type': ['fp32', 'fp64', 'cplx'],
    'device': None,
    'nomp': [1, 2, 4, 8, 16, 32],
    'nmpi': [1, 2, 4, 8, 16, 32],
    'gpun': [0, 1],
    'spin': [2],
    'chiL': [32,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024],
    'chiR': [32,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024],
    'mpoD': [14],
    'itrn': None,
    'itrs': None,
    't_contr': None,
    't_total': None,
}

tbdb_dtypes = {
    'mode': np.dtype('U16'),
    'type': np.dtype('U4'),
    'device': np.dtype('U128'),
    'nomp': np.dtype('i4'),
    'nmpi': np.dtype('i4'),
    'gpun': np.dtype('i4'),
    'spin': np.dtype('i8'),
    'chiL': np.dtype('i8'),
    'chiR': np.dtype('i8'),
    'mpoD': np.dtype('i8'),
    'itrn': np.dtype('u8'),
    'itrs': np.dtype('u8'),
    't_contr': np.dtype('f8'),
    't_total': np.dtype('f8'),
}

tbdb_typename = {
    'fp32': 'float32',
    'fp64': 'float64',
    'cplx': 'complex128',
}

tbdb_modename = {
    'eigen1': 'Eigen 3.4.0',
    'tblis': 'TBLIS 1.2.0',
    'xtensor': 'xtensor 0.24.6',
    'cyclops': 'Cyclops 1.5.5',
    'cutensor': 'cuTENSOR 1.7.0'
}



def get_rows(table, fields, values):
    mask = np.ma.make_mask(table[()])
    for field, value in zip(fields, values):
        if isinstance(value,list):
            mask = mask & np.isin(table.fields([field])[()].astype(dtype=tbdb_dtypes[field]), value)
        else:
            mask = mask & (table.fields([field])[()].astype(dtype=tbdb_dtypes[field]) == value)
    return table[mask].astype(dtype=[dt for dt in tbdb_dtypes.items()])


def get_rows_stats(table,fields,values, xkey, ykey, yinv=False):
    yavg = []
    ystd = []
    yste = []
    xval = []
    for val in tbdb_values[xkey]:
        rows = get_rows(table, fields + [xkey], values + [val])
        xval.append(val)
        if yinv:
            data = 1.0/rows[ykey]
        else:
            data = rows[ykey]
        if np.size(data) == 0:
            yavg.append(np.nan)
            ystd.append(np.nan)
            yste.append(np.nan)
        else:
            yavg.append(np.nanmean(data))
            ystd.append(np.nanstd(data))
            yste.append(ystd[-1]/np.sqrt(len(rows)))
    return xval, yavg, ystd, yste

def tbdb_plot(ax, table, fields, values, xkey, ykey, yinv=False,hline=False, **kwargs):
    xval, yavg, ystd, yste = get_rows_stats(table=table,fields=fields,values=values, xkey=xkey, ykey=ykey, yinv=yinv)
    if hline:
        yavg = np.asarray([yavg[0]]*len(yavg))
        yste = np.asarray([yste[0]]*len(yste))
    return ax.errorbar(x=xval, y=yavg, yerr=yste, **kwargs)


for f in files:
    with h5py.File(f, 'r') as h5f:
        figrows = 3
        figcols = 4
        fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(3 * figcols, 3 * figrows),sharex='all', sharey='row')
        fig.tight_layout(pad=5.0, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(wspace=0.01, hspace=0.01, top=0.85)
        fig.suptitle('Tensor contraction benchmark\n'
                     '$L_{mjo} H_{opli} \Psi_{lmn} R_{nkp} = E\Psi_{ijk}$\n'
                     'MPS (state) $\Psi$, MPO (model) $H$, environments $L,R$, energy $E$ (scalar)\n'
                     'Bond dimensions $j=k=m=n$. Other dims. $i=l=2$, $o=p=14$')
        formatter = ScalarFormatter()
        formatter.set_scientific(False)

        lstyles = ['solid', 'dashed', 'dotted']
        palette = color_palette(palette='Dark2', n_colors=len(tbdb_values['mode']))
        bonds = [128,256,512,1024]
        timekey = 't_contr'
        for bidx, bond in enumerate(bonds):
            legend_mode = {'handle': [], 'labels': []}
            legend_gpus = {'handle': [], 'labels': []}
            type_handle = [Line2D([], [], color='black')]
            for tidx, (ax, type) in enumerate(zip(np.ravel(axes[:,bidx]), tbdb_values['type'])):
                print(f'{bond=} {type=}')
                gpus = []
                lstyle = None
                if tidx == 0:
                    ax.set_title(f'Bond dimension {bond}')
                if bidx == 0:
                    ax.set_ylabel('op/s')
                if type == tbdb_values['type'][-1]:
                    ax.set_xlabel('cores')
                ax.set_xscale('log', base=2)
                ax.set_yscale('log', base=2)
                ax.set_xticks(tbdb_values['nomp'])
                ax.set_yticks([2 ** j for j in range(-6, 10)])
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                ax.add_artist(Legend(parent=ax, handles=type_handle, labels=[tbdb_typename[type]], loc='lower right'))

                for midx, (mode, color) in enumerate(zip(tbdb_values['mode'], palette)):
                    if mode == 'cutensor':
                        tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'],
                                         values=[mode, type, bond, 0], xkey='nomp', ykey=timekey, yinv=True,
                                         hline=True, color=color, linestyle=lstyle, marker='o')
                        tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'],
                                         values=[mode, type, bond, 1], xkey='nomp', ykey=timekey, yinv=True,
                                         hline=True, color=color, linestyle=lstyle, marker='v')
                        for gpu, mrk in zip(['TITAN V', 'RTX2080TI'], ['o', 'v']):
                            if not gpu in legend_gpus['labels']:
                                legend_gpus['handle'].append(Line2D([], [], linestyle = 'None', color=color, marker=mrk))
                                legend_gpus['labels'].append(gpu)

                    elif mode == 'cyclops':
                        tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'],
                                         values=[mode, type, bond],
                                         xkey='nmpi', ykey=timekey, yinv=True, color=color, linestyle=lstyle)
                    else:
                        tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'],
                                         values=[mode, type, bond],
                                         xkey='nomp', ykey=timekey, yinv=True, color=color, linestyle=lstyle)
                    if not tbdb_modename[mode] in legend_mode['labels']:
                        legend_mode['handle'].append(Line2D([], [], color=color))
                        legend_mode['labels'].append(tbdb_modename[mode])
                if bidx == 0 and tidx == 0:
                    ax.add_artist(Legend(parent=ax, handles=legend_mode['handle'], labels=legend_mode['labels'], loc='lower left'))
                    ax.add_artist(Legend(parent=ax, handles=legend_gpus['handle'], labels=legend_gpus['labels'], loc='center left'))

plt.show()
