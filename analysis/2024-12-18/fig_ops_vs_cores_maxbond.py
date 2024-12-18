import numpy as np
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

files = ['tbdb.h5']

tbdb_values = {
    'mode': ['cutensor','tblis', 'eigen1','cyclops', 'xtensor'],
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
    'cplx': 'complex128'
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
    # ax.legend()
    # exit(0)

for f in files:
    with h5py.File(f, 'r') as h5f:
        figrows = 1
        figcols = 1
        fig, ax = plt.subplots(nrows=figrows, ncols=figcols, figsize=(7 * figcols, 7 * figrows))
        fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks(tbdb_values['nomp'])
        ax.set_yticks([ 2**j for j in range(-6,6) ])
        ax.set_title('Bond dimension 1024')
        ax.set_ylabel('op/s')
        ax.set_xlabel('cores')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        lstyles = ['solid', 'dashed', 'dotted']
        palette = sns.color_palette(palette='Dark2', n_colors=len(tbdb_values['mode']))
        legend_mode = {'handle': [], 'labels':[]}
        legend_type = {'handle': [], 'labels':[]}

        for mode, color in zip(tbdb_values['mode'], palette):
            gpus = []
            for type, lstyle in zip(tbdb_values['type'], lstyles):
                if mode == 'cutensor':
                    line = tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'], values=[mode, type, 1024, 0], xkey='nomp', ykey='t_contr', yinv=True, hline=True, color=color,linestyle=lstyle, marker='o')
                    gpus.append('TITAN V')
                    # legend1['handle'].append(Line2D([0], [0], color=color, label=f'{mode}-TITAN V'))
                    # legend1['labels'].append(f'{mode} TITAN V')
                    line = tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'], values=[mode, type, 1024, 1], xkey='nomp', ykey='t_contr', yinv=True, hline=True, color=color, linestyle=lstyle, marker='v')
                    gpus.append('RTX 2080 TI')
                    for gpu,mrk in zip(gpus, ['o','v']):
                        modegpu = f'{mode} {gpu}'
                        if not modegpu in legend_mode['labels']:
                            legend_mode['handle'].append(Line2D([0], [0], color=color, marker=mrk, label=modegpu))
                            legend_mode['labels'].append(modegpu)

                else:
                    if mode == 'cyclops':
                        line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'], values=[mode, type, 1024],
                                  xkey='nmpi', ykey='t_contr', yinv=True, color=color,linestyle=lstyle)
                    else:
                        line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'], values=[mode, type, 1024],
                                  xkey='nomp', ykey='t_contr', yinv=True, color=color,linestyle=lstyle)
                    if not f'{mode}' in legend_mode['labels']:
                        legend_mode['handle'].append(Line2D([0], [0], color=color, label=mode))
                        legend_mode['labels'].append(mode)
                if not type in legend_type['labels']:
                    legend_type['handle'].append(Line2D([0], [0], color='black', linestyle=lstyle, label=type))
                    legend_type['labels'].append(type)

        legend_mode = plt.legend(handles=legend_mode['handle'], loc='upper left')
        legend_type = plt.legend(handles=legend_type['handle'], loc='lower right')

        ax.add_artist(legend_mode)
        ax.add_artist(legend_type)
        # ax.add_artist(legend3)
plt.show()
