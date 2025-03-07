import numpy as np
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

files = [
    'tbdb.h5',
    'tbdb-landau-full.h5',
    'tbdb-oppenheimer-full.h5',
    'tbdb-hertz-full.h5',
]

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

tbdb_typedict = {
    'fp32': 'FP32',
    'fp64': 'FP64',
    'cplx': 'CX64'
}

# tbdb_libname = {
#     'eigen1': 'Eigen 3.4.0',
#     'tblis': 'TBLIS 1.2.0',
#     'xtensor': 'xtensor 0.24.6',
#     'cyclops': 'Cyclops 1.5.5',
#     'cutensor': 'cuTENSOR 1.7.0'
# }
palette = [x for x in sns.color_palette(palette='Dark2', n_colors=5)]
print(palette)
tbdb_libdict = {
    'eigen1':   {'color': palette[0],'name': 'eigen', 'tag': 'eigen',},
    'tblis':    {'color': palette[1],'name': 'tblis', 'tag': 'tblis',},
    'xtensor':  {'color': palette[2],'name': 'xtensor', 'tag': 'xtensor',},
    'cyclops':  {'color': palette[3],'name': 'cyclops', 'tag': 'cyclops',},
    'cutensor': {'color': palette[4],'name': 'cutensor', 'tag': 'cutensor',},
}

tbdb_gpudict = {
    'RTX2080': {'tag': 'RTX2080TI',  'marker' : 'o', 'color': palette[4],'name': 'NVIDIA GeForce RTX 2080 Ti',},
    'RTX3090': {'tag': 'RTX3090',  'marker' : 'x', 'color': palette[4],'name': 'NVIDIA GeForce RTX 3090'   ,},
    'RTX4090': {'tag': 'RTX4090',  'marker' : '^', 'color': palette[4],'name': 'NVIDIA GeForce RTX 4090'   ,},
    'TITANV' : {'tag': 'TITAN V',  'marker' : 'v', 'color': palette[4],'name': 'NVIDIA TITAN V'            ,},
}

tbdb_cpudict = {
    'RZ5950-16c': {'tag' : 'RZ5950-16c' , 'color': palette[0], 'linestyle': 'dashdot', 'name' : 'AMD Ryzen 9 5950X 16-Core Processor'            ,},
    'RZ7950-16c': {'tag' : 'RZ7950-16c' , 'color': palette[1], 'linestyle': 'dotted', 'name' : 'AMD Ryzen 9 7950X 16-Core Processor'            ,},
    'TR2990-32c': {'tag' : 'TR2990-32c' , 'color': palette[2], 'linestyle': 'dashed', 'name' : 'AMD Ryzen Threadripper 2990WX 32-Core Processor',},
    'TR3990-64c': {'tag' : 'TR3990-64c' , 'color': palette[3], 'linestyle': 'solid', 'name' : 'AMD Ryzen Threadripper 3990X 64-Core Processor' ,},
}



def get_rows(table, fields, values):
    mask = np.ma.make_mask(table[()])
    for field, value in zip(fields, values):
        if isinstance(value,list):
            mask = mask & np.isin(table.fields([field])[()].astype(dtype=tbdb_dtypes[field]), value)
        else:
            mask = mask & (table.fields([field])[()].astype(dtype=tbdb_dtypes[field]) == value)
    return table[mask].astype(dtype=[dt for dt in tbdb_dtypes.items()])


def get_rows_stats(table,tbdb_vals, fields,values, xkey, ykey, yinv=False):
    yavg = []
    ystd = []
    yste = []
    xval = []
    for val in tbdb_vals[xkey]:
        rows = get_rows(table, fields + [xkey], values + [val])
        xval.append(val)
        if yinv:
            data = 1.0/rows[ykey]
        else:
            data = rows[ykey]
        yavg.append(np.nanmean(data))
        ystd.append(np.nanstd(data))
        yste.append(ystd[-1]/np.sqrt(len(rows)))
    return np.asarray(xval), np.asarray(yavg), np.asarray(ystd), np.asarray(yste)

def tbdb_plot(ax, table, tbdb_vals, fields, values, xkey, ykey, yinv=False,hline=False, **kwargs):
    xval, yavg, ystd, yste = get_rows_stats(table=table, tbdb_vals=tbdb_vals,fields=fields,values=values, xkey=xkey, ykey=ykey, yinv=yinv)
    if hline:
        yavg = np.asarray([yavg[0]]*len(yavg))
        yste = np.asarray([yste[0]]*len(yste))
    return ax.errorbar(x=xval, y=yavg, yerr=yste, **kwargs)
    # ax.legend()
    # exit(0)


def create_plot(ax, tbdb_vals, leglibs=None, legtype=None, legcpus=None, leggpus=None):
    legend_libs = {'handle': [], 'labels': []}
    legend_type = {'handle': [], 'labels': []}
    legend_cpus = {'handle': [], 'labels': []}
    legend_gpus = {'handle': [], 'labels': []}
    for f in files:
        with h5py.File(f, 'r') as h5f:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.set_xticks(tbdb_vals['nomp'])
            ax.set_yticks([ 2**j for j in range(-6,6) ])
            # ax.set_title('Bond dimension 1024')
            ax.set_ylabel('op/s')
            ax.set_xlabel('cores')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            device_list = set(h5f['tbdb']['device'][()])
            device_list = [ x.decode('utf-8') for x in device_list ]
            print(f'{device_list=}')
            for lib in tbdb_vals['mode']:
                for type in tbdb_vals['type']:
                    for gpu in tbdb_vals['gpus']:
                        if lib == 'cutensor' and tbdb_gpudict[gpu]['name'] in device_list:
                            print(f'plotting {lib} {type} {gpu} {f}')
                            line = tbdb_plot(ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals,
                                             fields=['mode', 'type', 'chiL', 'device'],
                                             values=[lib, type, 1024, tbdb_gpudict[gpu]['name']],
                                             xkey='nomp', ykey='t_contr', yinv=True, hline=True, color=tbdb_gpudict[gpu]['color'],
                                             linestyle='solid', marker=tbdb_gpudict[gpu]['marker'])
                            if not gpu in legend_gpus['labels']:
                                legend_gpus['handle'].append(Line2D([0], [0], color=tbdb_gpudict[gpu]['color'], linestyle='solid', label=tbdb_gpudict[gpu]['tag'], marker=tbdb_gpudict[gpu]['marker']))
                                legend_gpus['labels'].append(tbdb_gpudict[gpu]['tag'])
                            if not tbdb_libdict[lib]['tag'] in legend_libs['labels']:
                                legend_libs['handle'].append(Line2D([0], [0], color=tbdb_libdict[lib]['color'], label=tbdb_libdict[lib]['tag']))
                                legend_libs['labels'].append(tbdb_libdict[lib]['tag'])
                    for cpu in tbdb_vals['cpus']:
                        print(f'plotting {lib} {type} {cpu} {f}')
                        if lib != 'cutensor' and tbdb_cpudict[cpu]['name'] in device_list:
                            if lib == 'cyclops':
                                line = tbdb_plot(ax=ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals,  fields=['mode', 'type', 'chiL', 'device'], values=[lib, type, 1024, tbdb_cpudict[cpu]['name']],
                                          xkey='nmpi', ykey='t_contr', yinv=True, color=tbdb_libdict[lib]['color'],linestyle=tbdb_cpudict[cpu]['linestyle'])
                            else:
                                line = tbdb_plot(ax=ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals,  fields=['mode', 'type', 'chiL', 'device'], values=[lib, type, 1024, tbdb_cpudict[cpu]['name']],
                                          xkey='nomp', ykey='t_contr', yinv=True, color=tbdb_libdict[lib]['color'],linestyle=tbdb_cpudict[cpu]['linestyle'])
                            if not tbdb_libdict[lib]['tag'] in legend_libs['labels']:
                                legend_libs['handle'].append(Line2D([0], [0], color=tbdb_libdict[lib]['color'], label=tbdb_libdict[lib]['tag']))
                                legend_libs['labels'].append(tbdb_libdict[lib]['tag'])
                            if not tbdb_typedict[type] in legend_type['labels']:
                                legend_type['handle'].append(Line2D([0], [0], color='#E5E5E5', linestyle=None, label=tbdb_typedict[type]))
                                legend_type['labels'].append(tbdb_typedict[type])
                            if not tbdb_cpudict[cpu]['tag'] in legend_cpus['labels']:
                                legend_cpus['handle'].append(Line2D([0], [0], color='black', linestyle=tbdb_cpudict[cpu]['linestyle'], label=tbdb_cpudict[cpu]['tag']))
                                legend_cpus['labels'].append(tbdb_cpudict[cpu]['tag'])

    arg = np.argsort(legend_libs['labels'])
    legend_libs['labels'] = list(np.asarray(legend_libs['labels'])[arg])
    legend_libs['handle'] = list(np.asarray(legend_libs['handle'])[arg])

    arg = np.argsort(legend_cpus['labels'])
    legend_cpus['labels'] = list(np.asarray(legend_cpus['labels'])[arg])
    legend_cpus['handle'] = list(np.asarray(legend_cpus['handle'])[arg])

    arg = np.argsort(legend_gpus['labels'])
    legend_gpus['labels'] = list(np.asarray(legend_gpus['labels'])[arg])
    legend_gpus['handle'] = list(np.asarray(legend_gpus['handle'])[arg])

    if leglibs is True:
        legend_libs = ax.legend(handles=legend_libs['handle'], loc='upper left', fontsize=11, bbox_to_anchor=(1.1, 1.1))
        ax.add_artist(legend_libs)
    if legtype is True:
        legend_type = ax.legend(handles=legend_type['handle'], loc='lower right', handlelength=0, handleheight=0, handletextpad=0)
        ax.add_artist(legend_type)
    if legcpus is True:
        legend_cpus = ax.legend(handles=legend_cpus['handle'], loc='lower left', fontsize=10, bbox_to_anchor=(1.1, -0.17))
        ax.add_artist(legend_cpus)
    if leggpus is True:
        legend_gpus = ax.legend(handles=legend_gpus['handle'], loc='center left', fontsize=10, bbox_to_anchor=(1.1, 0.40))
        ax.add_artist(legend_gpus)



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "cm",
    "mathtext.fontset": "cm",
    'figure.constrained_layout.w_pad':  0.0,   # inches
    'figure.constrained_layout.h_pad':  0.0,   # inches
    'figure.subplot.top'   : 0.97,
    'figure.subplot.bottom': 0.06,
    'figure.subplot.left': 0.06,
    'figure.subplot.right' : 0.98,
    'figure.subplot.wspace' : 0.15,
    'figure.subplot.hspace' : 0.5,
    'savefig.bbox'           : 'standard',
    'savefig.pad_inches'     : 0.5,
    # 'pgf.texsystem'          : 'pdflatex',
})

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, axes = plt.subplot_mosaic('''
                                ABCL
                                ''',
                              figsize=(1024*px, 320*px),
                              layout="constrained",
                              sharex=True, sharey=True
                              )

# fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
# fig.subplots_adjust(wspace=0.2, hspace=0.2)



tbdb_fp32 = {
    'mode': ['cutensor','tblis', 'eigen1', 'xtensor', 'cyclops'],
    'type': ['fp32'],
    'cpus': ['RZ5950-16c', 'RZ7950-16c','TR2990-32c', 'TR3990-64c'],
    'gpus': ['RTX2080', 'RTX3090', 'RTX4090', 'TITANV'],
    'device': None,
    'nomp': [1, 2, 4, 8, 16, 32, 64],
    'nmpi': [1, 2, 4, 8, 16, 32, 64],
    'gpun': [0, 1],
    'spin': [2],
    'chiL': [1024],
    'chiR': [1024],
    'mpoD': [14],
    'itrn': None,
    'itrs': None,
    't_contr': None,
    't_total': None,
}

tbdb_fp64 = {
    'mode': ['cutensor','tblis', 'eigen1', 'xtensor', 'cyclops'],
    'type': ['fp64'],
    'cpus': ['RZ5950-16c', 'RZ7950-16c','TR2990-32c', 'TR3990-64c'],
    'gpus': ['RTX2080', 'RTX3090', 'RTX4090', 'TITANV'],
    'device': None,
    'nomp': [1, 2, 4, 8, 16, 32, 64],
    'nmpi': [1, 2, 4, 8, 16, 32, 64],
    'gpun': [0, 1],
    'spin': [2],
    'chiL': [1024],
    'chiR': [1024],
    'mpoD': [14],
    'itrn': None,
    'itrs': None,
    't_contr': None,
    't_total': None,
}
tbdb_cplx = {
    'mode': ['cutensor','tblis', 'eigen1', 'xtensor', 'cyclops'],
    'type': ['cplx'],
    'cpus': ['RZ5950-16c', 'RZ7950-16c','TR2990-32c', 'TR3990-64c'],
    'gpus': ['RTX2080', 'RTX3090', 'RTX4090', 'TITANV'],
    'device': None,
    'nomp': [1, 2, 4, 8, 16, 32, 64],
    'nmpi': [1, 2, 4, 8, 16, 32, 64],
    'gpun': [0, 1],
    'spin': [2],
    'chiL': [1024],
    'chiR': [1024],
    'mpoD': [14],
    'itrn': None,
    'itrs': None,
    't_contr': None,
    't_total': None,
}

axes['L'].set_axis_off()
axes['B'].yaxis.label.set_visible(False)
axes['C'].yaxis.label.set_visible(False)
axes['A'].set_box_aspect(aspect=1)
axes['B'].set_box_aspect(aspect=1)
axes['C'].set_box_aspect(aspect=1)
axes['L'].set_box_aspect(aspect=2)
fig.suptitle('Bond dimension $\chi = 1024$', y = 0.95)
create_plot(axes['A'], tbdb_fp32, leglibs=False, legtype=True, legcpus=False, leggpus=False)
create_plot(axes['B'], tbdb_fp64, leglibs=False, legtype=True, legcpus=False, leggpus=False)
create_plot(axes['C'], tbdb_cplx, leglibs=True, legtype=True, legcpus=True, leggpus=True)

plt.savefig('ops_vs_cores_libs:all_cpus:all_prec:all.pdf', format='pdf')
plt.show()
