import numpy as np
from src.io.h5ops import *
# from src.plotting.style import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

files = [
    # '2023-04-18/tbdb-2023-04-18-eigen-tblis-xtensor-cyclops-cutensor.h5',
    '2023-04-18/tbdb.h5'
]


def get_file_data(h5path, table_pattern=None):
    table_data = {}
    version_data = {}
    h5file = h5open(h5path, 'r')
    for name, attr in h5file.attrs.items():
        version_data[name] = attr

    for table in h5py_dataset_iterator(node=h5file, keypattern=table_pattern, dep=2):
        print(table)
        table_name = table[0]
        table_path = table[1]
        table_dset = table[2]

        time_itr = table_dset['t_iter']
        time_avg = np.nanmean(time_itr, axis=0)
        time_ste = np.nanstd(time_itr, axis=0) / np.shape(time_itr)[0]

        threads = table_dset['threads'][0]
        chiL = table_dset['chiL'][0]
        chiR = table_dset['chiR'][0]
        bond = np.min([chiL, chiR])
        if not table_path in table_data:
            table_data[table_path] = {}

        table_data[table_path]['time_itr'] = time_itr
        table_data[table_path]['time_avg'] = time_avg
        table_data[table_path]['time_ste'] = time_ste
        table_data[table_path]['thread'] = table_dset['threads'][0]
        table_data[table_path]['chiL'] = chiL
        table_data[table_path]['chiR'] = chiR
        table_data[table_path]['bond'] = bond
        table_data[table_path]['mpod'] = table_dset['mpod'][0]
        table_data[table_path]['spin'] = table_dset['spin'][0]
        table_data[table_path]['name'] = table_dset['name'][0].decode("utf-8")
        table_data[table_path]['table_name'] = table_name
        table_data[table_path]['table_path'] = table_path
    return table_data, version_data


def get_table_data(key, table_data, include=None):
    res = []
    for path, table in table_data.items():
        if include == None:
            res.append(table[key])
        elif isinstance(include, dict):
            accept = True
            for ikey, ival in include.items():
                if table[ikey] != ival:
                    accept = False
                    break
            if accept:
                res.append(table[key])
        elif isinstance(include, list):
            if any(k in path for k in include):
                res.append(table[key])
        elif include in path:
            res.append(table[key])
    return res


def plot_time_vs_bond(table_data, version_data, ax, include=None):
    # Collect data
    names = get_table_data('name', table_data, include=include)
    threads = get_table_data('thread', table_data, include=include)

    # There will be one plot per unique name and bond
    names = sorted(list(set(names)))
    threads = sorted(list(set(threads)))

    lwidth = 1.2
    lalpha = 1.0
    lstyle = '-'
    lstyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    for thread, name, in itertools.product(threads, names):
        subinclude = {'name': name, 'thread': thread} | include
        bonds = get_table_data("bond", table_data, include=subinclude)
        time_avgs = get_table_data("time_avg", table_data, include=subinclude)
        time_stes = get_table_data("time_ste", table_data, include=subinclude)

        label = "{} $t$:{}".format(name, thread)
        if 'eigen' in name:
            label = "{} {} $t$:{}".format(version_data['eigen_version'], name, thread)
            lstyle = lstyles[threads.index(thread)]
            color = 'blue'
            if name == 'eigen2':
                color = 'black'
            if name == 'eigen3':
                color = 'cyan'

        if 'xtensor' in name:
            lstyle = lstyles[threads.index(thread)]
            color = 'green'
        if 'tblis' in name:
            lwidth = 1.6
            lstyle = lstyles[threads.index(thread)]
            color = 'red'
        ax.errorbar(x=bonds, y=np.asarray(time_avgs), yerr=np.asarray(time_stes), label=label, capsize=2,
                    linestyle=lstyle, color=color,
                    elinewidth=0.3, markeredgewidth=0.8, linewidth=lwidth, alpha=lalpha)

    ax.set_xlabel('Bond dimension')
    ax.set_ylabel('Time [s]')
    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    title = 'Benchmark for tensor contraction'
    if include:
        title = "{} ${}$".format(title, include)
    ax.set_title(title)


def plot_time_vs_threads(table_data, version_data, ax, include=None):
    # Collect data
    bonds = get_table_data('bond', table_data, include=include)
    names = get_table_data('name', table_data, include=include)
    # There will be one plot per unique name and bond
    bonds = sorted(list(set(bonds)))
    names = sorted(list(set(names)))
    print("bonds", bonds)
    print("names", names)
    lwidth = 1.2
    lalpha = 1.0
    lstyle = '-'
    for bond, name, in itertools.product(bonds, names):
        subinclude = {'name': name, 'bond': bond} | include
        threads = get_table_data("thread", table_data, include=subinclude)
        time_avgs = get_table_data("time_avg", table_data, include=subinclude)
        time_stes = get_table_data("time_ste", table_data, include=subinclude)
        label = "{} $\chi$:{}".format(name, bond)

        if 'eigen' in name:
            label = "{} {} $\chi$:{}".format(version_data['eigen_version'], name, bond)
            lstyle = '-'
        if 'xtensor' in name:
            lstyle = ':'
        if 'tblis' in name:
            lwidth = 1.6
            lstyle = ':'
        ax.errorbar(x=threads, y=time_avgs, yerr=time_stes, label=label, capsize=2,
                    linestyle=lstyle,
                    elinewidth=0.3, markeredgewidth=0.8, linewidth=lwidth, alpha=lalpha)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time [s]')
    ax.legend()
    # plt.yscale('log')
    title = 'Benchmark for tensor contraction'
    if include:
        title = "{} {}".format(title, include)
    ax.set_title(title)

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

tbdb_values = {
    'mode': ['cutensor','tblis', 'eigen1','cyclops', 'xtensor'],
    'type': [#'fp32',
             'fp64', 'cplx'],
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
    # with h5py.File(f, 'r') as h5f:
    #     figrows = 2
    #     figcols = 2
    #     fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows))
    #     fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
    #     fig.subplots_adjust(wspace=0.2, hspace=0.2)
    #
    #     for nomp in tbdb_values['nomp']:
    #         tbdb_plot(ax=axes[0,0], table=h5f['tbdb'], fields=['mode', 'type', 'nomp'], values=['eigen1','fp32', nomp], xkey='chiL', ykey='t_contr')
    #
    #     for mode in tbdb_values['mode']:
    #         if mode == 'cyclops':
    #             tbdb_plot(ax=axes[1,0], table=h5f['tbdb'], fields=['mode', 'type', 'nmpi'],values=[mode, 'fp32', 32], xkey='chiL', ykey='t_contr')
    #         elif mode == 'cutensor':
    #             tbdb_plot(ax=axes[1,0], table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, 'fp32', 0], xkey='chiL', ykey='t_contr')
    #             tbdb_plot(ax=axes[1,0], table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, 'fp32', 1], xkey='chiL', ykey='t_contr')
    #         else:
    #             tbdb_plot(ax=axes[1,0], table=h5f['tbdb'], fields=['mode', 'type', 'nomp'], values=[mode, 'fp32', 32], xkey='chiL', ykey='t_contr')
    #
    #
    #     for mode in tbdb_values['mode']:
    #         if mode == 'cyclops':
    #             tbdb_plot(ax=axes[0,1], table=h5f['tbdb'], fields=['mode', 'type', 'nmpi'],values=[mode, 'fp32', 32], xkey='chiL', ykey='t_contr',yinv=True)
    #         elif mode == 'cutensor':
    #             tbdb_plot(ax=axes[0,1], table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, 'fp32', 0], xkey='chiL', ykey='t_contr',yinv=True)
    #             tbdb_plot(ax=axes[0,1], table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, 'fp32', 1], xkey='chiL', ykey='t_contr',yinv=True)
    #         else:
    #             tbdb_plot(ax=axes[0,1], table=h5f['tbdb'], fields=['mode', 'type', 'nomp'], values=[mode, 'fp32', 32], xkey='chiL', ykey='t_contr',yinv=True)

    # with h5py.File(f, 'r') as h5f:
    #     figrows = 2
    #     figcols = 2
    #     fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows))
    #     fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
    #     fig.subplots_adjust(wspace=0.2, hspace=0.2)
    #
    #     for ax, type in zip(np.ravel(axes), tbdb_values['type']):
    #         for mode in tbdb_values['mode']:
    #             if mode == 'cyclops':
    #                 tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'nmpi'], values=[mode, type, 32],
    #                           xkey='chiL', ykey='t_contr')
    #             elif mode == 'cutensor':
    #                 tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, type, 0],
    #                           xkey='chiL', ykey='t_contr')
    #                 tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'gpun'], values=[mode, type, 1],
    #                           xkey='chiL', ykey='t_contr')
    #             else:
    #                 tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'nomp'], values=[mode, type, 32],
    #                           xkey='chiL', ykey='t_contr')

    # 1 subfigure cores on xaxis, ops on yaxis
    with h5py.File(f, 'r') as h5f:
        figrows = 1
        figcols = 1
        plt.style.use('src/plotting/stylesheets/prl.mplstyle')
        fig, ax = plt.subplots(nrows=figrows, ncols=figcols, figsize=(4.6 * figcols, 3 * figrows))
        ax.set_box_aspect(1.0)
        # fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(left=0.09, bottom=0.10, right=0.65, top=0.94 )
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xticks(tbdb_values['nomp'])
        ax.set_yticks([ 2**j for j in range(-6,6) ])
        # ax.set_yticklabels([f'{2**j:.3f}' if j < 1 else f'{2**j:.0f}' for j in range(-6,6)  ])
        # print([f'{2**j:.3f}' if j < 1 else f'{2**j:.0f}' for j in range(-6,6)  ])
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

        ax.set_title('Bond dimension 1024')
        ax.set_ylabel('op/s')
        ax.set_xlabel('cores')
        lstyles = ['solid', 'dashed', 'dotted']
        palette = sns.color_palette(palette='Dark2', n_colors=len(tbdb_values['mode']))
        legend1 = {'handle': [], 'labels':[]}
        legend2 = {'handle': [], 'labels':[]}

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
                        if not modegpu in legend1['labels']:
                            legend1['handle'].append(Line2D([0], [0], color=color, marker=mrk, label=modegpu))
                            legend1['labels'].append(modegpu)

                else:
                    mode_lbl = mode if mode != 'eigen1' else 'eigen'
                    if mode == 'cyclops':
                        line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'], values=[mode, type, 1024],
                                  xkey='nmpi', ykey='t_contr', yinv=True, color=color,linestyle=lstyle)
                    else:
                        line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'], values=[mode, type, 1024],
                                  xkey='nomp', ykey='t_contr', yinv=True, color=color,linestyle=lstyle)
                    if not mode_lbl in legend1['labels']:
                        legend1['labels'].append(mode_lbl)
                        legend1['handle'].append(Line2D([0], [0], color=color, label=mode_lbl))

                type_lbl = type if type != 'fp64' else 'float64'
                type_lbl = type_lbl if type_lbl != 'cplx' else 'complex128'
                if not type_lbl in legend2['labels']:
                    legend2['handle'].append(Line2D([0], [0], color='black', linestyle=lstyle, label=type_lbl))
                    legend2['labels'].append(type_lbl)

        legend1 = plt.legend(handles=legend1['handle'], loc="upper left", bbox_to_anchor=(0.98,1.0) )
        legend2 = plt.legend(handles=legend2['handle'], loc="upper left", bbox_to_anchor=(0.98,0.6), prop={'family': 'monospace'})

        ax.add_artist(legend1)
        ax.add_artist(legend2)
        # ax.add_artist(legend3)

        # 1 subfigure cores on xaxis, ops on yaxis
        continue
        with h5py.File(f, 'r') as h5f:
            figrows = 3
            figcols = 3
            fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows),sharey='all')
            fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
            fig.subplots_adjust(wspace=0.2, hspace=0.2)
            formatter = ScalarFormatter()
            formatter.set_scientific(False)

            lstyles = ['solid', 'dashed', 'dotted']
            palette = sns.color_palette(palette='Dark2', n_colors=len(tbdb_values['mode']))
            for bidx, bond in enumerate([256,512,1024]):
                for tidx, (ax, type)  in enumerate(zip(np.ravel(axes[bidx,:]), tbdb_values['type'])):
                    gpus = []
                    lstyle = None
                    ax.set_xscale('log', base=2)
                    ax.set_yscale('log', base=2)
                    ax.set_xticks(tbdb_values['nomp'])
                    ax.set_yticks([2 ** j for j in range(-6, 8)])
                    ax.set_title(f'Bond dimension {bond}')
                    ax.set_ylabel('op/s')
                    ax.set_xlabel('cores')
                    ax.xaxis.set_major_formatter(formatter)
                    ax.yaxis.set_major_formatter(formatter)
                    legend_mode = {'handle': [], 'labels': []}
                    legend_type = {'handle': [], 'labels': []}
                    for midx, (mode, color) in enumerate(zip(tbdb_values['mode'], palette)):
                        if mode == 'cutensor':
                            line = tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'],
                                             values=[mode, type, bond, 0], xkey='nomp', ykey='t_contr', yinv=True,
                                             hline=True, color=color, linestyle=lstyle, marker='o')
                            gpus.append('TITAN V')
                            line = tbdb_plot(ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL', 'gpun'],
                                             values=[mode, type, bond, 1], xkey='nomp', ykey='t_contr', yinv=True,
                                             hline=True, color=color, linestyle=lstyle, marker='v')
                            gpus.append('RTX 2080 TI')
                            for gpu, mrk in zip(gpus, ['o', 'v']):
                                modegpu = f'{mode} {gpu}'
                                if not modegpu in legend_mode['labels']:
                                    legend_mode['handle'].append(Line2D([0], [0], color=color, marker=mrk, label=modegpu))
                                    legend_mode['labels'].append(modegpu)

                        else:
                            if mode == 'cyclops':
                                line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'],
                                                 values=[mode, type, bond],
                                                 xkey='nmpi', ykey='t_contr', yinv=True, color=color, linestyle=lstyle)
                            else:
                                line = tbdb_plot(ax=ax, table=h5f['tbdb'], fields=['mode', 'type', 'chiL'],
                                                 values=[mode, type, bond],
                                                 xkey='nomp', ykey='t_contr', yinv=True, color=color, linestyle=lstyle)
                            if not mode in legend_mode['labels']:
                                legend_mode['handle'].append(Line2D([0], [0], color=color, label=mode))
                                legend_mode['labels'].append(mode)
                        if not type in legend_type['labels']:
                            legend_type['handle'].append(Line2D([0], [0], color='black', linestyle=lstyle, label=type))
                            legend_type['labels'].append(type)
                    print(f'adding artist {tidx=} {midx=}')
                    legend_type_artist = ax.legend(handles=legend_type['handle'], loc='lower right')
                    ax.add_artist(legend_type_artist)
                    if bidx == 0 and tidx == 0:
                        legend_mode_artist = ax.legend(handles=legend_mode['handle'], loc='lower left')
                        ax.add_artist(legend_mode_artist)

plt.savefig('2023-04-18/bond1024-fp64-cplx.pdf', format='pdf', backend='pgf')
plt.show()
