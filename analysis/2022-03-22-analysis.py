import numpy as np
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot as plt

files = [
    # 'tbdb-2021-08-22-eigen-3.3.7.h5',
    # 'tbdb-2021-08-23-eigen-3.4.0.h5',
    # 'tbdb-2021-08-23-eigen-3.4.0-tblis.h5',
    # 'tbdb-2021-08-23-eigen-3.4.0-tblis-openblas.h5'
    # 'tbdb-2022-03-22.h5'
    'tbdb-2022-03-23.h5'
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
    lstyles = ['-','--',':','-.', '.']
    for thread, name, in itertools.product(threads, names):
        subinclude = {'name': name, 'thread': thread} | include
        bonds = get_table_data("bond", table_data,         include=subinclude)
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
    # plt.yscale('log')
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
        threads = get_table_data("thread", table_data,    include=subinclude)
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


figrows = 2
figcols = 2
fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows))
fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
fig.subplots_adjust(wspace=0.2, hspace=0.2)

for f in files:
    table_data, version_data = get_file_data(f, ["eigen1", "tblis"])
    plot_time_vs_bond(table_data, version_data, axes[0, 0], include={"spin": 2})
    plot_time_vs_bond(table_data, version_data, axes[1, 0], include={"spin": 4})
    plot_time_vs_threads(table_data, version_data, axes[0, 1], include={"spin": 2})
    plot_time_vs_threads(table_data, version_data, axes[1, 1], include={"spin": 4})

figrows = 2
figcols = 2
fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows))
fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
fig.subplots_adjust(wspace=0.2, hspace=0.2)
#
# for f in files:
#     table_data, version_data = get_file_data(f, ["tblis"])
#     plot_time_vs_bond(table_data, version_data, axes[0, 0], include={"spin": 2})
#     plot_time_vs_bond(table_data, version_data, axes[1, 0], include={"spin": 4})
#     plot_time_vs_threads(table_data, version_data, axes[0, 1], include={"spin": 2})
#     plot_time_vs_threads(table_data, version_data, axes[1, 1], include={"spin": 4})

plt.show()
