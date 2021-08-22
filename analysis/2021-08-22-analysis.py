import numpy as np
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot as plt

files = ['tbdb-2021-08-22-eigen-3.3.7.h5',
         'tbdb-2021-08-23-eigen-3.4.0.h5',
         'tbdb-2021-08-23-eigen-3.4.0-tblis.h5'
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
        table_data[table_path]['threads'] = table_dset['threads'][0]
        table_data[table_path]['chiL'] = chiL
        table_data[table_path]['chiR'] = chiR
        table_data[table_path]['bond'] = bond
        table_data[table_path]['mpod'] = table_dset['mpod'][0]
        table_data[table_path]['spin'] = table_dset['spin'][0]
        table_data[table_path]['name'] = table_dset['name'][0].decode("utf-8")
        table_data[table_path]['table_name'] = table_name
    return table_data, version_data


def plot_time_vs_bond(table_data, version_data, ax):
    # Collect data
    names = [data['name'] for data in table_data.values()]
    threads = [data['threads'] for data in table_data.values()]
    # There will be one plot per unique name and bond
    names = list(set(names))
    threads = list(set(threads))

    lwidth = 1.8
    lalpha = 1.0
    lstyle = '-'
    for thread, name, in itertools.product(threads, names):
        bonds = [data['bond'] for data in table_data.values() if name == data['name'] and thread == data['threads']]
        time_avgs = [data['time_avg'] for data in table_data.values() if name == data['name'] and thread == data['threads']]
        time_stes = [data['time_ste'] for data in table_data.values() if name == data['name'] and thread == data['threads']]
        label = "{} $t$:{}".format(name, thread)
        if 'eigen' in name:
            label = "{} {} $t$:{}".format(version_data['eigen_version'], name, thread)
            lstyle = '-'
        if 'xtensor' in name:
            lstyle = ':'
        if 'tblis' in name:
            lstyle = '-.'
        ax.errorbar(x=bonds, y=time_avgs, yerr=time_stes, label=label, capsize=2,
                    linestyle=lstyle,
                    elinewidth=0.3, markeredgewidth=0.8, linewidth=lwidth, alpha=lalpha)

    ax.set_xlabel('Bond dimension')
    ax.set_ylabel('Time [s]')
    ax.legend()
    # plt.yscale('log')
    ax.set_title('Benchmark for tensor contraction')


def plot_time_vs_threads(table_data, version_data, ax):
    # Collect data
    bonds = [data['bond'] for data in table_data.values()]
    names = [data['name'] for data in table_data.values()]
    # There will be one plot per unique name and bond
    bonds = list(set(bonds))
    names = list(set(names))

    lwidth = 1.8
    lalpha = 1.0
    lstyle = '-'
    for bond, name, in itertools.product(bonds, names):
        threads = [data['threads'] for data in table_data.values() if name == data['name'] and bond == data['bond']]
        time_avgs = [data['time_avg'] for data in table_data.values() if name == data['name'] and bond == data['bond']]
        time_stes = [data['time_ste'] for data in table_data.values() if name == data['name'] and bond == data['bond']]

        label = "{} $\chi$:{}".format(name, bond)

        if 'eigen' in name:
            label = "{} {} $\chi$:{}".format(version_data['eigen_version'],name, bond)
            lstyle = '-'
        if 'xtensor' in name:
            lstyle = ':'
        if 'tblis' in name:
            lstyle = '-.'

        ax.errorbar(x=threads, y=time_avgs, yerr=time_stes, label=label, capsize=2,
                    linestyle=lstyle,
                    elinewidth=0.3, markeredgewidth=0.8, linewidth=lwidth, alpha=lalpha)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time [s]')
    ax.legend()
    # plt.yscale('log')
    ax.set_title(
        'Benchmark for tensor contraction')


figrows = 1
figcols = 2
fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows))
fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
fig.subplots_adjust(wspace=0.2, hspace=0.2)

for f in files:
    table_data,version_data = get_file_data(f,["eigen","tblis"])
    plot_time_vs_bond(table_data,version_data, axes[0])
    plot_time_vs_threads(table_data,version_data, axes[1])
plt.show()
