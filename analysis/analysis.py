import numpy as np
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot  as plt

h5_src = h5open('tbdb.h5', 'r')

table_names = ['cpu_v3', 'cute']

table_data = {}
for name in table_names:
    thread = '' # Num threads
    print(name)
    # for key,group in h5_src.items():
    for table in h5py_dataset_finder(g=h5_src, filter=name, num=0, dep=20):
        print(table)
        table_name = table[0]
        table_path = table[1]
        table_dset = table[2]

        t_iter= table_dset['t_iter'][2:-1]
        t_avg = np.nanmean(t_iter,axis=0)
        t_ste = np.nanstd(t_iter,axis=0) / np.shape(t_iter)[0]

        if not table_name in table_data:
            table_data[table_name] = {'x_data': [], 'y_data':[], 'e_data':[],'thread':''}
        # if table_name in y_data:
        #     y_data[table_name] = []
        # if table_name in e_data:
        #     e_data[table_name] = []
        # x_data[table_name] = []
        # x_data[table_name] = []
        table_data[table_name]['x_data'].append(table_dset['chiL'][-1])
        table_data[table_name]['y_data'].append(t_avg)
        table_data[table_name]['e_data'].append(t_ste)
        # x_data[table_name].append(table_dset['chiL'][-1])
        # y_data[table_name].append(t_avg)
        # e_data[table_name].append(t_ste)
        if('cpu' in table_name):
            table_data[table_name]['thread'] = str(table_dset['threads'][-1])

for name,data in table_data.items():

    time_cpun = table_data[name]['y_data'][-1]
    time_cute = table_data['cute']['y_data'][-1]

    lwidth = 1.8
    lalpha = 1.0
    nicename = name
    if 'cpu' in name:
        speedupstr = "{:.1f}x".format(time_cpun / time_cute)
        nicename = 'cpu' + ' ' + data['thread'] + ' threads -- ' + speedupstr
    if 'cute' in name:
        nicename = 'cuTENSOR (RTX 2080 Ti) -- 1.0x'

    plt.errorbar(x=data['x_data'], y=data['y_data'], yerr=data['e_data'], label=nicename, capsize=2,
                                                        #color=color,
                                                        elinewidth=0.3, markeredgewidth=0.8, linewidth=lwidth, alpha=lalpha)
plt.xlabel('Bond dimension')
plt.ylabel('Time [s]')
# plt.yscale('log')
plt.title('Benchmark for tensor contraction $H^2 |\psi\\rangle$\n spin dim $= 4$ | mpo dim $=5$ \n double precision (64bit)')
plt.legend()
plt.show()