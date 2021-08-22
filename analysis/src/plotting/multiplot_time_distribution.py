
from src.plotting.tools import *
from src.io.h5ops import *
import matplotlib.pyplot  as plt
from src.plotting.filter import *
import datetime

def multiplot_time_distribution(src,plotdir='',algo_filter='', state_filter='',type='typical'):
    print('Plotting: Time distribution for: ', algo_filter, state_filter)
    h5_src = h5open(src, 'r')
    path_L = h5py_unique_finder(h5_src,keypattern='L_',dep=1)
    path_l = h5py_unique_finder(h5_src,keypattern='l_',dep=2)
    path_d = h5py_unique_finder(h5_src,keypattern='d_',dep=3)

    timecounter = 0
    simcounter  = 0
    # One figure per unique_l, unique_J and unique_h
    for l in path_l:
        for d in path_d:
            # In each figure we want one subplot per unique_L
            rows, cols = get_optimal_subplot_num(len(path_L))
            fig,axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7 * cols, 7 * rows))
            fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
            fig.subplots_adjust(wspace=0.3, hspace=0.3)
            used_ax = 0
            delt = 0
            lamb = 0
            for ax, L in zip(np.ravel(axes), path_L):
                current_palette = itertools.cycle(sns.color_palette())
                key_num = 0
                basenode = h5_src[L][l][d]
                chain_length = basenode.attrs['model_size']
                delt = basenode.attrs['delta']
                lamb = basenode.attrs['lambda']
                for algokey, algopath, algonode in h5py_group_iterator(node=basenode, keypattern=algo_filter, dep=1):
                    for statekey, statepath, statenode in h5py_group_iterator(node=algonode, keypattern=state_filter,dep=1):
                        for win_idx, win in enumerate(variance_window_limits):
                            idx = get_v_filtered_index_list(statenode, win)
                            for datakey, datapath, datanode in h5py_node_finder(node=statenode,
                                                                                keypattern='algorithm_time',
                                                                                dep=8):
                                if "checkpoint" in datapath:
                                    continue
                                if not idx:
                                    data = np.asarray(datanode['data'])
                                else:
                                    data = np.asarray(datanode['data'][idx])
                                key_num = key_num + 1
                                num = datanode['num'][()]
                                avg = datanode['avg'][()]
                                hist,edges  = np.histogram(data/60, bins=20, density=False)
                                nicename = re.sub(r'[\W_]', ' ', str(algokey + " " + statekey))
                                nicename = nicename + ' (' + str(num) + ')'
                                bincentres = [(edges[i] + edges[i + 1]) / 2. for i in range(len(edges) - 1)]
                                widths = np.diff(edges)
                                color = next(current_palette)
                                lwidth = 1.25
                                lalpha = 1.0
                                ax.step(bincentres, hist, where='mid', label=nicename,
                                        color=color, alpha=lalpha, linewidth=lwidth)
                                ax.axvline(avg / 60, color=color, linestyle='dashed', linewidth=lwidth)
                                timecounter = timecounter + np.sum(data)
                                simcounter = simcounter + num
                ax.set_xlabel('Time [minutes]')
                ax.set_ylabel('Histogram')
                ax.set_title('$L = ' + str(chain_length) + '$')
                ax.set_xlim(left=0)
                ax.set_yscale('log')
                ax.legend()
                used_ax = used_ax + 1
            fig.suptitle(
                'Distribution Simulation time @ $\Delta = ' + str(delt) + '\quad \lambda = ' + str(lamb) + '$')
            for ax in np.ravel(axes)[used_ax:]:
                fig.delaxes(ax)

            if plotdir != '':
                plt.savefig(plotdir + '/Time_distribution_' + l + '_' + d + '.pdf', format='pdf')
                plt.savefig(plotdir + '/Time_distribution_' + l + '_' + d + '.png', format='png')

    time_tot = str(datetime.timedelta(seconds=timecounter))
    time_sim = str(datetime.timedelta(seconds=timecounter / simcounter))
    print("Total sim time      : ", time_tot)
    print("Total num sims      : ", simcounter)
    print("Average time per sim: ", time_sim)


    h5close(h5_src)
