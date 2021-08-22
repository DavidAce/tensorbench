from src.plotting.tools import *
import matplotlib.pyplot  as plt
from src.database.database import *
from src.general.filter import *
import itertools

def multiplot_var_distribution(h5_src, db=None, plotdir='',algo_filter='', state_filter='',vwin=None):
    print('Plotting: Energy Variance distribution for: ', algo_filter, state_filter)
    if not db:
        db = load_database(h5_src, 'energy_variance', algo_filter, state_filter)
    # One figure per L
    # One subplot per delta
    # One line per lambda per state
    for sizekey in db['keys']['size']:
        figrows, figcols = get_optimal_subplot_num(1+len(db['keys']['delta'])) # Add one for the legend
        fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(5 * figcols, 5 * figrows), sharey='all')
        fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        axes_used = []
        max_S = None
        min_S = None
        for deltaidx, (deltakey, ax) in enumerate(zip(db['keys']['delta'], np.ravel(axes))):
            ed_palette = itertools.cycle(sns.color_palette("Set2"))
            numcolors = len(db['keys']['state'])*len(db['keys']['lambda'])
            current_palette = itertools.cycle(sns.color_palette("colorblind", numcolors))
            lstyles = itertools.cycle(['-.','-','--',':',])
            mstyles = itertools.cycle(('.', ',', '+', 'o', '*'))
            delt = None
            size = None
            for algoidx, algokey in enumerate(db['keys']['algo']):
                for stateidx, statekey in enumerate(db['keys']['state']):
                    if not contains(algokey,algo_filter) or not contains(statekey,state_filter):
                        continue
                    dsetkeys = [x for x in db['dsets'] if
                                sizekey in x and deltakey in x and algokey in x and statekey in x]
                    if not dsetkeys:
                        continue

                    lstyle = next(lstyles)
                    mstyle = next(mstyles)
                    # Now we have a set of setkeys with fixed L,d,algo and state, varying l.
                    for dsetidx, dsetkey in enumerate(dsetkeys):
                        meta = db['dsets'][dsetkey]
                        midx = meta['midx']
                        lamb = meta['lambda']
                        delt = meta['delta']
                        size = meta['size']
                        ndata = meta['num']
                        style = meta['style']

                        if "states" in statekey:
                            continue
                        data = meta['datanode']['data']
                        bins = np.logspace(start=-18, stop=0, num=64, endpoint=True)
                        hist, edges = np.histogram(data, bins=bins, density=False)
                        nicename = re.sub(r'[\W_]', ' ', str(algokey + " " + statekey))
                        nicename = nicename + ' (' + str(ndata) + ')'
                        color = next(current_palette)
                        line = ax.hist(x=np.array(data), bins=np.array(edges), linewidth=1, histtype='step', density=False,
                                label=nicename, color=color)

            if vwin:
                ax.axvline(max(1e-16,vwin[0]), color='grey', linestyle='dashed', linewidth=1.5)
                ax.axvline(min(1e+04,vwin[1]), color='grey', linestyle='dashed', linewidth=1.5,
                           label='Var$(H) \in [{:.2e}, {:.2e}]$'.format(vwin[0],vwin[1]))
            # for win_idx, win in enumerate(variance_window_limits):
            #     for lim_idx, lim in enumerate(win):
            #         ax.axvline(lim, color='grey', linestyle='dashed', linewidth=1.5,
            #                    label=variance_window_names[win_idx][lim_idx])

            ax.set_xlabel('Var$(H)$')
            ax.set_ylabel('Histogram')
            ax.set_xscale('log')
            ax.legend(loc='upper right')
            fig.suptitle('$L = {}$'.format(size))
            fig.suptitle('Distribution of energy variance @ $L = {} $'.format(size))
            #$\Delta = $' + str(delt) + '$\lambda = $' + str(lamb))
            axes_used.append(deltaidx) if not deltaidx in axes_used else axes_used

        remove_empty_subplots(fig=fig,axes=axes,axes_used=axes_used)
        if plotdir != '':
            plt.savefig(plotdir + '/Var_distribution_' + sizekey + '.pdf', format='pdf')
            plt.savefig(plotdir + '/Var_distribution_' + sizekey + '.png', format='png')
    return




def multiplot_var_distribution_old(src, plotdir='',algo_filter='', state_filter='',bins=200):
    print('Plotting: Energy Variance distribution for: ', algo_filter, state_filter)
    h5_src = h5open(src, 'r')
    path_L = h5py_unique_finder(h5_src,filter='L_',dep=1)
    path_l = h5py_unique_finder(h5_src,filter='l_',dep=2)
    path_d = h5py_unique_finder(h5_src,filter='d_',dep=3)
    # One figure per path_l, path_d and path_d
    for l in path_l:
        for d in path_d:
            # In each figure we want one subplot per path_l
            rows, cols = get_optimal_subplot_num(len(path_L))
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7 * cols, 7 * rows))
            fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
            fig.subplots_adjust(wspace=0.3, hspace=0.3)
            used_ax = 0
            delt = 0
            lamb = 0
            for ax, L in zip(np.ravel(axes),path_L):
                current_palette = itertools.cycle(sns.color_palette())
                basenode = h5_src[L][l][d]
                chain_length = basenode.attrs['model_size']
                delt = basenode.attrs['delta']
                lamb = basenode.attrs['lambda']
                # h5keys = [item for item in h5_src[L][l][d].keys() if any(s in item for s in key_list)]
                # key_sorted = sorted(h5keys, key=natural_keys)
                for algokey,algopath,algonode in h5py_group_iterator(g=basenode,filter=algo_filter,dep=1):
                    for statekey,statepath,statenode in h5py_group_iterator(g=algonode,filter=state_filter,dep=1):
                        for win_idx, win in enumerate(variance_window_limits):
                            idx = get_v_filtered_index_list(statenode, win)
                            for datakey,datapath,datanode in h5py_node_finder(g=statenode,filter='energy_variance',dep=8):
                                if not idx:
                                    data = datanode['data']
                                else:
                                    data = datanode['data'][idx]
                                num = datanode['num'][()]
                                bins = np.logspace(start=-18, stop=0, num=64, endpoint=True)
                                hist, edges = np.histogram(data, bins=bins, density=False)
                                nicename = re.sub(r'[\W_]', ' ', str(algokey + " " + statekey))
                                nicename = nicename + ' (' + str(num) + ')'
                                color = next(current_palette)
                                ax.hist(x=np.array(data), bins=np.array(edges), linewidth=1, histtype='step', density=False,
                                        label=nicename,color=color)
                                ax.set_xlabel('$\sigma(E)^2$')
                                ax.set_ylabel('Histogram')
                                ax.set_xscale('log')
                                ax.set_title('$L = ' + str(chain_length) + '$')
                used_ax = used_ax + 1
                # color = next(current_palette)
                for win_idx, win in enumerate(variance_window_limits):
                    color = next(current_palette)
                    for lim_idx, lim in enumerate(win):
                        # ax.axvline(lim, color=variance_colors[win_idx], linestyle='dashed', linewidth=1.5, label='Avg ' + nicename)
                        ax.axvline(lim, color='grey', linestyle='dashed', linewidth=1.5,
                                   label=variance_window_names[win_idx][lim_idx])
                ax.legend(loc='upper right')
            fig.suptitle('Distribution of energy variance @ $\Delta = $' + str(delt) + '$\lambda = $' + str(lamb))
            for ax in np.ravel(axes)[used_ax:]:
                fig.delaxes(ax)
            if plotdir != '':
                plt.savefig(plotdir + '/Var_distribution_' + l + '_' + d + '.pdf', format='pdf')
                plt.savefig(plotdir + '/Var_distribution_' + l + '_' + d + '.png', format='png')

    h5close(h5_src)
