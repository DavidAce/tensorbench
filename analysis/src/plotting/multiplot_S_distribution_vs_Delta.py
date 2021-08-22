from src.plotting.tools import *
import matplotlib.pyplot  as plt
from src.database.database import *
from src.general.filter import *
import itertools


def multiplot_Smid_vs_Delta(h5_src, db=None, plotdir='', algo_filter='', state_filter=''):
    print('Plotting: Entanglement entropy distribution vs Delta for: ', algo_filter, state_filter)
    if not db:
        db = load_database(h5_src, 'entanglement_entropy_midchain', algo_filter, state_filter)

    for sizekey in db['keys']['size']:
        figrows, figcols = get_optimal_subplot_num(1+len(db['keys']['lambda']))
        fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(4.5 * figcols, 4.5 * figrows), sharey='all',
                                 sharex='all')
        fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)
        axes_used = []
        for lambdaidx, (lambdakey, ax) in enumerate(zip(db['keys']['lambda'], np.ravel(axes))):
            ed_palette = itertools.cycle(sns.color_palette("Set2"))
            current_palette = itertools.cycle(sns.color_palette())
            lamb = None
            size = None
            nlabel = {'line':[], 'text': []}
            for algoidx, algokey in enumerate(db['keys']['algo']):
                for stateidx, statekey in enumerate(db['keys']['state']):
                    if not contains(algokey,algo_filter) or not contains(statekey,state_filter):
                        continue
                    dsetkeys = [x for x in db['dsets'] if
                                sizekey in x and lambdakey in x and algokey in x and statekey in x]
                    if not dsetkeys:
                        continue

                    xdata, ydata, edata, ndata = [], [], [], []
                    lgnd, style = None, None
                    color = next(ed_palette) if "states" in statekey else next(current_palette)
                    for idx, dsetkey in enumerate(dsetkeys):
                        meta = db['dsets'][dsetkey]
                        midx = meta['midx']
                        ndata.append(meta['num'])
                        xdata.append(meta['delta'])
                        ydata.append(meta['datanode']['avg'][midx])
                        edata.append(meta['datanode']['ste'][midx])
                        lamb = lamb if lamb else meta['lambda']
                        size = size if size else meta['size']
                        style = style if style else meta['style']
                        lgnd = lgnd if lgnd else (
                            "ED" if "states" in statekey else
                            re.sub(r'[\W_]', ' ', str(algokey + " " + statekey)))

                    sortIdx = np.argsort(np.asarray(xdata))
                    xdata = np.asarray(xdata)[sortIdx]
                    ydata = np.asarray(ydata)[sortIdx]
                    edata = np.asarray(edata)[sortIdx]
                    line = ax.errorbar(x=xdata, y=ydata, yerr=edata, label=lgnd, color=color, capsize=2,
                                elinewidth=0.3, markeredgewidth=0.8, marker=style['mstyle'],
                                linestyle=style['lstyle'],
                                linewidth=style['lwidth'], alpha=style['lalpha'])
                    nlabel['line'].append(line)
                    nlabel['text'].append('{}'.format(ndata))
                    # for i, n in enumerate(ndata):
                    #     min_S = db['min']['mvg']
                    #     max_S = db['max']['mvg']
                    #     ytext = min_S - 0.06 * (max_S - min_S) * i
                    #     # ax.annotate(txt, (val['x'][i], val['y'][i]), textcoords='data',
                    #     #             xytext=[val['x'][i] - 0.2, ytext],
                    #     #             color=val['color'], fontsize='x-small')
                    #     ax.annotate(n, xy=(xdata[i], ydata[i]), xytext=(xdata[i], ytext), color=color,
                    #                 alpha=0.8, fontsize='x-small')
                    axes_used.append(lambdaidx) if not lambdaidx in axes_used else axes_used

            ax.set_title('$\lambda = {:.4f}$'.format(lamb))
            ax.set_xlabel('$\Delta = \log \\bar J - \log \\bar h$')
            ax.set_ylabel('$S_E(L/2)$')
            # ax.legend(nlabel['line'], nlabel['text'], title='Realizations',
            #           loc='lower right', framealpha=0.2, fontsize='small', labelspacing=0.25, ncol=2)
            fig.suptitle('$L = {}$'.format(size))

        prettify_plot(fig,axes,rows=figrows,cols=figcols,axes_used=axes_used,ymin=db['min']['mvg'], ymax=db['max']['mvg'],nlabel=nlabel)
        if plotdir != '':
            plt.savefig(plotdir + '/Smid_vs_Delta_' + sizekey + '.pdf', format='pdf')
            plt.savefig(plotdir + '/Smid_vs_Delta_' + sizekey + '.png', format='png')



    # for dsetkey, dsetmeta in db['dsets'].items():
    #     k = dsetmeta['keys']
    #     for win_idx, win in enumerate(variance_window_limits):
    #         statenode = h5_src[k['size']][k['lambda']][k['delta']][k['algo']][k['state']]
    #         datanode = h5_src[dsetkey]
    #         idx = get_v_filtered_index_list(statenode, win)
    #         if not idx:
    #             data = datanode['data']
    #         else:
    #             data = datanode['data'][idx]
    #         if np.any(np.isnan(data)):
    #             raise ValueError("Data contains nan's")
    #         num = datanode['num'][()]
    #         avg = datanode['avg'][()]
    #         datarange = [0, db['max']]
    #
    #         hist, edge = np.histogram(data, bins=bins, range=datarange, density=False)
    #         bincentres = [(edge[i] + edge[i + 1]) / 2. for i in range(len(edge) - 1)]
    #         width = np.diff(edge)
    #         norm = np.dot(hist, width)
    #         hist = hist / norm
    #         dsetmeta['data'] = data
    #         dsetmeta['hist'] = hist
    #         dsetmeta['edge'] = edge
    #         if "states" in statenode.name:
    #             nicename = "ED e=[" + statenode.attrs["efmt"] + "]"
    #         else:
    #             nicename = re.sub(r'[\W_]', ' ', str(k['algo'] + " " + k['state']))
    #
    #         dsetmeta['legend'] = nicename + ' (' + str(num) + ')'

    # Let's try plotting
    # num = 0
    # for sizekey in db['keys']['size']:
    #     figrows = len(db['keys']['lambda'])
    #     figcols = len(db['keys']['state']) * len(db['keys']['algo'])
    #     fig, axes = plt.subplots(nrows=figrows, ncols=figcols, figsize=(7 * figrows, 7 * figcols))
    #     fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
    #     fig.subplots_adjust(wspace=0.3, hspace=0.3)
    #     used_ax = 0
    #     for lambdaidx, lambdakey in enumerate(db['keys']['lambda']):
    #         for algoidx, algokey in enumerate(db['keys']['algo']):
    #             for stateidx, statekey in enumerate(db['keys']['state']):
    #                 if num > 10:
    #                     continue
    #                 # Let's histograms into a matrix at fixed lambda varying delta
    #                 rows = bins
    #                 cols = len(db['keys']['delta'])
    #                 data2d = []
    #                 hist2d = np.zeros(shape=(rows, cols))
    #                 deltas = np.zeros(shape=(cols))
    #                 dsetkeys = [x for x in db['dsets'] if
    #                             sizekey in x and lambdakey in x and algokey in x and statekey in x]
    #                 if not dsetkeys:
    #                     continue
    #                 for idx, dsetkey in enumerate(dsetkeys):
    #                     dsetmeta = db['dsets'][dsetkey]
    #                     deltas[idx] = dsetmeta['delta']
    #                     hist2d[:, idx] = dsetmeta['hist']
    #                     data2d.append(dsetmeta['data'])
    #                 sortIdx = np.argsort(deltas)
    #                 deltas = deltas[sortIdx]
    #                 hist2d = hist2d[:, sortIdx]
    #                 extent = [np.min(deltas), np.max(deltas), 0, db['max']]
    #
    #                 ax = axes[lambdaidx, stateidx * len(db['keys']['algo']) + algoidx]
    #                 im = ax.imshow(hist2d,
    #                                origin='lower',
    #                                aspect='auto',
    #                                extent=extent,
    #                                interpolation='nearest',
    #                                cmap=plt.get_cmap('viridis'),  # use nicer color map
    #                                )
    #                 plt.colorbar(im, ax=ax)
    #                 # fig.colorbar(im, orientation='vertical')
    #                 ax.set_xlabel('x')
    #                 ax.set_ylabel('y')
    #                 num = num + 1
    #                 used_ax = used_ax + 1
    #     for ax in np.ravel(axes)[used_ax:]:
    #         fig.delaxes(ax)
    # h5close(h5_src)
    # plt.show()
    # exit(0)

# print('Finding unique')
# path_L = h5py_unique_finder(h5_src, filter='L_', dep=1)
# path_l = h5py_unique_finder(h5_src, filter='l_', dep=2)
# path_d = h5py_unique_finder(h5_src, filter='d_', dep=3)
# # We make one subplot for each system size L
# # Collect 1d-histograms of S_E for each delta
# print('Collecting data')
# hists = {'path_L': path_L, 'path_l': path_l, 'path_d': path_d}
# max_S = 0
# for L in path_L:
#     for l in path_l:
#         for d in path_d:
#             path = L + '/' + l + '/' + d
#             if not path in h5_src:
#                 continue
#             basenode = h5_src[path]
#             hists[path] = {}
#             hists[path]['length'] = basenode.attrs['model_size']
#             hists[path]['delta'] = basenode.attrs['delta']
#             hists[path]['lambda'] = basenode.attrs['lambda']
#             for algokey, algopath, algonode in h5py_group_iterator(g=basenode, filter=algo_filter, dep=1):
#                 for statekey, statepath, statenode in h5py_group_iterator(g=algonode, filter=state_filter, dep=1):
#                     for win_idx, win in enumerate(variance_window_limits):
#                         idx = get_v_filtered_index_list(statenode, win)
#                         hists[path][algopath + statepath] = {}
#                         for datakey, datapath, datanode in h5py_node_finder(g=statenode, filter='entanglement_entropy',
#                                                                             dep=8):
#                             fullpath = path + algopath + statepath + datapath
#                             print('Processing path', fullpath)
#                             if not idx:
#                                 data = datanode['data']
#                             else:
#                                 data = datanode['data'][idx]
#                             if np.any(np.isnan(data)):
#                                 raise ValueError("Data contains nan's")
#                             hists[path][algopath + '/' + statepath]['name'] = datanode.name
#                             hists[path][algopath + '/' + statepath]['path'] = datapath
#                             hists[path][algopath + '/' + statepath]['node'] = datanode
#                             max_S = np.max([max_S, datanode['max'][()]])

# Now we have a list full of data


# hists['']
# num = datanode['num'][()]
# avg = datanode['avg'][()]
# datarange = [np.min(data), np.max(data)]
# hist, edges = np.histogram(data, bins=bins, range=datarange, density=False)
# bincentres = [(edges[i] + edges[i + 1]) / 2. for i in range(len(edges) - 1)]
# widths = np.diff(edges)
# norm = np.dot(hist, widths)
# if "states" in statekey:
#     color = next(ed_palette)
#     nicename  = "ED e=[" + statenode.attrs["efmt"] + "]"
#     lwidth = 2.4
#     lalpha = 0.8
# else:
#     color = next(current_palette)
#     nicename = re.sub(r'[\W_]', ' ',str(algokey + " " + statekey))
#     lwidth = 1.4
#     lalpha = 0.9
#     max_S = np.max([max_S, np.max(data)])
# nicename = nicename + ' (' + str(num) + ')'
# ax.step(bincentres, hist / norm, where='mid', label=nicename, linewidth=lwidth,alpha=lalpha,color=color)
# ax.axvline(avg, linestyle='dashed', linewidth=lwidth,alpha=lalpha,color=color)


# # One figure per unique_l, unique_J and unique_h
# for l in path_l:
#     for d in path_d:
#         # In each figure we want one subplot per unique_L
#         rows, cols = get_optimal_subplot_num(len(path_L))
#         fig,axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7 * cols, 7 * rows))
#         fig.tight_layout(pad=5, w_pad=1.0, h_pad=1.0)
#         fig.subplots_adjust(wspace=0.3, hspace=0.3)
#         used_ax = 0
#         delt = 0
#         lamb = 0
#         for ax, L in zip(np.ravel(axes), path_L):
#             if not L in h5_src or not l in h5_src[L] or not d in h5_src[L][l]:
#                 continue
#             basenode = h5_src[L][l][d]
#             ed_palette = itertools.cycle(sns.color_palette("Set2"))
#             current_palette = itertools.cycle(sns.color_palette("colorblind", 5))
#             chain_length = basenode.attrs['model_size']
#             delt = basenode.attrs['delta']
#             lamb = basenode.attrs['lambda']
#             max_S = 0
#             for algokey,algopath,algonode in h5py_group_iterator(g=basenode,filter=algo_filter,dep=1):
#                 for statekey,statepath,statenode in h5py_group_iterator(g=algonode,filter=state_filter,dep=1):
#                     for win_idx, win in enumerate(variance_window_limits):
#                         idx = get_v_filtered_index_list(statenode, win)
#                         for datakey,datapath,datanode in h5py_node_finder(g=statenode,filter='entanglement_entropy',dep=8):
#                             if not idx:
#                                 data = datanode['data']
#                             else:
#                                 data = datanode['data'][idx]
#                             if np.any(np.isnan(data)):
#                                 raise ValueError("Data contains nan's")
#                             num = datanode['num'][()]
#                             avg = datanode['avg'][()]
#                             datarange = [np.min(data), np.max(data)]
#                             hist, edges = np.histogram(data, bins=bins, range=datarange, density=False)
#                             bincentres = [(edges[i] + edges[i + 1]) / 2. for i in range(len(edges) - 1)]
#                             widths = np.diff(edges)
#                             norm = np.dot(hist, widths)
#                             if "states" in statekey:
#                                 color = next(ed_palette)
#                                 nicename  = "ED e=[" + statenode.attrs["efmt"] + "]"
#                                 lwidth = 2.4
#                                 lalpha = 0.8
#                             else:
#                                 color = next(current_palette)
#                                 nicename = re.sub(r'[\W_]', ' ',str(algokey + " " + statekey))
#                                 lwidth = 1.4
#                                 lalpha = 0.9
#                                 max_S = np.max([max_S, np.max(data)])
#                             nicename = nicename + ' (' + str(num) + ')'
#                             ax.step(bincentres, hist / norm, where='mid', label=nicename, linewidth=lwidth,alpha=lalpha,color=color)
#                             ax.axvline(avg, linestyle='dashed', linewidth=lwidth,alpha=lalpha,color=color)
#
#                 if max_S > 0:
#                     ax.set_xlim(0,max_S)
#                 ax.set_xlabel('$S_E$')
#                 ax.set_ylabel('$P(S_E)$')
#                 ax.set_title('$L = ' + str(chain_length) + '$')
#                 # ax.set_xlim(1e-21,100)
#             used_ax = used_ax + 1
#             ax.legend()
#         fig.suptitle(
#             'Distribution of mid-chain entanglement entropy @ $\Delta = $' + str(delt) + '$\lambda = $' + str(lamb))
#         for ax in np.ravel(axes)[used_ax:]:
#             fig.delaxes(ax)
#         if plotdir != '':
#             Jh = re.sub('/', '_', str(d))
#             plt.savefig(plotdir + '/S_distribution_' + l + '_' + d + '.pdf', format='pdf')
#             plt.savefig(plotdir + '/S_distribution_' + l + '_' + d + '.png', format='png')
