import numpy as np
from common_good import *
from src.io.h5ops import *
from src.plotting.style import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D




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
    ymed = []
    ystd = []
    yste = []
    xval = []
    xkey_internal = xkey
    if 'cores' in xkey_internal:
        xkey_internal = 'nmpi'
    for val in tbdb_vals[xkey_internal]:
        rows = get_rows(table, fields + [xkey_internal] , values + [val])
        if len(rows) == 0:
            xval.append(np.nan)
            yavg.append(np.nan)
            ymed.append(np.nan)
            ystd.append(np.nan)
            yste.append(np.nan)
            continue
            raise ValueError(f"no rows found in file: {table.file=} : {fields + [xkey_internal]=} , {values + [val] = }")
        if 'cores' in xkey:
            nomp_val = rows['nomp'][-1]
            nmpi_val = rows['nmpi'][-1]
            xval.append(nomp_val * nmpi_val)
        elif 'nmpi' in xkey:
            xval.append(rows['nmpi'][-1])
        elif 'nmpo' in xkey:
            xval.append( rows['nomp'][-1])
        else:
            xval.append(rows[xkey][-1])
        # print(f'{rows[xkey]=} | {rows['nomp']}')
        if yinv:
            data = 1.0/rows[ykey]
        else:
            data = rows[ykey]
        yavg.append(np.nanmean(data))
        ymed.append(np.nanmedian(data))
        ystd.append(np.nanstd(data))
        yste.append(ystd[-1]/np.sqrt(len(rows)))
    return np.asarray(xval,dtype=np.float64), np.asarray(yavg,dtype=np.float64), np.asarray(ymed,dtype=np.float64), np.asarray(ystd,dtype=np.float64), np.asarray(yste,dtype=np.float64)



def get_processed_data(table, tbdb_vals, fields, values, xkey, ykey, yinv=False,hline=False, **kwargs):
    xval, yavg, ymed, ystd, yste = get_rows_stats(table=table, tbdb_vals=tbdb_vals,fields=fields,values=values, xkey=xkey, ykey=ykey, yinv=yinv)
    if hline and len(yavg) > 0 and len(ystd)>0 and len(yste) > 0:
        if xkey == 'cores':
            xval = np.asarray(np.unique([nomp * nmpi for nomp in tbdb_vals['nomp'] for nmpi in tbdb_vals['nmpi']]), np.float64)
        elif xkey in tbdb_vals:
            xval = np.asarray(tbdb_vals[xkey], dtype=np.float64)
        else:
            raise KeyError(f'{xkey} not in {tbdb_vals.keys()=}')
        yavg = np.asarray([yavg[0]]*len(xval))
        ymed = np.asarray([ymed[0]]*len(xval))
        ystd = np.asarray([ystd[0]]*len(xval))
        yste = np.asarray([yste[0]]*len(xval))
    if 'device' in fields:
        index = fields.index('device')
        device = values[index]
        if device in tbdb_cpurdict:
            if maxcores := tbdb_cpurdict.get(device).get('maxcores'):
                for i, x in enumerate(xval):
                    if x > maxcores:
                        yavg[i] = np.nan
                        ymed[i] = np.nan
                        ystd[i] = np.nan
                        yste[i] = np.nan
                        xval[i] = np.nan

    print(f'{xval=} | {yavg=} | {ystd}')
    if np.isnan(xval).all():
        print(f"--- no data found: {table.file=} | {xkey=}: {xval=} | {fields=} | {values=}")
        yavg *= np.nan
        ymed *= np.nan
        ystd *= np.nan
        yste *= np.nan
        xval *= np.nan

    yavg = yavg[~np.isnan(yavg)]
    ymed = ymed[~np.isnan(ymed)]
    ystd = ystd[~np.isnan(ystd)]
    yste = yste[~np.isnan(yste)]
    xval = xval[~np.isnan(xval)]

    return xval, yavg , ymed, ystd, yste

def tbdb_plot(ax, table, tbdb_vals, fields, values, xkey, ykey, yinv=False,hline=False, **kwargs):
    xval, yavg, ymed, ystd, yste = get_processed_data(table=table, tbdb_vals=tbdb_vals, fields=fields, values=values, xkey=xkey, ykey=ykey, yinv=yinv, hline=hline)
    return ax.errorbar(x=xval, y=ymed, yerr=ystd, **kwargs)

def create_plot(axes, tbdb_vals, legends = None):
    if not isinstance(axes, list):
        raise TypeError("axes must be a list")

    for type, ax in zip(tbdb_vals['type'], axes):
        if legends is None:
            legends = {}
        if not ax.get_label() in legends:
            legends[ax.get_label()] = {}
        if not 'libs' in  legends[ax.get_label()]:
            legends[ax.get_label()]['libs'] = {'handle': [], 'labels': []}
            legends[ax.get_label()]['type'] = {'handle': [], 'labels': []}
            legends[ax.get_label()]['cpus'] = {'handle': [], 'labels': []}
            legends[ax.get_label()]['gpus'] = {'handle': [], 'labels': []}

        legs = legends[ax.get_label()]
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        if 'cores' in tbdb_vals['xkey']:
            xlabel = 'cores'
            xlimit = np.max(np.unique([nomp * nmpi for nomp in tbdb_vals['nomp'] for nmpi in tbdb_vals['nmpi']]))
        if 'omp' in tbdb_vals['xkey']:
            xlabel = 'cores'
            xlimit = np.max(tbdb_vals['nomp'])
        if 'mpi' in tbdb_vals['xkey']:
            xlabel = 'nodes'
            xlimit = np.max(tbdb_vals['nmpi'])
        xticks = np.unique([np.min([2 ** j, xlimit]) for j in range(0, 12)])
        ax.set_xticks(xticks)
        ax.set_yticks([2 ** j for j in range(-6, 6)])
        # ax.set_title('Bond dimension 1024')
        ax.set_ylabel('op/s')
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        for f in tbdb_vals['file']:
            with h5py.File(f, 'r') as h5f:
                device_list = set(h5f['tbdb']['device'][()])
                device_list = [x.decode('utf-8') for x in device_list]
                print(f'{device_list=}')
                for lib in tbdb_vals['mode']:
                    for chiL in tbdb_vals['chiL']:
                        for gpu in tbdb_vals['gpus']:
                            if not tbdb_gpudict[gpu]['name'] in device_list:
                                continue
                            if lib == 'cutensor' and tbdb_gpudict[gpu]['name'] in device_list:
                                print(f'plotting {lib} {type} {gpu} {f}')
                                line = tbdb_plot(ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals,
                                                 fields=['mode', 'type', 'chiL', 'device'],
                                                 values=[lib, type, chiL, tbdb_gpudict[gpu]['name']],
                                                 xkey=tbdb_vals['xkey'], ykey='t_contr', yinv=True, hline=True,
                                                 color=tbdb_gpudict[gpu]['color'],
                                                 linestyle='solid', marker=tbdb_gpudict[gpu]['marker'])
                                if not gpu in legs['gpus']['labels']:
                                    legs['gpus']['handle'].append(
                                        Line2D([0], [0], color=tbdb_gpudict[gpu]['color'], linestyle='solid',
                                               label=tbdb_gpudict[gpu]['tag'], marker=tbdb_gpudict[gpu]['marker']))
                                    legs['gpus']['labels'].append(tbdb_gpudict[gpu]['tag'])
                                if not tbdb_libdict[lib]['tag'] in legs['libs']['labels']:
                                    legs['libs']['handle'].append(Line2D([0], [0], color=tbdb_libdict[lib]['color'],
                                                                        label=tbdb_libdict[lib]['tag']))
                                    legs['libs']['labels'].append(tbdb_libdict[lib]['tag'])
                        for cpu in tbdb_vals['cpus']:
                            if not tbdb_cpudict[cpu]['name'] in device_list:
                                continue
                            print(f'plotting {lib} {type} {cpu} {f}')
                            if lib != 'cutensor' and tbdb_cpudict[cpu]['name'] in device_list:
                                # if lib == 'cyclops':
                                #     line = tbdb_plot(ax=ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals,  fields=['mode', 'type', 'chiL', 'device'], values=[lib, type, 1024, tbdb_cpudict[cpu]['name']],
                                #               xkey='nmpi', ykey='t_contr', yinv=True, color=tbdb_libdict[lib]['color'],linestyle=tbdb_cpudict[cpu]['linestyle'])
                                # else:
                                line = tbdb_plot(ax=ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals, fields=['mode', 'type', 'chiL', 'device'], values=[lib, type, chiL, tbdb_cpudict[cpu]['name']], xkey=tbdb_vals['xkey'], ykey='t_contr', yinv=True, color=tbdb_libdict[lib]['color'], linestyle=tbdb_cpudict[cpu]['linestyle'])
                                # line = tbdb_bar(ax=ax, table=h5f['tbdb'], tbdb_vals=tbdb_vals, fields=['mode', 'type', 'chiL', 'device'], values=[lib, type, chiL, tbdb_cpudict[cpu]['name']], xkey=tbdb_vals['xkey'], ykey='t_contr', yinv=True, color=tbdb_libdict[lib]['color'], linestyle=tbdb_cpudict[cpu]['linestyle'])

                                if not tbdb_libdict[lib]['tag'] in legs['libs']['labels']:
                                    legs['libs']['handle'].append(Line2D([0], [0], color=tbdb_libdict[lib]['color'], label=tbdb_libdict[lib]['tag']))
                                    legs['libs']['labels'].append(tbdb_libdict[lib]['tag'])
                                if not tbdb_typedict[type] in legs['type']['labels']:
                                    legs['type']['handle'].append(Line2D([0], [0], color='#E5E5E5', linestyle=None, label=tbdb_typedict[type]))
                                    legs['type']['labels'].append(tbdb_typedict[type])
                                if not tbdb_cpudict[cpu]['tag'] in legs['cpus']['labels']:
                                    legs['cpus']['handle'].append(Line2D([0], [0], color='black', linestyle=tbdb_cpudict[cpu]['linestyle'], label=tbdb_cpudict[cpu]['tag']))
                                    legs['cpus']['labels'].append(tbdb_cpudict[cpu]['tag'])

    return legends


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
                              figsize=(1152*px, 320*px),
                              layout="constrained",
                              sharex=True, sharey=True
                              )

axes['L'].set_axis_off() # The legend panel
axes['B'].yaxis.label.set_visible(False) # use the yaxis labels from A
axes['C'].yaxis.label.set_visible(False) # Use the yaxis labels from A
axes['A'].set_box_aspect(aspect=1)
axes['B'].set_box_aspect(aspect=1)
axes['C'].set_box_aspect(aspect=1)
axes['L'].set_box_aspect(aspect=2)
# fig.suptitle('Large bond dimension $\\chi = 1024$', y = 0.95)

legends = create_plot([axes['A'], axes['B'], axes['C']], tbdb_gpu, legends=None)
legends = create_plot([axes['A'], axes['B'], axes['C']], tbdb_omp, legends=legends)
legends = create_plot([axes['A'], axes['B'], axes['C']], tbdb_omp_cyclops, legends=legends)



for axlabel in ['A', 'B', 'C']:

    legs = legends[axlabel]
    ax  = axes[axlabel]

    legend_type = ax.legend(handles=legs['type']['handle'], loc='lower right', handlelength=0, handleheight=0, handletextpad=0)
    ax.add_artist(legend_type)

    if axlabel != 'C':
        continue

    # arg = np.argsort(legs['libs']['labels'])
    legs['libs']['labels'] = list(np.asarray(legs['libs']['labels'])[:])
    legs['libs']['handle'] = list(np.asarray(legs['libs']['handle'])[:])

    # arg = np.argsort(legs['cpus']['labels'])
    legs['cpus']['labels'] = list(np.asarray(legs['cpus']['labels'])[:])
    legs['cpus']['handle'] = list(np.asarray(legs['cpus']['handle'])[:])

    arg = np.argsort(legs['gpus']['labels'])
    legs['gpus']['labels'] = list(np.asarray(legs['gpus']['labels'])[arg])
    legs['gpus']['handle'] = list(np.asarray(legs['gpus']['handle'])[arg])

    # legend_libs = ax.legend(handles=legs['libs']['handle'], loc='upper left', fontsize=10, bbox_to_anchor=(1.010, 1.0))
    # legend_cpus = ax.legend(handles=legs['cpus']['handle'], loc='lower left', fontsize=10, bbox_to_anchor=(1.010, 0.0))
    # legend_gpus = ax.legend(handles=legs['gpus']['handle'], loc='lower right', fontsize=10, bbox_to_anchor=(1.990, 0.0))


    legend_libs = axes['L'].legend(handles=legs['libs']['handle'], loc='upper center', fontsize=11, bbox_to_anchor=(0.500, 1.0))
    legend_cpus = axes['L'].legend(handles=legs['cpus']['handle'], loc='lower right', fontsize=10, bbox_to_anchor=(0.500, 0.0))
    legend_gpus = axes['L'].legend(handles=legs['gpus']['handle'], loc='lower left', fontsize=10, bbox_to_anchor=(0.500, 0.0))


    axes['L'].add_artist(legend_libs)
    axes['L'].add_artist(legend_cpus)
    axes['L'].add_artist(legend_gpus)




plt.savefig('figs/ops_vs_cores_libs:all_cpus:all_prec:all-chi1024.pdf', format='pdf')
plt.savefig('figs/ops_vs_cores_libs:all_cpus:all_prec:all-chi1024.png', format='png', dpi=600)
plt.show()
