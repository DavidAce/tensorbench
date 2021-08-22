import numpy as np
import matplotlib.pyplot as plt

def write_attributes(*args, **kwargs):
   for a in args:
       print(a)
   for k,v in kwargs.items():
       print ("%s = %s" % (k, v))



def get_optimal_subplot_num(numplots):
    r = np.sqrt(numplots)
    cols = int(np.ceil(r))
    rows = int(np.floor(r))
    while cols*rows < numplots:
        if (cols <= rows):
            cols = cols+1
        else:
            rows = rows+1
    return rows,cols


def get_empty_figs_w_subplots(num_figs, num_subfigs, figsize=3.5):
    figures = []
    axes    = []
    rows,cols = get_optimal_subplot_num(num_subfigs)
    for i in range(num_figs):
        fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize * cols, figsize * rows),num=i)
        figures.append(fig)
        axes.append(ax)
    return figures,axes



def remove_empty_subplots(fig,axes,axes_used):
    for idx, ax in enumerate(np.ravel(axes)):  # Last one is for the legend
        if not idx in axes_used:
            fig.delaxes(ax)




def prettify_plot(fig,axes,cols,rows,axes_used,xmax=None,xmin=None,ymax=None,ymin=None,nlabel=None):
    for idx, ax in enumerate(np.ravel(axes)):  # Last one is for the legend
        if not idx in axes_used:
            fig.delaxes(ax)
        else:
            ax.set_ylim(ymin=ymin, ymax=ymax)
            ax.set_xlim(xmin=xmin, xmax=xmax)

    handles_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    unique_handles = []
    unique_labels = []
    for handles, labels in handles_labels:
        for handle, label in zip(handles, labels):
            if not label in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)
    ax = fig.add_subplot(rows,cols, rows * cols)
    # Create a legend for the first line.
    if nlabel:
        loc = 'upper center'
    else:
        loc = 'center'

    first_legend = plt.legend(handles=unique_handles,labels=unique_labels,
                              loc=loc,labelspacing=0.25,fontsize='small')

    # Add the legend manually to the current Axes.
    plt.gca().add_artist(first_legend)

    if nlabel:
        second_legend = plt.legend(handles=nlabel['line'],labels=nlabel['text'],
                                   title='Realizations',loc='lower center',
                                   framealpha=0.7, fontsize='x-small', labelspacing=0.25, ncol=1)
        plt.gca().add_artist(second_legend)
    ax.axis('off')
