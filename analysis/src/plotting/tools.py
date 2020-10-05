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