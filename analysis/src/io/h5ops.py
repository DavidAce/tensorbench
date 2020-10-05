import numpy as np
import h5py
from src.general.natural_sort import *



def h5open(h5filenames, permission='r',chunk_cache_mem_size=1000*1024**2,driver=None,swmr=False):
    if len(h5filenames) == 1 or isinstance(h5filenames,str):
        try:
            return h5py.File(h5filenames, permission,swmr=swmr,rdcc_nbytes=chunk_cache_mem_size,rdcc_nslots=chunk_cache_mem_size,driver=driver)
        except Exception as err:
            print('Could not open file!:', err)

    else:
        h5files = []
        for name in h5filenames:
            h5files.append(h5open(name,permission))
        return h5files


def h5close(h5files):
    if isinstance(h5files, h5py.File):  # Just HDF5 files
        try:
            h5files.flush()
            h5files.close()
        except:
            pass  # Was already closed

    elif isinstance(h5files,list):
        for file in h5files:
            h5close(file)
    else:
        print("Trying to close h5 files that aren't in a list.")
        exit(1)


def h5py_dataset_iterator(g, prefix='', filter='',dep=1):
    if dep == 0:
        return
    key_sorted = sorted(g.keys(), key=natural_keys)
    for key in key_sorted:
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if key.startswith(tuple(filter)) and isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group) and dep > 0: # test for group (go down)
            yield from h5py_dataset_iterator(g=item, prefix=path, filter=filter,dep=dep-1)

def h5py_group_iterator(g, prefix='', filter='',dep=1):
    if dep == 0:
        return
    key_sorted = sorted(g.keys(), key=natural_keys)
    for key in key_sorted:
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(filter,list) or isinstance(filter,dict):
            if any(f in key for f in filter) and isinstance(item, h5py.Group):  # test for group
                yield (key, path, item)
            elif isinstance(g[key], h5py.Group) and dep > 0:  # test for group (go down)
                yield from h5py_group_iterator(g=g[key], prefix=path, filter=filter, dep=dep - 1)
        elif filter in key and isinstance(item, h5py.Group):
            yield (key, path, item)
        elif isinstance(item, h5py.Group) and dep > 0: # test for group (go down)
            yield from h5py_group_iterator(g=item, prefix=path, filter=filter,dep=dep-1)


def h5py_node_iterator(g, prefix='', filter='',  dep=1):
    if dep == 0:
        return
    key_sorted = sorted(g.keys(), key=natural_keys)
    for key in key_sorted:
        path = '{}/{}'.format(prefix, key)
        if isinstance(filter,list) or isinstance(filter,dict):
            if any(f in key for f in filter):
                yield (key,path, g[key])
            elif isinstance(g[key], h5py.Group) and dep > 0:  # test for group (go down)
                yield from h5py_node_iterator(g=g[key], prefix=path, filter=filter, dep=dep - 1)
        elif filter in key:
            yield (key,path,g[key])
        elif isinstance(g[key], h5py.Group) and dep > 0:  # test for group (go down)
            yield from h5py_node_iterator(g=g[key], prefix=path, filter=filter,dep=dep-1)

def h5py_node_finder(g, filter='', num=0, dep=1, includePath=True):
    matches = []
    for (key,path,node) in h5py_node_iterator(g,filter=filter,dep=dep):
        matches.append((key,path,node))
        if len(matches) >= num and num > 0:
            break
    if includePath:
        return matches
    else:
        return [x[2] for x in matches]

def h5py_unique_finder(g, filter='',dep=1):
    matches = h5py_node_finder(g=g,filter=filter,dep=dep)
    matches = [x[1].split("/")[-1] for x in matches]
    return list(sorted(set(matches)))


def h5py_dataset_finder(g, filter='', num=0, dep=1,includePath=True):
    matches = h5py_node_finder(g=g,filter=filter,num=num,dep=dep,includePath=includePath)
    result = []
    for match in matches:
        result.append(match)
    return result

def load_component_cplx(hdf5_obj,path,filter_name='', type=np.complex128):
    key_sorted = sorted(hdf5_obj[path].keys(), key=natural_keys)
    ret_list = []
    if filter_name=='':
        for key in key_sorted:
            ret_list.append(np.asarray(hdf5_obj[path][key].value.view(dtype=np.complex128)))
    else:
        for key in filter(lambda list: filter_name in list, key_sorted):
            ret_list.append(np.asarray(hdf5_obj[path][key].value.view(dtype=np.complex128)))
    return ret_list

    # if len(ret_list) == 1:
    #     return ret_list[0]
    # else:
    #

