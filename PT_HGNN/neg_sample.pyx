# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector
from libc.stdlib cimport rand, srand

from numpy cimport ndarray as array
import numpy as np

ctypedef cset[int] int_set

cdef llrand():
    cdef unsigned long long r = 0
    cdef int i = 0
    for i in range(5):
        r = (r << 15) | (rand() & 0x7FFF)
    return r & 0xFFFFFFFFFFFFFFFFULL

def negative_sample1(array[long, ndim=1] source_node_list, array[long, ndim=1] pos_node_list, int size = 100):

    cdef cvector[int] c_arr
    cdef int_set omission
    cdef int i = 0

    for elem in pos_node_list:
        omission.insert(elem)

    s_high = source_node_list.shape[0]
    while size - i:
        idx = llrand() % s_high
        if not omission.count(source_node_list[idx]):
            c_arr.push_back(source_node_list[idx])
            i += 1
    
    if size == 1:
        return c_arr[0]
    else:
        return c_arr;

def negative_sample(array[long, ndim=1] source_node_list, array[long, ndim=1] pos_node_list, int size = 100):

    cdef cvector[int] c_arr
    cdef int_set omission
    cdef int i = 0

    for elem in pos_node_list:
        omission.insert(elem)

    for node_id in source_node_list:
        if not omission.count(node_id):
            c_arr.push_back(node_id)
            i += 1
        if i == size:
            break
    
    if size == 1:
        return c_arr[0]
    else:
        return c_arr;

def to2Darr(list xy):

    cdef int i, j, h=len(xy), w=len(xy[0])
    cdef array[long, ndim=2] new = np.empty((h,w), dtype=np.int_)
    for i in range(h):
        for j in range(w):
            new[i, j] = xy[i][j]
    return new

def to1Darr(list xy):

    cdef int i, h = len(xy)
    cdef array[long, ndim=1] new = np.empty((h,), dtype=np.int_)
    for i in range(h):
        new[i] = xy[i]
    
    return new