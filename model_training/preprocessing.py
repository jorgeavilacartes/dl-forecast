"""
Preprocessing is designed to run on batches 

ecg.shape = (batch, signal, lead) or (batch, lead, signal)
"""
# Standard library imports
import functools
import random 
from collections import OrderedDict

import numpy as np

from .pipeline import (
    Pipeline,
    register_in_pipeline,
)

@register_in_pipeline
def subtract_mean(time_serie, *, axis_signal=0):
    "Restar la media a cada feature"
    ts_mean = time_serie.mean(axis=axis_signal, keepdims=True)
    return time_serie-ts_mean

@register_in_pipeline
def minmax(time_serie, *, axis_signal=0, list_min=[], list_max=[]):
    ts_shape = time_serie.shape
    if axis_signal == 0:
        reshape_minmax = (1,ts_shape[1])
    else: 
        reshape_minmax = (ts_shape[1],1)
    
    list_min = np.array(list_min).reshape(reshape_minmax)
    list_max = np.array(list_max).reshape(reshape_minmax)
    epsilon = 0.000001 # TODO verificar si sirve esto
    return (time_serie - list_min + epsilon) / (list_max-list_min)


@register_in_pipeline
def znormalization(time_serie, *, axis_signal=0, list_mean=[], list_std=[]):
    ts_shape = time_serie.shape
    if axis_signal == 0:
        reshape = (1,ts_shape[1])
    else: 
        reshape = (ts_shape[1],1)
    
    list_mean = np.array(list_mean).reshape(reshape)
    list_std = np.array(list_std).reshape(reshape)

    return (time_serie - list_mean) / list_std

