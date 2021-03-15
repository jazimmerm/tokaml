from numpy import argmin, abs, array

def index_match(arr1, time):
    return argmin(abs(array(arr1) - time))