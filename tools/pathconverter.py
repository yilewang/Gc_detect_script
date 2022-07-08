#!/usr/bin/python
import platform

def prefix_path():
    if 'WSL2' in platform.release():
        path = '/mnt/c/Users/Wayne/tvb'
    else:
        path = '/home/yat-lok/workspace'
    return path