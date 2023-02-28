#!/usr/bin/python
import platform

def auto_replacer(path):
    path_seg = path.split('/',2)[2]
    mac_path = '/Users/'
    linux_path = '/home/'
    if 'Darwin' in platform.system():
        local_path = mac_path + path_seg
    elif 'Linux' in platform.system():
        local_path = linux_path + path_seg
    return local_path
