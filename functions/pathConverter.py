#!/usr/python

import platform

def pathcon(extend):
    """
    function path converter:
    Input: 
        windows path
    Output:
        linux path
    """

    if platform.system() == 'Linux':
        path = '/media/wayne/Linux/tvb/' + extend
    else:
        path = 'C:/Users/Wayne/tvb/' + extend
    return path