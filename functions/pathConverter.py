#!/usr/python

import platform
def pathcon(extend):
    """
    function path converter:
    Input: 
        desktop path
    Output:
        laptop path
    """

    if platform.node() == 'laptop':
        path = 'C:/Users/wayne/tvb/' + extend
    else:
        path = 'C:/Users/Wayne/tvb/' + extend
    return path