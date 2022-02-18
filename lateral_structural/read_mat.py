
## read data
import scipy.io
import pandas as pd
"""
@Author: Yile Wang

An easy script to read mat file
"""

class Case:
    caseCount = 0
    def __init__(self, path):
        self.file_path = path
        Case.caseCount +=1

    def readFile(self):
        mat = scipy.io.loadmat(self.file_path)
        return mat

        



