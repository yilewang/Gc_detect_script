
## read data
import scipy.io
import pandas as pd

class Case:
    caseCount = 0
    def __init__(self, path):
        self.file_path = path
        Case.caseCount +=1

    def readFile(self):
        mat = scipy.io.loadmat(self.file_path)
        return mat

        



