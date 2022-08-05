## Detect G max from original data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filePath = 'C:/Users/Wayne/tvb/TVB_workflow/new_g_max/0306A_0.01.csv'
    df = pd.read_csv(filePath, index_col=0)
    plt.plot(df)
    plt.show()

