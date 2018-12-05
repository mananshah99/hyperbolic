import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(color_codes=True)


f = open('distribution.txt', 'r')
x = np.array([float(x) for x in f.readlines()])

sns.distplot(x,kde=False, rug=False)
plt.show()