# Example: 
# gcc-8 hyperbolicity.c -fopenmp -o hyperbolicity
# ./hyperbolicity algo=1 file=../../data/generated/waxman_hyperbolic.txt -verbose ; python distribution.py
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