import numpy as np
import csv
from matplotlib import pyplot as plt

path = '../results/'
results = np.load(path + "allExperiments.npy" )
opt = np.load(path + "opt.npy")