import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

"""
Code adapted from:
Keen, Ben. Parallel Coordinates in Matplotlib. May 2017
benalexkeen.com/parallel-coordinates-in-matplotlib
"""


def ParallelPlot(data, columns, labels = []):

    num_Ticks = 6
    
    if len(labels) == 0:
        labels = np.zeros(len(data))
    
    cm = plt.get_cmap('jet')
    num_clusters = int(max(labels+1))
    colours = [cm(1.*i/num_clusters) for i in range(num_clusters)]
    colours.append((0,0,0,1)) #-1 label will be black
    
    
    x = [i for i, _ in enumerate(columns)]

    plt.rc('axes', labelsize = 18)
    plt.rc('xtick', labelsize = 14)
    plt.rc('ytick', labelsize = 14)
    fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,10))
    plt.subplots_adjust(wspace=0)

    
    min_max_range = []
    for i in x:
        mini = np.min(data[:,i])
        maxi = np.max(data[:,i])
        ran = maxi-mini
        
        min_max_range.append([mini, maxi, ran])
        data[:,i] = np.true_divide(data[:,i] - mini, ran)
        
    for i, ax in enumerate(axes):
        for j in range(np.shape(data)[0]):
            ax.plot(x, data[j], c = colours[int(labels[j])])
        ax.set_xlim(x[i], x[i+1])
            
    def set_ticks_for_axis(i, ax, ticks):
        min_val, max_val, val_range = min_max_range[i]
        step = val_range / (ticks - 1.0)
        tick_labels = [round(min_val + step*j, 2) for j in range(ticks)]
        
        norm_min = np.min(data[:,i])
        norm_range = np.ptp(data[:,i])
        norm_step = norm_range / (ticks - 1.0)
        ticks = [round(norm_min + norm_step*j, 2) for j in range(ticks)]
        
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)
        
    for i,ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([i]))
        set_ticks_for_axis(i, ax, ticks = num_Ticks)
        ax.set_xticklabels([columns[i]])
        
    ax = plt.twinx(axes[-1])
    i = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(i, ax, ticks = num_Ticks)
    ax.set_xticklabels([columns[-2], columns[-1]])
    
    