import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.gridspec as gs

from DataGen import *
from DataHandle import *

#this file needs reviewing, methods may be obsolete and/or outdated 

def plot_25_ims(outlines = False): #plots a 5x5 grid of images as examples
    """
    Plots a 5x5 grid of training images as examples, used in notebook demo
    Parameters:
        arg1: boolean
            If true, outlines will be drawn around the sources
    Returns:
        None
    """
    data, labels = make_data(25)
    fig, axs = plt.subplots(5,5, figsize=(10,10))
    for i in range(5):
        for j in range(5):
            axs[i][j].imshow(data[5*i+j].T, origin = 'lower', vmax = 1, vmin = 0)
            axs[i][j].axis("off")
            if outlines == True:
                o = 0
                while(labels[5*i+j,o][2] != 0):
                    axs[i][j].add_patch(patch.Rectangle((labels[5*i+j,o][0], labels[5*i+j,o][1]),\
                        labels[5*i+j,o][2], labels[5*i+j,o][3], ec='w', fc='none'))
                    o += 1
                    if o > config.max_objects -1:
                        break

                    
def plot_True_Example():
    """
    Plot a single image with the bounding boxes drawn
    Parameters:
        None
    Returns:
        None 
    """
    img, label = make_data(1)
    true = get_tiled_labels(label[0])
    fig = plt.imshow(img[0], vmax = 1, vmin = 0)
    ax = plt.gca()
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.grid(True)
    
    boxes = process_pred(true) #gives true positions, see DataHandle.py 
    for z in range(len(boxes)):
        cx, cy, w, h = boxes[z]
        ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'r', fc = 'r'))
        ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
            h, w, ec='r', fc='none'))  

    
def plot_Pred(img, label, pred, showTrue = True):
    """
    Plot a single image with the the predicted bounding box drawn
    Currently not in use
    """
    
    if showTrue == True:
        true = get_tiled_labels(label[0])
    fig = plt.imshow(img[0], vmax = 1, vmin = 0)
    ax = plt.gca()
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.grid(True)

    for z in range(config.gridN**2):
        i = z//config.gridN
        j = z%config.gridN
        if showTrue == True:
            if true[0][z][0] == 1:
                truex = true[0][z][1]*config.f + i*config.f
                truey = true[0][z][2]*config.f + j*config.f
                truew = true[0][z][3]*config.L
                trueh = true[0][z][4]*config.L
                ax.add_patch(patch.Circle((truey,truex), 0.5, ec = 'r', fc = 'r'))
                ax.add_patch(patch.Rectangle((truey-trueh/2, truex-truew/2),\
                            trueh, truew, ec='r', fc='none'))    
                      
        if pred[0][z][0] > config.filter_threshold:
            w = pred[0][z][3]*config.L
            h = pred[0][z][4]*config.L
            cx = pred[0][z][1]*config.f + i*config.f
            cy = pred[0][z][2]*config.f + j*config.f
            ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'w', fc = 'w'))
            ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
                            h, w, ec='w', fc='none'))             
    
