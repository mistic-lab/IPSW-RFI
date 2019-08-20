import os

class Config:
    L = 32#one dimensional length of image 
    f = 8 #focus of one tile 
    gridN = int(L/f) #number of grids to split the input image into     
    c = 4 #context 
    
    #More than 1 box is not implemented - #TODO 
    boxN = 1 #number of boxes for each grid square
    
    #image generation
    min_objects = 0
    max_objects = 5
    filter = 0.8 #applies a gaussian filter with this number as sigma 
    noise = 0.2 #background noise of the generated image
    #source generation
    min_l = 4
    max_l = 10
    min_w = 1
    max_w = 4
     
    #default: d_low = 1, d_high = 6, , r_flip = 0.5
    #to get straight vertical lines, set d_low = max_l, d_high = max_l + 1, r_flip = 1
    
    #d represents the diagonal ratio of the sources. i.e d=2 means it takes 2 horizontal to get 1 vertical
    d_low = 1 
    d_high = 6
    
    #percentage to flip, when no rotation is added, all lines are 3rd quadrant to 1st quadrant
    r_flip = 0.5 #percentage to flip
    
    #learning parameters 
    dataN = 2500 #size of data set to generate
    train_percent = 0.8 #split the data into train and test sets
    lr = 0.001 #learning rate for optimizer 
    batch_size = 100 #mini-batching 
    keep_prob = 0.75 #probability that a node will remain during dropout 
    
    checkpoint = 10 #how many epochs between saving the network
    
    filter_threshold = 0.5 #minimum confidence for box not to be removed
    mixing_iou = 0.0 #minimum IOU needed to merge boxes