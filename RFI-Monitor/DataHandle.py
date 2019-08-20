import numpy as np
from skimage.transform import resize 
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

from Config import Config
config = Config()

class ImgBuffer():
    """
    Buffer that collects images from different frames
    """
    
    def __init__(self):
        """
        Initialise the data structures required
        """
        self.timer = [] #timer indicates how long ago the source was spotted/updated
        self.duration = [] #how long the source has been in the buffer
        self.appearances = [] #how many times the source appeared 
        self.first_detection = [] #first time the source is observed
        self.latest_detection = [] #last time the source was observed
        self.min_freq = [] #lowest frequency
        self.max_freq = [] #highest frequency
        
        self.final_array = [] #temporary 
        
    def len(self):
        """
        Shortened method for obtaining the length of the buffer
        """
        return len(self.timer)
    
    def update_timer(self):
        """
        Increases time step of all elements in buffer
        """
        self.timer = [x+1 for x in self.timer]
   

    def add(self, box, t,  x):
        """
        Add a new identification to the buffer
        Parameters:
            arg1: numpy.ndarray of size (4), dtype = np.float
                A single bounding box
            arg2: float
                Time since start
            arg3: float
                Lowest frequency, i.e. left most side of the bounding box
        Returns:
            None
        """
    
        self.timer.append(0)
        self.duration.append(1)
        self.appearances.append(1)
        self.first_detection.append(t + box[0] - box[2]/2)
        self.latest_detection.append(t + box[0] + box[2]/2)
        self.min_freq.append(x + box[1] - box[3]/2)
        self.max_freq.append(x + box[1] + box[3]/2)

    def remove(self, ind):
        """
        Remove one source based on index
        Parameters:
            arg1: int
                Index of the source to be removed
        Returns:
            None
        """
        
        self.timer.pop(ind)
        self.duration.pop(ind)
        self.appearances.pop(ind)
        self.first_detection.pop(ind)
        self.latest_detection.pop(ind)
        self.min_freq.pop(ind)
        self.max_freq.pop(ind)
        
    
    def process_new(self, boxes, t, x):
        """
        When a new source is added, check against those in the buffer
        Parameters:
            arg1: list of boxes (_,4)
                All the bounding boxes detected
            arg2: float
                Time since start
            arg3: float
                Lowest frequency, i.e. left most side of the bounding box
        Returns:
            None
        """
        
        for z in range(len(boxes)): 
            inbuffer = False #if this box doesn't match any in the buffer, this will stay false
            for j in range(len(self.timer)):   
                new_minfreq = x + boxes[z][1] - boxes[z][3]/2
                new_maxfreq = x + boxes[z][1] + boxes[z][3]/2
                new_firstdet = t + boxes[z][0] - boxes[z][2]/2
                new_lastdet = t + boxes[z][0] + boxes[z][2]/2
                
                #compare the frequencies and the time of the two boxes
                if compare_freq(self.min_freq[j], self.max_freq[j], new_minfreq, new_maxfreq) \
                    and compare_time(self.first_detection[j], self.latest_detection[j], new_firstdet, new_lastdet): 
                    
                    #update the frequencies to include all the data
                    self.min_freq[j] = min(self.min_freq[j], new_minfreq)
                    self.max_freq[j] = max(self.max_freq[j], new_maxfreq)
                    
                    self.latest_detection[j] = t + boxes[z][0] + boxes[z][2]/2   
                    self.duration[j] += self.timer[j]
                    self.timer[j] = 0 
                    self.appearances[j] += 1
                    inbuffer = True
            
            if inbuffer == False:
                self.add(boxes[z], t, x) #if not comparable to anything in buffer, add the source
    
    #process the current sources in the buffer
    def process_existing(self):
        """
        Iterates through all entries in the buffer
        Checks if the source has not been updated recently and can be removed
        """
        
        z = 0 
        while z < self.len():
            if self.timer[z] > 8: #if the source has not been updated recently 
                if self.appearances[z] > config.gridN*1: #has to appear enough times to be added
                    self.final_array.append([self.first_detection[z], self.min_freq[z], self.latest_detection[z], self.max_freq[z]])
                self.remove(z)     
                z -= 1
            z += 1
            
    def clear_buffer(self):
        """
        Clears the buffer at the end of a search, storing those that are persistant enough
        """
        z = 0
        while z < self.len():
            if self.appearances[z] > config.gridN*1:
                self.final_array.append([self.first_detection[z], self.min_freq[z], self.latest_detection[z], self.max_freq[z]])
                self.remove(z)     
                z -= 1
            z += 1    
    
            
def process_pred(pred):
    """
    Short form of the process a new predicted goes through
    Gets the position of all boxes, relative to entire image.
    Merges those that are in the same tile so that one box encompasses both
    """
    
    boxes = true_position(pred)
    merged = merge_all(boxes)
    return merged 

def IOU(box1, box2): 
    """
    Calculate the Intersection Over Union
    """
    
    x1,y1,w1,h1 = box1[0],box1[1],box1[2],box1[3]
    x2,y2,w2,h2 = box2[0],box2[1],box2[2],box2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    
    I = w_I * h_I
    U = w1 * h1 + w2 * h2 - I

    return I / U
 
def mse(img1, img2): 
    """
    Calculate the Mean Squared Error
    """
    
    if img1.shape != img2.shape: #images need to be same shape
        return -1
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

    
def true_position(pred):
    """
    Returns middle of box and dimensions relative to entire image
    """
    
    boxes = [] 
    for z in range(config.gridN**2):
        if pred[0][z][0] > config.filter_threshold:
            i = z//config.gridN
            j = z%config.gridN
            w = pred[0][z][3]*config.L
            h = pred[0][z][4]*config.L
            cx = pred[0][z][1]*config.f + i*config.f
            cy = pred[0][z][2]*config.f + j*config.f
            boxes.append([cx, cy, w, h]) 
    return boxes 


def box_to_corners(box):
    """
    Takes the center point and dimensions of a box and gives the corner coordinates
    """
    
    cx, cy, w, h = box
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2 
    return np.array([x1, y1, x2, y2])

def corners_to_box(corners):
    """
    Takes the corner coordinates and gives the center point and dimensions
    """
 
    x1, y1, x2, y2 = corners
    cx = (x1 + x2)/2
    cy = (y1 + y2)/2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return np.array([cx, cy, w, h])

def merge_boxes(box1, box2):
    """
    Merge the coordinates of two boxes so that one will encompasses both
    """
    
    c1 = box_to_corners(box1)
    c2 = box_to_corners(box2)
    x1 = min(c1[0], c2[0])
    y1 = min(c1[1], c2[1])
    x2 = max(c1[2], c2[2])
    y2 = max(c1[3], c2[3])
    return corners_to_box([x1,y1,x2,y2])

    
def identify_overlap(boxes):
    """
    WIP 
    Identify all boxes that overlap 
    """ 

    overlaps = []
    for i in range(len(boxes)):
        this_overlap = []
        for j in range(len(boxes)):
            if (i != j) and (IOU(boxes[i], boxes[j]) > config.mixing_iou):
                this_overlap.append(j)
        overlaps.append(this_overlap)
    return overlaps

    
def merge_all(boxes):
    """
    WIP 
    Merge any boxes that overlap 
    """
    
    overlaps = identify_overlap(boxes)
    merged = []
    for i in range(len(overlaps)):
        n = len(overlaps[i])
        if n == 0:
            merged.append(boxes[i])
        else:
            curr = boxes[i]
            for j in range(n):                
                curr = (merge_boxes(curr, boxes[overlaps[i][j]]))
            merged.append(curr)
    return merged
    
    

def extract(img, boxes): 
    """
    Isolate the predicted source from the image 
    """
    
    img = np.reshape(img, [config.L, config.L]) #assumes one image

    imgs = []
    for z in range(len(boxes)):
        cx, cy = boxes[z][0], boxes[z][1]
        w, h = boxes[z][2], boxes[z][3]
        
        thisimg = img[max(int(np.floor(cx - w/2)), 0): min(int(np.ceil(cx + w/2)), config.L), \
                      max(int(np.floor(cy - h/2)), 0): min(int(np.ceil(cy + h/2)), config.L)]
        imgs.append(thisimg)

    return imgs
    
    
def standardize(img, size = 8):
    """
    Resize the img to a set size, allows direct comparison    
    Not in use
    """
    return resize(img, (size,size), anti_aliasing = True, mode = 'constant') 

def compare_images(img1, img2):
    """
    Compares two images in two metrics, MSE and Structural similarity
    Not in use
    """
   
    img1 = standardize(img1)
    img2 = standardize(img2)
    m = mse(img1, img2) #0 implies every pixel is the same between images
    s = ssim(img1, img2) #1 implies the images are the same
    return m, s
    
    
def compare_time(t1first, t1last, t2first, t2last):
    """
    Compares the time ranges of two boxes. 
    If they are within a set range, return true.
    """
    dif = 16
    if t2last < (t1first - dif) or t2first > (t1last + dif): 
        return False
    else:
        return True  
        
def compare_freq(f1low, f1high, f2low, f2high):
    """
    Compares the frequency ranges of two boxes. 
    If they are within a set range, return true.
    """
    dif = 2
    if f2high < (f1low - dif) or f2low > (f1high + dif): 
        return False
    else:
        return True         