import sys 
import numpy as np

from Config import Config
from Net import Network
from DataHandle import *
import h5py

config = Config()

def RectIntersect(R1, R2):

    if (R1[0] < R2[2]) and (R1[2] > R2[0]) and (R1[1] < R2[3]) and (R1[3] > R2[1]):
        return True
    return False



def main(file_name):
    
    #try:
    #    data = np.load('Data/' + str(file_name) + '.npy')
    #    data = np.clip(data,0,5)/5
    #except:
    #    print("Invalid File Name")
    #    return
    h5 = h5py.File(file_name, "r+")

    data = np.transpose(h5['psd'])
    data = 10.*np.log10(data)
    data = np.clip(data, 0, 5)/5.

    if 'detections' in h5:
        del h5['detections']

    d = h5.create_dataset("detections", (4,0), maxshape=(4,None), dtype="float32")

    
    net = Network()
    net.load()
 
    #f = open('Detections_' + str(file_name) + '.txt', 'a')
    
    for x in np.arange(0, np.shape(data)[1]-config.L,config.L):
        c = data[:, x:x+config.L]

        buffer = ImgBuffer()
        for n in range(0, np.shape(c)[0]-config.L, config.f):
    
            currIm = np.reshape(c[n:n+config.L,:], [1, config.L, config.L])
            pred = net.predict(currIm) 

            boxes = process_pred(pred)

            buffer.update_timer()
            buffer.process_new(boxes, n, x)
            buffer.process_existing()

        buffer.clear_buffer()

        currsz = d.shape[1]
        d.resize(currsz+len(buffer.final_array), 1)
        for i in np.arange(0, len(buffer.final_array)):
            d[:,i+currsz] = buffer.final_array[i]


        #np.savetxt(f, buffer.final_array, delimiter=',', newline='\n')
    #f.close()  

    boxes = np.transpose(h5['detections']).tolist()

    if 'merged_detections' in h5:
        del h5['merged_detections']

    newboxes = []
    while len(boxes):
        new = [boxes.pop()]
        for testbox in new:
            addIdx = []
            for i, box in enumerate(boxes):
                if RectIntersect(testbox, box):
                    new.append(box)
                    addIdx.append(i)
            for i in sorted(addIdx, reverse=True):
                del boxes[i]
        new = np.asarray(new)
        print(new.shape)
        newboxes.append([np.min(new[:,0]), np.min(new[:,1]), np.max(new[:,2]), np.max(new[:,3])])
    newboxes = np.transpose(np.asarray(newboxes))
    print(newboxes.shape)
    if newboxes.shape[0] != 0:
        merged = h5.create_dataset("merged_detections", newboxes.shape, dtype="float32")
        merged[:,:] = newboxes
    else:
        del h5['merged_detections']

    h5.close()  
        
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Need one argument for file name")