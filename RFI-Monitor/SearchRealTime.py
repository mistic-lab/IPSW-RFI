import zmq
import time 
import os 
import sys

from Config import Config
from Net import Network
from DataHandle import *
config = Config()


def main():
    file_name = 'tmp'
    
    USRP_Host = '127.0.0.1'
    data_port = 5678
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://%s:%s' %(USRP_Host, data_port))
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    
    f = 460
    x = np.linspace(f-1.25, f+1.25, 2500).reshape(-1,1)

    thermalfloor = np.load('ThermalFloorModel.npy')
    relevant_floor = thermalfloor[2245:2245+32]
    net = Network()
    net.load()
    f = open('Detections_' + str(file_name) + '.txt', 'a')

    #fig, ax = plt.subplots(1,1)
    #plt.ion()
    
    imgbuffer = ImgBuffer()
    i = 0
    imgwindow = np.zeros([1,config.L, config.L])
    while True:
        #plt.cla()
        try: 
            md = socket.recv_json(flags=0)
            message = socket.recv(flags=zmq.NOBLOCK, copy=True, track=False)
            
            indata = np.frombuffer((message), dtype=md['dtype'])
            indata = indata.reshape(md['shape'])

            for k in range(config.L - 1):
                imgwindow[0,k,:] = imgwindow[0,k+1,:]
            imgwindow[0,-1] = np.clip(10.*np.log10(indata) - relevant_floor.T, 0, 5)/5
            
            pred = net.predict(imgwindow)
            boxes = process_pred(pred)
            
            imgbuffer.update_timer()
            imgbuffer.process_new(boxes, i, 2245)
            imgbuffer.process_existing()
            
            #ax.imshow(imgwindow[0], vmax = 1, vmin = 0)
            #plt.savefig(str(i))
            
        except zmq.ZMQError:
            time.sleep(.1)
        
        except KeyboardInterrupt:
            socket.close()
            context.term()
            sys.exit()
        i += 1    
    
    
    
if __name__ == "__main__":
    main()