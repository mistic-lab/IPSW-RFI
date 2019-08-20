#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from scipy.signal import lfilter, remez, freqz, spectrogram
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--h5", help="HDF5 File to parse.")
parser.add_argument("--raw", help="Raw data file.")
args = parser.parse_args()

h5 = h5py.File(args.h5, 'r+')

# System Parameters
nChan = len(h5['freqs'])
integration = 100
sample_size = 4 # complex int16

if 'features' in h5:
    del h5['features']

features = h5.create_dataset("features", (6,h5['merged_detections'].shape[1]),  dtype="float32")



for i in np.arange(0, h5['merged_detections'].shape[1]):

    # Compute the parameters for extraction
    x1, y1, x2, y2 = h5['merged_detections'][:,i]

    # Start time is first X coord
    t0 = max(int(np.round(x1)), 0)
    print('Start time (integrations): %d' %(t0))

    # Total time is x2-x1
    # Limit the max time extractions are done for to 2s.
    t = min(int(np.round(x2-x1)), 20)
    print('Total time (integrations): %d' %(t))

    if t == 0:
        continue

    # Center frequency chan is average of y1, y2
    fc = (y1+y2)/2. + y1 # I don't think it is.
    print('Center Freq (channels): %.2f' %(fc))

    # Bandwidth is y2-y1
    bwc = (y2-y1)
    print('Bandwidth (channels): %.2f' %(bwc))

    # Normalized Frequency Adjusted for channel offset.
    cFreq = (fc-nChan/2)/float(nChan) 
    print('Center Freq (normalized): %.2f' %(cFreq))

    # Normalized bandwidth
    bw = bwc/float(nChan)
    print('Bandwidth (normalized): %.2f' %(bw))

    # Number of octaves to reduce the bandwidth is from estimated bandwidth
    # relative to sample rate.
    octaves = np.ceil(np.log2(1./bw))-1

    # Seek to offset based on integration length, number of channels.
    seek = int(t0 * nChan * integration * sample_size)

    # Load count samples based on x2-x1
    count = int(t * nChan * integration * 2) # 2 for complex.


    print('Opening %s, ' %(args.raw)),
    with open(args.raw, 'rb') as f:
        print('Seeking to %d bytes,' %(seek)),
        f.seek(seek)
        print('Extracting %d samples,' %(count/2)),
        d = np.fromfile(f, dtype=np.int16, count=int(count)).astype(np.float32).view(np.complex64)
        
        print('Mixing to %f,' %(cFreq)),
        d *= np.exp(-2.0j*np.pi*cFreq*np.arange(len(d)))

        # Make a halfband filter. 
        h = remez(65, [0, 0.2, 0.3, 0.5], [1, 0], [1, 1])

        # Apply it a bunch of times. The sample rate is now 2MSPS/2**N
        print('Reducing bandwidth by %d octaves,' %(octaves)),
        for j in np.arange(octaves):
            d = lfilter(h, 1, d)[::2]

        # Make a final filter. 
        if 1.1*bw < 1:
            print('Reducing final bandwidth to %f.' %(bw)),
            h = remez(101, [0, bw/2, 1.1*bw/2, 0.5], [1, 0], [1, 1])
            d = lfilter(h, 1, d)
        print('')

#        d.astype(np.complex64).tofile('extracted/%05d.raw' %(r))
        #vec_len = 1024
        #for i in np.arange(0, len(d)-vec_len, vec_len):
        #x = d[i:i+vec_len]
        x = d
        x -= np.mean(x)
        #plt.figure()
        #plt.specgram(x)
        #plt.show()
        c21 = np.mean(np.abs(x)**2)
        c42 = np.mean(np.abs(x)**4)-np.abs(np.mean(x)**2)**2-2*np.mean(np.abs(x)**2)**2
        c63 = np.mean(np.abs(x)**6)-9*np.mean(np.abs(x)**4)*np.mean(np.abs(x)**2)+12*np.abs(np.mean(x)**2)**2**np.mean(np.abs(x)**2)+12*np.mean(np.abs(x)**2)**3
        c42 /= c21**2
        c63 /= c21**3
        feat = np.asarray([np.interp(fc, np.arange(len(h5['freqs'])), h5['freqs'][:]), 
              bwc*(h5['freqs'][1]-h5['freqs'][0]), 
              c42, 
              c63, 
              (x2-x1)/10.,
              10.*np.log10(c21)])
        print(feat)
        features[:,i] = feat
        print(i, features[:,i])

           # print "\n\n"

h5.close()