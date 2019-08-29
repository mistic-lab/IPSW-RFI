#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from scipy.signal import lfilter, remez, freqz, spectrogram
import matplotlib.pyplot as plt
import os

# parser = argparse.ArgumentParser()
# parser.add_argument("h5", help="HDF5 File to parse.")
# parser.add_argument("raw", help="Raw data file.")
# args = parser.parse_args()

fileroots = [
'/media/nsbruce/Backup Plus/460MHz/1565289740',  '/media/nsbruce/Backup Plus/460MHz/1565292314',  '/media/nsbruce/Backup Plus/460MHz/1565294703',  '/media/nsbruce/Backup Plus/460MHz/1565297086',
'/media/nsbruce/Backup Plus/460MHz/1565290032',  '/media/nsbruce/Backup Plus/460MHz/1565292618',  '/media/nsbruce/Backup Plus/460MHz/1565294998',  '/media/nsbruce/Backup Plus/460MHz/1565297386',
'/media/nsbruce/Backup Plus/460MHz/1565290425',  '/media/nsbruce/Backup Plus/460MHz/1565292921',  '/media/nsbruce/Backup Plus/460MHz/1565295293',  '/media/nsbruce/Backup Plus/460MHz/1565297744',
'/media/nsbruce/Backup Plus/460MHz/1565290811',  '/media/nsbruce/Backup Plus/460MHz/1565293225',  '/media/nsbruce/Backup Plus/460MHz/1565295592',  '/media/nsbruce/Backup Plus/460MHz/1565298142',
'/media/nsbruce/Backup Plus/460MHz/1565291105',  '/media/nsbruce/Backup Plus/460MHz/1565293518',  '/media/nsbruce/Backup Plus/460MHz/1565295890',  '/media/nsbruce/Backup Plus/460MHz/1565298448',
'/media/nsbruce/Backup Plus/460MHz/1565291410',  '/media/nsbruce/Backup Plus/460MHz/1565293815',  '/media/nsbruce/Backup Plus/460MHz/1565296189',  '/media/nsbruce/Backup Plus/460MHz/1565298746',
'/media/nsbruce/Backup Plus/460MHz/1565291711',  '/media/nsbruce/Backup Plus/460MHz/1565294109',  '/media/nsbruce/Backup Plus/460MHz/1565296488',
'/media/nsbruce/Backup Plus/460MHz/1565292014',  '/media/nsbruce/Backup Plus/460MHz/1565294405',  '/media/nsbruce/Backup Plus/460MHz/1565296786'
]

for fr in fileroots:
    h5Name = fr+'.h5'
    rawName = fr+'.dat'

    print('Opening {}'.format(h5Name))
    with h5py.File(h5Name, 'r') as h5:

        print('Opening {}'.format(rawName))
        with open(rawName, 'rb') as f:

            # System Parameters
            nChan = len(h5['freqs'])
            integration = 100
            sample_size = 4 # complex int16

            print("Num of merged detections to extract: {}".format(h5['merged_detections'].shape[1]))

            for i in range(h5['merged_detections'].shape[1]): #[118]: #np.arange(0, h5['merged_detections'].shape[1]):
                
                print("Extracting {}/{}".format(i,h5['merged_detections'].shape[1]))
                
                # Compute the parameters for extraction
                x1, y1, x2, y2 = h5['merged_detections'][:,i]

                # Start time is first X coord
                t0 = max(int(np.round(x1)), 0)
                print('Start time (integrations): %d' %(t0))

                # Total time is x2-x1
                # Limit the max time extractions are done for to 2s.
                t = min(int(np.round(x2-x1)), 40)
                print('Total time (integrations): %d' %(t))

                if t == 0:
                    continue

                # Center frequency chan is average of y1, y2
                fc = (y1+y2)/2.
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

                print('Seeking to %d bytes,' %(seek))
                f.seek(seek, 0)
                print('Extracting %d samples,' %(count/2))
                d = np.fromfile(f, dtype=np.int16, count=int(count)).astype(np.float32).view(np.complex64)
                
                print('Mixing to %f,' %(cFreq)),
                d *= np.exp(-2.0j*np.pi*cFreq*np.arange(len(d)))

                # Make a halfband filter. 
                h = remez(65, [0, 0.2, 0.3, 0.5], [1, 0], [1, 1])

                # Apply it a bunch of times. The sample rate is now 2MSPS/2**N
                # print('Reducing bandwidth by %d octaves,' %(octaves)),
                for j in np.arange(octaves):
                    d = lfilter(h, 1, d)[::2]

                x = d

                sxx = spectrogram(x)
                sxx = np.fft.fftshift(sxx[2], axes=0)

                # Compute centroid
                print(np.mean(sxx, axis=1).shape)
                centroid = np.sum(np.linspace(-0.5, 0.5, sxx.shape[0])*np.mean(sxx, axis=1))/np.sum(np.mean(sxx, axis=1))
                print(centroid)

                x *= np.exp(-2.0j*np.pi*centroid*np.arange(len(x)))
                x -= np.mean(x)

                f_out = os.path.basename(rawName)

                x.astype(np.complex64).tofile('/media/nsbruce/Backup Plus/460MHz/waveforms/'+f_out + '.%d' %(i) + '.c64')

