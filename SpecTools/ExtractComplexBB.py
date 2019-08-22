#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from scipy.signal import lfilter, remez, freqz, spectrogram, find_peaks, filtfilt
import matplotlib.pyplot as plt
from skimage.transform import radon

parser = argparse.ArgumentParser()
parser.add_argument("h5", help="HDF5 File to parse.")
parser.add_argument("raw", help="Raw data file.")
parser.add_argument("--ind", help="Detection index", nargs='+', type=int)
parser.add_argument("--plt", help="Activate Plotting", action="store_true")
args = parser.parse_args()

h5 = h5py.File(args.h5, 'r+')

# System Parameters
nChan = len(h5['freqs'])
integration = 100
sample_size = 4 # complex int16

if 'features' in h5:
    del h5['features']

features = h5.create_dataset("features", (13,h5['merged_detections'].shape[1]),  dtype="float32")

cluster1 = [114,115,116,117,128,205]
cluster2 = [69,93,94,95,168,169,180,191,209]
cluster3 = [193,204]
cluster4 = [11,17,56,241,243,244,251,252,282]
cluster5 = [35,57,139,242]

detections = np.arange(0, h5['merged_detections'].shape[1])
if ( args.ind ):
    detections = list(map(int, args.ind))   
    
#for i in np.arange(0, h5['merged_detections'].shape[1]):
for i in detections:


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
        #h = remez(65, [0, 0.2, 0.3, 0.5], [1, 0], [1, 1])
        #x = lfilter(h, 1, x)

        if ( args.plt ):
            plt.figure()
            plt.title('Complex values: Index %d' %(i))
            plt.plot(np.real(x))
            plt.plot(np.imag(x))
            plt.draw()
    
            plt.figure()
            plt.title('Amplitude: Index %d' %(i))
            plt.plot(np.abs(x))
            plt.draw()
    
            plt.figure()
            plt.title('Phase: Index %d' %(i))
            plt.plot(np.angle(x))
            plt.draw()

        sxx = spectrogram(x)
        sxx = np.fft.fftshift(sxx[2], axes=0)

        # Compute centroid
        print(np.mean(sxx, axis=1).shape)
        centroid = np.sum(np.linspace(-0.5, 0.5, sxx.shape[0])*np.mean(sxx, axis=1))/np.sum(np.mean(sxx, axis=1))
        print(centroid)

        x *= np.exp(-2.0j*np.pi*centroid*np.arange(len(x)))
        x -= np.mean(x)

        sxx = spectrogram(x)
        sxx = np.fft.fftshift(sxx[2], axes=0)


#        theta = np.linspace(0., 180., max(sxx.shape), endpoint=False)
        #sinogram = radon(sxx, theta=[0, 90, 180])

        f = np.sum(sxx, axis=1)
        f[128] = (f[127]+f[129])/2.
        fm = 10.*np.log10(np.mean(f))
        if len(f) > 10:
            f = filtfilt(np.ones(3)/3., 1, 10.*np.log10(f))
        else:
            f = 10.*np.log10(f)

        peaks, props = find_peaks(f, prominence=3)
        print(peaks)
        fpeaks = len(peaks)


        if fpeaks > 1:
            promindex  = np.argmax(props['prominences'])
            prominence = props['prominences'][promindex]
            width      = props['right_bases'][promindex] - props['left_bases'][promindex]
            mean_fpk_spacing = np.mean(peaks[1:] - peaks[:-1])
        else:
            mean_fpk_spacing = 0
            prominence = 0
            width = 0   

        xi = np.arange(0, len(f))
        if ( args.plt ):
            plt.figure()
            plt.title('%d Frequency Peaks' %(len(peaks)))
            plt.plot(xi, f)
            plt.plot(xi[peaks], f[peaks], 'x')
            plt.draw()

        t = np.sum(sxx, axis=0)
        if len(t) > 10:
            t = filtfilt(np.ones(3)/3., 1, 10.*np.log10(t))
        else:
            t = 10.*np.log10(t)

        peaks, props = find_peaks(t, prominence=3)
        print(peaks)
        tpeaks = len(peaks)
        print(props)
        if tpeaks > 1:
            mean_tpk_spacing = np.mean(peaks[1:] - peaks[:-1])
        else:
            mean_tpk_spacing = 0

        tcentroid = np.sum(np.arange(0, sxx.shape[1])*np.mean(sxx, axis=0))/np.sum(np.mean(sxx, axis=0))
        tcentroid /= sxx.shape[1]
        print(sxx.shape[1])

        xi = np.arange(0, len(t))
        if ( args.plt ):
            plt.figure()
            plt.title('%d Time Peaks' %(len(peaks)))
            plt.plot(xi, t)
            plt.plot(xi[peaks], t[peaks], 'x')
            plt.draw()

            #plt.figure()
            #plt.plot(sinogram[:,0], label='0')
            #plt.plot(sinogram[:,1], label='90')
            #plt.plot(np.arange(-127, 129)+sinogram.shape[0]/2, f, '--', label='FFT')
            #plt.plot(np.sum(sxx, axis=0), '--', label='Total Power')
            #plt.legend()
            #plt.grid()
            #plt.draw()

            #plt.figure()
            #plt.imshow(sinogram,
            #   extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
            #plt.draw()
    
            plt.figure()
            plt.title('Index %d' %(i))
            plt.imshow(10.*np.log10(sxx))
            plt.draw()


        x -= np.mean(x)
        #plt.figure()
        #   plt.specgram(x)
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
              10.*np.log10(c21),
              fpeaks,
              tpeaks,
              prominence,
              mean_fpk_spacing,
              mean_tpk_spacing,
              tcentroid,
              width
              ])
        print(feat)
        features[:,i] = feat
        print(i, features[:,i])

        if ( args.plt ):
            plt.show()
           # print "\n\n"

h5.close()