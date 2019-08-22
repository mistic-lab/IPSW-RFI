#!/usr/bin/env python3

import h5py
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib

def getFilesByExt(path, ext, verbose=False):
    files = []
    # r=root, d=directories, f = file
    for r, d, f in os.walk(path):
        for file in f:
            if ext in file:
                files.append(os.path.join(r, file))
    if verbose:
        for f in files:
            print(f)
        print("End of getFilesByExt")
    return files

def extractGridWide(inputFile, gridWidth, startIndex, getFreqs=True):
    with h5py.File(inputFile, 'r') as f:
        gridWide = f['psd'][startIndex:startIndex + gridWidth]
        freqs = f['freqs'][startIndex:startIndex + gridWidth]
    if getFreqs:
        return gridWide, freqs
    else:
        return gridWide

def main(args):
    inputFiles = getFilesByExt(args.d,'.h5')

    with h5py.File(args.o, "w") as fo:

        for file in [inputFiles[0]]:
            psd, freqs = extractGridWide(file, args.w, args.s)

            # Create vector of real RF frequencies. Only need to do it once
            fo.create_dataset("freqs", data=freqs)
            d = fo.create_dataset("psd", (len(freqs),0), maxshape=(len(freqs),None), dtype="float32")


        for file in inputFiles:
            print(file)
            psd = extractGridWide(file, args.w, args.s, getFreqs=False)
                        
            d.resize(d.shape[1]+psd.shape[1], 1)
            d[:,d.shape[1]-psd.shape[1]:] = psd
        
        #hacky hacky bad bad
        fo.create_dataset('janky_times', (d.shape[1],), dtype="float64")[:] = np.linspace(0, d.shape[1]*(50000*100)/50000000, d.shape[1], endpoint=False)+1565289740


    # with h5py.File(args.o, 'r') as fi:
    #     fig, ax = plt.subplots()
    #     ax.imshow(fi['psd'], aspect='auto')
    #     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='converts TBN file to hdf5 file', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-d", help="directory of input h5 files")
    parser.add_argument("-o", help="output h5 file")
    parser.add_argument("-s", type=int, help="start index")
    parser.add_argument("-w", type=int, help="grid width (channels)")
    args = parser.parse_args()
    main(args)
