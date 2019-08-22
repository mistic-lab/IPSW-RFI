#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("h5", help="HDF5 File to parse.")
parser.add_argument("BBdir", help="Directory of basebands")
# parser.add_argument("npy", help="output numpy file")
args = parser.parse_args()


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


with h5py.File(args.h5, 'r') as parent:
    
    # root, dir, parentFileName = os.walk(args.h5)
    parentName, ext = str(args.h5).rsplit('.')
    print("Parent name: {}".format(parentName))
    if ext != 'h5':
        raise Exception("Wrong parent file type")
    
    bbFiles = getFilesByExt(args.BBdir, parentName)

    indexes = []

    for f in bbFiles:
        # print(f)
        head, tail = os.path.split(f)
        parentName2, ext, index, c64 = tail.rsplit(".")
        indexes.append(int(index))
        if parentName != parentName2:
            raise Exception("Y'all messed up! Ha Ha!")
    
    # indexes now contains all the indexes for the available baseband signals
    print(parent['features'].shape)
    output=np.ndarray((parent['features'].shape[0], max(indexes)+1))

    for i in indexes:
        output[:,i] = parent['features'][:,i]

    print(output.shape)
    np.savetxt(parentName+'_features.txt', output)