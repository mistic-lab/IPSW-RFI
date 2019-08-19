#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.patches as patch

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="File to parse.")
parser.add_argument("--draw_boxes", action='store_true', help="Draw Detection Bounding Boxes.")
args = parser.parse_args()

h5 = h5py.File(args.filename, 'r')

extents = [np.min(h5['times']), np.max(h5['times']), np.max(h5['freqs'])/1e6, np.min(h5['freqs'])/1e6]

im = plt.imshow(10.*np.log10(h5['psd']), aspect='auto', interpolation='none', extent=extents)
cbar = plt.colorbar()
cbar.set_label('dB')
plt.clim(0,None)
plt.xlabel('Time (s)')
plt.ylabel('Freq (MHz)')

if args.draw_boxes:

    ax = plt.gca()
    for i in np.arange(h5['merged_detections'].shape[1]):
        x1, y1, x2, y2 = np.round(h5['merged_detections'][:,i])
        x1 = np.clip(x1, 0, h5['times'].shape[0]-1)
        x2 = np.clip(x2, 0, h5['times'].shape[0]-1)
        y1 = np.clip(y1, 0, h5['freqs'].shape[0]-1)
        y2 = np.clip(y2, 0, h5['freqs'].shape[0]-1)
        t1 = h5['times'][int(x1)]
        t2 = h5['times'][int(x2)]
        f1 = h5['freqs'][int(y1)]/1e6
        f2 = h5['freqs'][int(y2)]/1e6
        ax.add_patch(patch.Rectangle((t1, f1), t2-t1, f2-f1, ec='g', fc='none'))

im.set_extent(extents)
plt.tight_layout()
plt.show()

