#!/usr/bin/env python3
#
# Program to produce the power spectra from complex time series.
#
# Stephen Harrison
# NRC Herzberg
#

import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="File to parse.")
parser.add_argument("-n", "--nfft", type=int, help="Number of FFT channels.")
parser.add_argument("-m", "--navg", type=int, help="Number of FFTs to average.")
parser.add_argument("-o", "--output", help="Output file name.")
parser.add_argument("--fc", type=float, default=0, help="RF Center Frequency in Hz")
parser.add_argument("--fs", type=float, default=1, help="RF Sample Rate in Hz")
parser.add_argument("--t0", type=float, default=0, help="UTC start time.")
parser.add_argument('--trim', type=float, default=1000, help="PSD time samples to trim to.")
args = parser.parse_args()

sz = args.nfft*args.navg
with open(args.filename, 'rb') as fi:
	with h5py.File(args.output, "w") as fo:

		# Create vector of real RF frequencies.
		fo.create_dataset("freqs", (args.nfft,), dtype="float32")[:] = np.linspace(-args.fs/2, args.fs/2, args.nfft, endpoint=False)+args.fc

		# Create PSD dataset which we will resize.
		d = fo.create_dataset("psd", (args.nfft,0), maxshape=(args.nfft,None), dtype="float32")

		# Integrate and dump by reading from the input file.
		# Resize the dataset each time.
		x = np.fromfile(fi, dtype=np.int16, count=sz*2).astype(np.float32).view(np.complex64)
		while len(x) == sz:
			if d.shape[1] < args.trim:
				accum = np.zeros(args.nfft)
				for i in np.arange(0,sz,args.nfft):
					fft = np.fft.fftshift(np.fft.fft(x[i:i+args.nfft]*np.hanning(args.nfft)))
					accum += np.real(fft*np.conjugate(fft))
				d.resize(d.shape[1]+1, 1)
				d[:,d.shape[1]-1] = accum
				print('.', end='', flush=True)
				x = np.fromfile(fi, dtype=np.int16, count=sz*2).astype(np.float32).view(np.complex64)
			else:
				break

		# Add timestamps
		fo.create_dataset('times', (d.shape[1],), dtype="float64")[:] = np.linspace(0, d.shape[1]*sz/args.fs, d.shape[1], endpoint=False)+args.t0

		# Procedure to fit out the USRP bandpass
		y = np.mean(d, axis=1)
		x = np.arange(len(y))
		x_fit = np.copy(x)

		for i in np.arange(500):

			pf = np.polyfit(x_fit, y[x_fit], deg=18)
			p = np.polyval(pf, x_fit)
			diff = y[x_fit] / p
			rm = np.argmax(diff)
			x_fit = np.delete(x_fit, rm)

		p = np.polyval(pf, x)
		d[:,:] /= np.repeat(np.expand_dims(p, 1), d.shape[1], axis=1)

print()

##
## END OF CODE
##
