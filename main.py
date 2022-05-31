import argparse
import bandfilter_yirong as bandfilter
import numpy as np
import glob
import os
import combine_denoise as comdnr

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--outPath', type=str, default=None)
parser.add_argument('--lowfreq', type=int, default=1000)
parser.add_argument('--highfreq', type=int, default=12000)
parser.add_argument('--width_kernel', type=int, default=500)
args = parser.parse_args()

wav_files = sorted(glob.glob(os.path.join(args.inPath, '*.wav')))
## combine several together
folder = os.path.split(args.inPath)[1]
outFile = os.path.join(args.outPath, folder+'.wav')
comdnr.combine(wav_files, outFile)
## apply band filter
inFile = outFile
outFile = os.path.join(args.outPath, folder+'_filtered.wav')
bandfilter.bandfilter_main(inFile, outFile)
## denoise
inFile = outFile
outFile = os.path.join(args.outPath, folder+'_filtered_denoised.wav')
comdnr.denoise(inFile,outFile)

