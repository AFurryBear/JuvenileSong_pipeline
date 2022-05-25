import argparse
import bandfilter_yirong as bandfilter
import numpy as np
import glob
import os
import combine_denoise as comdnr

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--lowfreq', type=int, default=1000)
parser.add_argument('--highfreq', type=int, default=12000)
parser.add_argument('--width_kernel', type=int, default=500)
args = parser.parse_args()

wav_files = sorted(glob.glob(os.path.join(args.inPath, '*.wav')))
for start in range(0,len(wav_files),25):
    ## combine several together
    outFile = os.path.join(args.inPath + '_combined', '%05d_%05d.wav' % (start, end))
    end = np.min([len(wav_files), start+25])
    comdnr.combine(wav_files[start:end], outFile)
    ## apply band filter
    inFile = outFile
    outFile = os.path.join(args.inPath + '_combined_filtered', '%05d_%05d_filtered.wav' % (start, end))
    bandfilter.bandfilter_main(inFile, outFile)
    ## denoise
    inFile = outFile
    outFile = os.path.join(args.inPath + '_combined_filtered', '%05d_%05d_filtered_denoised.wav' % (start, end))
    comdnr.denoise(inFile,outFile)

