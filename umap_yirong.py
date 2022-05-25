import numpy as np
import sklearn
import matplotlib.pyplot as plt
import librosa.feature
import librosa.display
from matplotlib.colors import ListedColormap
import colorcet as cc
import sklearn.metrics
import scipy.stats
import hdbscan
import umap
import pandas as pd
import scipy.io.wavfile
from noisereduce.noisereducev1 import reduce_noise
import glob
import os
import das_unsupervised.spec_utils
import ipywidgets as widgets
import argparse

plt.style.use('ncb.mplstyle')


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--inPath', type=str, default=None)
parser.add_argument('--fileType', type=int, default='wav')

args = parser.parse_args()

file_list = glob.glob(args.inPath+'*.'+args.fileType)
specs = []
amps = []
types = []
durations = []

# these parameters allow you to select a specific frequency range and to threshold noise
# optimize them by visually inspecting the plotted spectrograms below this cell
freq_min_idx = 4
freq_max_idx = -1
freq_step_idx = 2
amplitude_thres = 2

for filename in file_list:
    print(filename)

    # load recording and annotations
    fs, x = scipy.io.wavfile.read(filename)
    x = x.astype(float)
    print(f"{len(x)} samples at {fs} Hz.")

    annotations = pd.read_csv(f'{os.path.splitext(filename)[0]}_annotations.csv')
    syllables = annotations.loc[annotations.name == 'syllable']
    print(f"{len(syllables)} annotated syllables.")

    hop_length = int(2 * fs / 1000)
    win_length = int(10 * fs / 1000 * 2)
    specgram = librosa.feature.melspectrogram(x, sr=fs, n_fft=win_length, hop_length=hop_length, power=2)
    # specgram = specgram[np.where(specgram[:,0]!=0)[0],:]
    sm = np.median(specgram, axis=1)

    onsets = syllables['start_seconds'] * fs  # samples
    offsets = syllables['stop_seconds'] * fs  # samples
    for cnt, (onsets, offsets) in enumerate(zip(onsets, offsets)):
        tmp = np.log2(specgram[:, int(onsets / hop_length):int(offsets / hop_length)] / sm[:, np.newaxis])
        amps.append(np.var(tmp))
        tmp = tmp[freq_min_idx:freq_max_idx:freq_step_idx, :]
        tmp = np.clip(tmp - amplitude_thres, 0, np.inf)
        specs.append(tmp)
        durations.append((offsets - onsets) / fs)

    print(f"Got {len(specs)} syllables.")
    print(onsets)

spec_rs = [das_unsupervised.spec_utils.log_resize_spec(spec, scaling_factor=8) for spec in specs]
max_len = np.max([spec.shape[1] for spec in spec_rs]) + 1
spec_rs = [das_unsupervised.spec_utils.pad_spec(spec, pad_length=max_len) for spec in spec_rs]

spec_flat = [spec.ravel() for spec in spec_rs]
spec_flat = np.stack(spec_flat, axis=1)

out = umap.UMAP(min_dist=0.1).fit_transform(spec_flat.T)
labels = hdbscan.HDBSCAN(
    min_samples=100,
    min_cluster_size=50,
).fit_predict(out)

arr = np.concatenate([out.T,labels.reshape((1,len(labels))),np.arange(1,len(labels)+1).reshape((1,len(labels)))],axis=0).T
np.savetxt('umap.csv',arr,delimiter=',')
devider = np.tile(np.max(spec_flat,axis=1),len(spec_flat[0])).reshape(spec_flat.shape)
np.savetxt('sylmtx_pred.csv',255*spec_flat/np.where(devider==0,1e-3,devider),delimiter=',')