import glob
import os
from scipy.io import wavfile
import noisereduce as nr
import os
import json
import numpy as np
from pydub import AudioSegment
import glob
# combine data
def combine(input_files,output_file):
    output_music = AudioSegment.empty()
    for f in input_files:
        input_music = AudioSegment.from_wav(f)
        output_music += input_music
    output_music.export(output_file, format="wav")

def denoise(input_file,output_file):
    rate, data = wavfile.read(input_file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    wavfile.write(output_file, rate, reduced_noise)



wav_files = sorted(glob.glob('/Volumes/Seagate Expansion Drive/dph069_wav_filtered/*.wav'))
for start in range(0,len(wav_files),25):
    end = np.min([len(wav_files), start+25])
    combine(wav_files[start:end],'/Volumes/Seagate Expansion Drive/dph069_wav_filtered_combined/%05d_%05d.wav'%(start,end))
for file in glob.glob('../dataj7054_f6552_combined/chan6/*/*.wav'):
    rate, data = wavfile.read(file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # perform noise reduction
    folder, fname = os.path.split(file)

    wavfile.write(os.path.join('../dataj7054_f6552_combined/chan6_filter_denoised',
                               os.path.split(folder)[-1] + 'filter_denoised.wav'), rate, reduced_noise)