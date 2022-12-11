
from meldataset import mel_spectrogram, load_wav
import argparse
import numpy as np
import os
from librosa.util import normalize
import librosa
import matplotlib.pyplot as plt

import torch

MAX_WAV_VALUE = 32768.0

def main(args):
  
    num_mels = 80
    n_fft = 1024
    hop_size = 256
    win_size = 1024

    sampling_rate = 22050

    fmin = 0
    fmax = 8000
    
    audio, sampling_rate = load_wav(args.wavs_path)
    audio = audio / MAX_WAV_VALUE
    
    audio = normalize(audio) * 0.95
    
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    
    """
    audio, _ = librosa.load(args.wavs_path)
    audio = audio.astype(np.float32)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    """
    
    mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
    mel = mel.squeeze()
    mel = mel.numpy()
    
    np.save(
        os.path.join('sample_mel', 'mel.npy'),
        mel.T,
    )
    
    c = plt.imshow(mel, origin ='lower')
    plt.title('HiFi-GAN')
    plt.savefig(os.path.join('sample_mel', 'mel.jpg'))
    print(mel.T)
    print(mel.T.shape)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavs_path', required=True, help="path to wav files")
    args = parser.parse_args()
    
    main(args)
    
 # python create_spec_from_wav.py --wavs_path /mnt/hdd1/adibian/multispeaker_data/test_wav/00048-001423-1.wav
 
