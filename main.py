
import os
import shutil
import random
import math
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import ffmpeg
import librosa
import webrtcvad
import librosa.display
import webrtcvad
baseName ='20210721_113650'
wav_file = 'voice/'+baseName+'.wav'
sample_rate, samples = wavfile.read(wav_file)
print('sample rate : {}, samples.shape : {}'.format(sample_rate, samples.shape))

def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate /1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                           fs=sample_rate,
                                           window='hann',
                                           nperseg=nperseg,
                                           noverlap=noverlap,
                                           detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + wav_file)
ax1.set_ylabel('Amplitude')
ax1.plot(samples)

freqs, times, spectrogram = log_specgram(samples, sample_rate)
ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + wav_file)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')
mean = np.mean(spectrogram, axis=0)
std = np.std(spectrogram, axis=0)
spectrogram = (spectrogram - mean) / std
librosa_samples, librosa_sample_rate = librosa.load(wav_file)
S = librosa.feature.melspectrogram(librosa_samples, sr=librosa_sample_rate, n_mels=128, fmax=8000)


log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.title('Mel Power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.show()

# ///////////////////////// cut audio

vad = webrtcvad.Vad()
# 1~3 까지 설정 가능, 높을수록 aggressive
vad.set_mode(1)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

#  tao casc frame tu frame ban ddau (10ms) 1 frame
def frame_generator(frame_duration_ms, audio, sample_rate):
    frames = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2) #360
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0 # 0.01
    while offset + n < len(audio):
        frames.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n

    return frames

#  lay ra wakeup "hi bixby"
def auto_vad(vad, samples, sample_rate, startpoint=0):
    not_speech = []
    samples = samples[startpoint:]
    frame_duration_ms = 10
    frames = frame_generator(frame_duration_ms, samples, sample_rate)
    n_frame = len(frames)
   # moi frame = 2 * sample rate / 100
    start_idx = 0
    for idx, frame in enumerate(frames):
        if not vad.is_speech(frame.bytes, sample_rate):
            not_speech.append(idx)
        else:
            if start_idx == 0:
                start_idx = idx

    prior = 0
    cutted_samples = []
    cutted_start = 0
    cutted_end = 0
    for i in not_speech:
        if i - prior > 2:
            start = int((float(prior) / n_frame) * sample_rate) #01ms 1 frame -> 160 ma
            end = int((float(i) / n_frame) * sample_rate)
            if len(cutted_samples) == 0:
                cutted_samples = samples[start:end]
            else:
                cutted_samples = np.append(cutted_samples, samples[start:end])
                cutted_end = end
        prior = i
    cutted_start = int((float(start_idx) / n_frame) * sample_rate)
    return samples[cutted_start:cutted_end], cutted_start + startpoint, cutted_end + startpoint
wakeup_samples,  wakeup_start, wakeup_end = auto_vad(vad, samples, sample_rate)
# ipd.Audio(wakeup_samples, rate=sample_rate)
# obj2write = wave.open( wav_wakeup_file, 'wb')
# obj2write.setnchannels( 1 )
# obj2write.setsampwidth( 2 )
# obj2write.setframerate( 16000 )
# obj2write.writeframes(wakeup_samples)
# obj2write.close()
# vad.set_mode(3)


_,  command_start, command_end = auto_vad(vad, samples, sample_rate, wakeup_end+(sample_rate*1)) #1초 뒤

while(True):
    _, next_start, next_end = auto_vad(vad, samples, sample_rate, command_end)
    print(command_start, command_end, next_start, next_end)
    wait_time = next_start - command_end
    duration = next_end - next_start
    print(wait_time, duration)
    if wait_time + duration  == 0:
        break
    elif wait_time <= 500 :
        command_end = next_end
        pass
    else:
        break

command_samples = samples[command_start:command_end]
# ipd.Audio(command_samples, rate=sample_rate)
# obj2write = wave.open( wav_command_file, 'wb')
# obj2write.setnchannels( 1 )
# obj2write.setsampwidth( 2 )
# obj2write.setframerate( 16000 )
# obj2write.writeframes(command_samples)
# obj2write.close()