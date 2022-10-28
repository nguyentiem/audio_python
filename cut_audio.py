import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.fft import fft, fftfreq
import string



def auto_cut(len_frame,threshold,audio,rangemin, rangemax):
    audio = audio[audio!=0]
    len  = audio.shape[0]
    # print(len)
    step = int(len_frame/10)
    index =0
    start =0
    end =0
    cnt_no_sound = 0;
    no_sound =0;
    tbno =0;
    isStart =0;
    while (index+len_frame)<len:
        arr = audio[index:index+len_frame]
        head = audio[index:(index+len_frame-step)]
        tail = audio[(index+len_frame-step):index+len_frame]
        tb1 = sum(np.abs(head))/(len_frame-step)
        tb2 = sum(np.abs(tail))/step
        tb = sum(arr)/len_frame
        if (tb2>tb1*threshold or (tb2>threshold*tbno and tbno>0)) and isStart==0: # start
          isStart=1;
          start = int(index+(len_frame/2))
        elif ((tbno>0 and (tb2>=rangemin*tbno and tb2<=rangemax*tbno)) or (tb1>=tb2*threshold)) and isStart==1:    # end
            end =  index+len_frame
            break
        elif (tb2>=tb1 and tb2<=threshold*tb1)or (tb1>=tb2 and tb1<=threshold*tb2) :
            cnt_no_sound+=1
            no_sound+=0.9*tb1+0.1*tb2
            tbno = no_sound/cnt_no_sound
        index+=step
    print(start)
    print(end)
    return start,end

def auto_cut2(len_frame,audio,num,threshold,compare):
    len = audio.shape[0]
    index = 0
    start = 0
    end = 0
    step = int(len_frame / 3)
    idmh = int(len_frame/2)-int(num/2)
    idmt=  int(len_frame/2)+int(num/2)
    idt= len_frame-num
    tbtong = 0;
    while (index + len_frame) < len:
        head = audio[index:index+num]
        mid =audio[index+idmh:index+idmt]
        tail = audio[index+idt:index+len_frame]
        tbh = max(head)
        tbm = max(mid)
        tbt =max(tail)
        tb_empty = 0
        if (tbt>(tbh)*threshold or (tbt>tbm*threshold and tbt>tbh*threshold) or tbm>tbh*threshold) and tbt>=0.02 and start==0:
            start=int(index+idmt)
            tb_empty = sum(np.abs(audio[index:start]))/start
            tbtong = tbt
        elif ( tbtong/compare-tbt>tbt-(tb_empty*compare) or tbt<tb_empty*compare)  and start!=0:
            # print('end')
            end = index+len_frame
            break
        elif start!=0 and index>start:
            tbtong = sum(np.abs(audio[start:index]))/(index-start)

        index += step
    if end==0:
        end =len
    print(start)
    print(end)
    return start,end

def auto_cut3(len_frame,audio,num,threshold,len=2000):
    la = audio.shape[0]
    index = 0
    start = 0
    idmh = int(len_frame / 2) - int(num / 2)
    idmt = int(len_frame / 2) + int(num / 2)
    step = int(len_frame/4)
    idt = len_frame - num
    while (index + len_frame) < la:
        head = audio[index:index+num]
        mid =audio[index+idmh:index+idmt]
        tail = audio[index+idt:index+len_frame]
        tbh = max(head)
        tbm = max(mid)
        tbt =max(tail)
        if (tbt>(tbh)*threshold or (tbt>tbm*threshold and tbt>tbh*threshold) or tbm>tbh*threshold) and tbt>=0.02 and start==0:
            start=int(index+idmt)
            if(start+len)<la:
                end = start+len
            else:
                end =la
            return start, end
        index+=step

def cut_save_empty(dir , file, dicDir):
        audio_file = dir+file
        audio_data, sr = librosa.load(audio_file, sr= 8000, mono=True)
        # audio_data = audio_data[audio_data!=0]
        print(audio_data.shape)
        start,end = auto_cut3(500,audio_data,50,4)
        newAudio = audio_data[start:end]
        write(dicDir,8000,newAudio)
        # plt.plot(audio_data)


    # newAudio = audio_data[start:end]
    # write(dicDir, 8000, newAudio)
def getMel(audio_data,sr):
    melspectrum = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length= 512, window='hann', n_mels=256)
    return melspectrum

def cut_empty(dir, file):
    audio_file = dir + file
    audio_data, sr = librosa.load(audio_file, sr=8000, mono=True)
    start, end = auto_cut3(500, audio_data, 50, 4)

    newAudio = audio_data[start:end]
    melspectrum= getMel(newAudio,sr)
    return melspectrum

def check(melspectrum,melspectrum1):
    melspectrum=melspectrum.T
    melspectrum1=melspectrum1.T
    r,c = melspectrum.shape
    r1,c1 = melspectrum1.shape
    if r!=4:
        return -1;
    if r>r1:
       return -1
    elif r<r1:
        return -1;
    else:
        melspectrum = np.where(melspectrum < 2, 0, melspectrum)
        melspectrum1 = np.where(melspectrum1 < 2, 0, melspectrum1)
        maxHead = np.max(melspectrum[:,:64],1)
        maxHeadIndex = np.argmax(melspectrum[:,:64],1)
        maxTail = np.max(melspectrum[:,64:],1)
        maxTailIndex = np.argmax(melspectrum[:, 64:], 1)
        maxHead1 = np.max(melspectrum1[:, :64], 1)
        maxHeadIndex1 = np.argmax(melspectrum1[:, :64], 1)
        maxTail1 = np.max(melspectrum1[:, 64:], 1)
        maxTailIndex1 = np.argmax(melspectrum1[:, 64:], 1)


        return 1


def check(melspectrum):
    melspectrum = melspectrum.T

    r, c = melspectrum.shape

    if r != 4:
        return -1
    else:
        melspectrum = np.where(melspectrum < 1, 0, melspectrum)
        maxHead = np.max(melspectrum[:, :64], 1)
        maxHeadIndex = np.argmax(melspectrum[:, :64], 1)
        maxTail = np.max(melspectrum[:, 64:], 1)
        maxTailIndex = np.argmax(melspectrum[:, 64:], 1)

        # print(maxHead)
        # print(maxHeadIndex)
        # print(maxTail)
        # print(maxTailIndex)
        if maxHeadIndex[0]!=50 or maxHeadIndex[1]!=50 or maxHeadIndex[2] !=50 or (maxHeadIndex[3]!=0 and maxHeadIndex[3]!=50) :
            print('sai do index head')
            return -1
        if (maxTailIndex[0]!=0 and maxTailIndex[0]<10 and maxTailIndex[0]>12)  or (maxTailIndex[1]<10 or maxTailIndex[1]>12) or (maxTailIndex[2]<10 or maxTailIndex[2]>12) or maxTailIndex[3]<10 or maxTailIndex[3]>12:
            print('sai do index tail')
            return -1
        if maxHead[0]>maxHead[1]or maxHead[2]>maxHead[0] or maxHead[3]>maxHead[2]:
            print('sai do thu tu head')
            return -1
        if maxTail[1]>maxTail[2] or maxTail[3]>maxTail[1] or maxTail[0]>maxTail[3]:
            print('sai sdo thu tu tail ')
            return -1

        return 1
# for i in range(1,13):
#     mel = cut_save_empty('voice/',str(i)+'.wav','traindata/bixby/'+'ting'+str(i+11)+'.wav')

# if check(mel) ==1 :
#     print('dung')
# else:
#     print('sai')
# plt.show()

# audio_data, sr = librosa.load('traindata/nonbixby/A.wav', sr= 8000, mono=True)
# melspectrum = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length=512, window='hann', n_mels=256)
# plt.plot(melspectrum)
# plt.show()


import numpy as np
import sounddevice as sd
import time
def cut(dir,file):
    audio_data, sr = librosa.load(dir+file, mono=True)
    print(sr)
    newAudio = audio_data[1500:20000]
    write('hi_bixby.wav', sr, newAudio)
    sd.play(newAudio, sr)
    time.sleep(newAudio.shape[0]*1000/sr)
    sd.stop()
    plt.plot(audio_data)
    plt.show()


cut('voice/', 'hi.wav')

