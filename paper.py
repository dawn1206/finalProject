import librosa
import os
import numpy as np
import preprocess
import ctypes
import time

def process(y,sr):
    # pre_emphasis1 = y.copy()
    # ptr1 = pre_emphasis1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # cDLLcontroller = ctypes.cdll.LoadLibrary(r'.\Dll2.dll')
    # cDLLcontroller.audioProcess1(ptr1, sr)  # 函数调用

    lframe = int(sr*0.03)  # 帧长(持续0.02秒)
    mframe = int(sr*0.015)  # 帧移

    endframe, fn1, numfillzero = preprocess.frame(y, lframe, mframe)  # 分帧

    # 加窗
    hanwindow = np.hamming(lframe)  # 调用汉明窗，把参数帧长传递进去
    signalwindow = endframe * hanwindow

    mlmcList = np.ndarray(shape=(145,1))
    for i in signalwindow:
        mfcc = librosa.feature.mfcc(i,n_mfcc=60,sr =sr)
        # librosa.display.specshow(mfcc, x_axis="time", y_axis="mel", sr=sr)
        
        lm = librosa.feature.melspectrogram(i,n_mels = 60,sr =sr)

        log_mel_spectrogram = librosa.power_to_db(lm)
        # librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)

        ch = librosa.feature.chroma_stft(i,sr =sr)
        spectralContrast = librosa.feature.spectral_contrast(i,sr =sr)
        tonnetz = librosa.feature.tonnetz(i,sr =sr)

        mlmc = np.concatenate((mfcc,log_mel_spectrogram,ch,spectralContrast,tonnetz),axis=0)
        mlmcList = np.concatenate((mlmcList,mlmc),axis =1)
    return mlmcList[1:]
    # MFCC_a = []  # MFCC用于存放每一段音频的MFCC特征参数
    # for l in range(0, len(ZERO2) - 1, 2):
    #     audio_dst = y[ZERO2[l] * mframe: ZERO2[l + 1] * mframe]  # 截取
    #     mfccs = librosa.feature.mfcc(audio_dst, n_mfcc=13)  # 求MFCC
    #     MFCC_a.append(mfccs)  # 把每一段的mfcc特征参数存进MFCC之中

if __name__ == "__main__":
    dataList = os.listdir("./Dataset")
    for i in dataList:
        print(i.split("-")[1][0])
        if os.path.exists(os.path.join("./data/",i+".npy")) or i.split("-")[1][0] != "3":
            continue
        y, sr = librosa.load("./Dataset/" + i)
        mlmc = process(y,sr)
        np.save(os.path.join("./data/",i+".npy"),mlmc)