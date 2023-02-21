import librosa
import os
import numpy as np
import preprocess
import ctypes
import time

def process(y,sr,length = 399,feature = 213):
    # pre_emphasis1 = y.copy()
    # ptr1 = pre_emphasis1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # cDLLcontroller = ctypes.cdll.LoadLibrary(r'.\Dll2.dll')
    # cDLLcontroller.audioProcess1(ptr1, sr)  # 函数调用

    lframe = int(sr*0.01)  # 帧长(持续0.02秒)
    mframe = int(sr*0.005)  # 帧移

    endframe, fn1, numfillzero = preprocess.frame(y, lframe, mframe)  # 分帧

    # 加窗
    hanwindow = np.hamming(lframe)  # 调用汉明窗，把参数帧长传递进去
    signalwindow = endframe * hanwindow

    m = np.ndarray(shape=(feature,1))
        # for i in signalwindow:
    mfcc = librosa.feature.mfcc(signalwindow,n_mfcc=60,sr =sr)
        # librosa.display.specshow(mfcc, x_axis="time", y_axis="mel", sr=sr)
        
    lm = librosa.feature.melspectrogram(signalwindow,n_mels = 128,sr =sr)

    log_mel_spectrogram = librosa.power_to_db(lm,ref=np.max)
        # librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)

    ch = librosa.power_to_db(librosa.feature.chroma_stft(signalwindow,sr =sr),ref=np.max)
    spectralContrast = librosa.feature.spectral_contrast(signalwindow,sr =sr)
    tonnetz = librosa.power_to_db(librosa.feature.tonnetz(signalwindow,sr =sr),ref=np.max)

    for i in range(len(signalwindow)):
        mlmc = np.concatenate((mfcc[i],log_mel_spectrogram[i],ch[i],spectralContrast[i],tonnetz[i]),axis=0)
        m = np.concatenate((m,mlmc),axis =1)
    m = m[:,1:]
    repeatTimes = int(np.floor(length/m.shape[1]))
    if m.shape[1] != length:
        n = np.zeros((1,1,feature,length))
        for ind in range(repeatTimes):
            n[0][0][:,ind*m.shape[1]:(ind+1)*m.shape[1]] = m[:,:m.shape[1]]
        m = n
        m = m[:,:length].reshape((1,1,feature,length))
    return m

if __name__ == "__main__":
    name = "/home/calino/soundClassification/dataTrulySin/dataSingle/cough"
    storeDir = "/home/calino/soundClassification/dataTrulySin/dataTrulySinNPY"
    dataList = os.listdir(name)
    # duraList = []
    # npyList = []
    for i in dataList:
        if len(i.split("."))!=2:
            continue
        y, sr = librosa.load(name+"/" + i)
        d = librosa.get_duration(y)
        if d < 0.01:
            continue
        mlmc = process(y,sr)
        # duraList.append(mlmc.shape[1])
        # npyList.append(mlmc)
        np.save(os.path.join(storeDir,"cough",i+".npy"),mlmc)

