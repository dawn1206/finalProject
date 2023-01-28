import librosa.display
import os
import numpy as np
import ctypes
import time
import librosa.display

def pre(x):#2n
    signal_points = len(x)  # 获取语音信号的长度
    signal_points = int(signal_points)  # 把语音信号的长度转换为整型
    X = list(range(0, signal_points, 1))  # 创建一个新的List存储预加重后的音频数据
    X[0] = x[0]
    for i in range(1, signal_points, 1):  # 对采样数组进行for循环计算
        X[i] = x[i] - 0.98 * x[i - 1]  # 一阶FIR滤波器
    return X  # 返回预加重以后的采样数组

def zcr(x,lf,fn):
    b = np.multiply(x[:,0:lf-1],x[:,1:])
    b1 = np.less(np.sign(b),0)
    b = np.take(np.cumsum(b1,axis=1),-1,axis=1)
    b = np.divide(b,lf)
    return b

#分帧
def frame(x, lframe, mframe):

    signal_length = len(x)  # 获取语音信号的长度
    fn = (signal_length-lframe)/mframe  # 分成fn帧
    fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
    fn1 = int(fn1)  # 将帧数化为整数
        # 求出添加的0的个数
    numfillzero = (fn1*mframe+lframe)-signal_length
        # 生成填充序列
    fillzeros = np.zeros(numfillzero)
        # 填充以后的信号记作fillsignal
    fillsignal = np.concatenate((x,fillzeros))  # concatenate连接两个维度相同的矩阵
        # 对所有帧的时间点进行抽取，得到fn1*lframe长度的矩阵d
    d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1*mframe, mframe), (lframe, 1)).T
        # 将d转换为矩阵形式（数据类型为int类型）
    d = np.array(d, dtype=np.int32)
    signal = fillsignal[d]
    return(signal, fn1, numfillzero) # 返回分帧数据、帧数和添零数

def endPointDetect(energy, zeroCrossingRate):#3n
        energyAverage = np.average(energy)
        ML = np.sum(energy[:5]) / 5
        MH = energyAverage / 2          # 较高的能量阈值，本文取平均能量的一半
        ML = (ML + MH) / 4    # 较低的能量阈值

        Zs = float(np.sum(zeroCrossingRate)) / len(zeroCrossingRate)
        A = []# 过零率阈值
        B = []
        C = []
            # 首先利用较大能量阈值 MH 进行初步检测

        flag = 0
        for i in range(len(energy)):
            if len(A) == 0 and flag == 0 and energy[i] > MH :
                A.append(i)
                flag = 1
            elif flag == 0 and energy[i] > MH and i - 200 > A[len(A) - 1]:  # 100为距离值，可调整
                A.append(i)
                flag = 1
            elif flag == 1 and energy[i] < MH and i - 200 > A[len(A) - 1]:
                A.append(i)
                flag = 0
       #     print("较高能量阈值，计算后的端点为:" + str(A))

            # 利用较小能量阈值 ML 进行第二步能量检测
        for j in range(len(A)) :
            i = A[j]
            if j % 2 == 1 :
                 while i < len(energy) and energy[i] > ML :
                     i = i + 1
                 B.append(i)
            else :
                while i > 0 and energy[i] > ML :
                   i = i - 1
                B.append(i)

        #    print("较低能量阈值，计算后端点为:" + str(B))


    # 利用过零率进行最后一步检测
        for j in range(len(B)) :
            i = B[j]
            if j % 2 == 1 :
                while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs :
                    i = i + 1
                C.append(i)
            else :
                while i > 0 and zeroCrossingRate[i] >= 3 * Zs :
                    i = i - 1
                C.append(i)
        #    print("过零率阈值，计算后端点为:" + str(C))
        return A, B, C

#时间规整
def adjust(M, L):
    M = np.array(M)  # 数组化
 # 数组化
    N1 = M.transpose()
    m2 = list(np.ones(L))  # m2为权重向量组

    if L > 5:  # 规整为5帧
        while L > 5:
            dl = np.zeros(L - 1)  # 创建存储欧几里得距离的数组
            for r in range(L - 1):
                dl[r] = np.linalg.norm(N1[r] - N1[r + 1])  # 计算欧几里得距离

            dlMin = dl.argmin()
             #   N1[dlMin] = (N1[dlMin] * m2[dlMin] + N1[dlMin + 1] * m2[dlMin + 1]) / (m2[dlMin] + m2[dlMin + 1])
            N1[dlMin] = np.divide(np.add(np.multiply(N1[dlMin], m2[dlMin]),np.multiply(N1[dlMin+1], m2[dlMin+1])) , (np.add(m2[dlMin] , m2[dlMin + 1])))  # 融合公式
                # 融合公式
            m2[dlMin] = m2[dlMin] + m2[dlMin + 1]

            N1 = np.delete(N1,dlMin+1,0)  # 删除被融合向量
            m2.pop(dlMin + 1)  # 删除被融合参数
            L = L - 1
        return (N1)
    elif L == 5:
        return (N1)
    elif L < 5:
        return 0

def sound(y, sr):
    #预加重

    pre_emphasis1 = y.copy()
    ptr1 = pre_emphasis1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cDLLcontroller = ctypes.cdll.LoadLibrary(r'.\Dll2.dll')
    cDLLcontroller.audioProcess1(ptr1,sr) # 函数调用

    lframe = int(sr*0.02)  # 帧长(持续0.02秒)
    mframe = int(sr*0.001)  # 帧移

    endframe, fn1, numfillzero = frame(y, lframe, mframe) #分帧

    # 加窗
    hanwindow = np.hanning(lframe)  # 调用汉明窗，把参数帧长传递进去

    signalwindow = endframe * hanwindow

    # 短时能量
    c = np.square(signalwindow)
    m_energy = np.take(np.cumsum(c,axis=1),-1,axis=1)

   # for i in range(0, fn1):
    #    m_energy[i] = np.sum(c[i])
    #    c = np.square(signalwindow[i])


    m_energy = m_energy * 1.0/(np.max(np.abs(m_energy))) # 归一化

    # 短时平均过零率
    m_energy1 = m_energy.copy()
    rate = zcr(signalwindow,lframe,fn1)

    # 利用短时能量，短时过零率，使用双门限法进行端点检测
    t1 = time.time_ns()
    MAX, MIN, ZERO2 = endPointDetect(m_energy, rate)
    ZERO2 = np.unique(ZERO2).tolist()

    if len(ZERO2) %2 !=0:
        ZERO2.pop(len(ZERO2)-1)
    # 截取有效音频段
    n = len(ZERO2) / 2  # n为截取的数目

    n = int(n)

    MFCC_a = [] # MFCC用于存放每一段音频的MFCC特征参数
    for l in range(0, len(ZERO2)-1, 2):
        audio_dst = y[ZERO2[l] * mframe : ZERO2[l + 1] * mframe]  #截取
        mfccs = librosa.feature.mfcc(audio_dst, n_mfcc=40)  # 求MFCC
        MFCC_a.append(mfccs) #把每一段的mfcc特征参数存进MFCC之中

    # 时间规整

    MFCC_adjust = np.zeros((n,5,40))
    for w in range(n):
        M = MFCC_a[w]
            # 取第w段语音的24维MFCC参数
            # 取帧数
        L = np.shape(M)[1]

        middle = adjust(M, L)  # 调用规整函数 middle 是array

        if isinstance(middle, int):
            continue
        # MFCC_adjust.append(middle)  #将规整后的每一段语音的MFCC特征值放入MFCC_adjust

        MFCC_adjust[w] = middle

    return(MFCC_adjust)

def mfcc(wav_dir):

    Path = os.listdir(wav_dir)
    output = []
    for i in range(len(Path)): #分别读取n种的音频

        print(Path[i] + 'start')
        y, sr = librosa.load(os.path.join(wav_dir,Path[i]))  # 读取文件
        duration = librosa.get_duration(y)

        if duration < 1:
            continue

        MFCC = sound(y, sr)  #调用sound函数

        #计算MFCC的一阶差分系数
        lens_of_MFCC = np.shape(MFCC)[0]
        for j in range(lens_of_MFCC):
            for z in range(4):
                n = list(map(lambda x: x[1] - x[0], zip(MFCC[j][z], MFCC[j][z+1])))
                n = np.array(n)
                n = np.expand_dims(n,axis=0)
                n = np.expand_dims(n, axis=0)
                if z ==0:
                    delta = n
                else:
                    delta = np.concatenate((delta,n),axis=1)
            if j ==0:
                delta_l = delta
            else:
                delta_l = np.concatenate((delta_l,delta),axis=0)
        MFCC = np.concatenate((MFCC,delta_l),axis=1)
        Mfcc =np.expand_dims(MFCC,3)
        output.append(MFCC)
    output = np.array(output)
    return output

def mfcc_single(wav_dir):
    y, sr = librosa.load(wav_dir)  # 读取文件
    duration = librosa.get_duration(y)

    if duration > 1:
        pushornot = True
    else:
        pushornot = False

    if pushornot == False:
        return None

    MFCC = sound.sound(y, sr)  # 调用sound函数

    # 计算MFCC的一阶差分系数
    lens_of_MFCC = np.shape(MFCC)[0]
    for j in range(lens_of_MFCC):
        for z in range(4):
            n = list(map(lambda x: x[1] - x[0], zip(MFCC[j][z], MFCC[j][z + 1])))
            n = np.array(n)
            n = np.expand_dims(n, axis=0)
            n = np.expand_dims(n, axis=0)
            if z == 0:
                delta = n
            else:
                delta = np.concatenate((delta, n), axis=1)
        if j == 0:
            delta_l = delta
        else:
            delta_l = np.concatenate((delta_l, delta), axis=0)
    MFCC = np.concatenate((MFCC, delta_l), axis=1)
    return MFCC