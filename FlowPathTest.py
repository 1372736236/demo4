# -*-coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import json
import numpy as np
import collections
import os
import time
from tqdm import tqdm
from sklearn.preprocessing import scale
from librosa import feature

# LeNet-5调参模型
class SoundClassificationNet(nn.Module):
    def __init__(self, num_classes):
        super(SoundClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 11, 128)  # 根据卷积和池化的参数计算的展平大小
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # 自动计算展平大小
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Features:
    def __init__(self, wavlist):
        """
        初始化需求
        :param wavlist: 传入原始音频数据
        """
        self.wavlist = wavlist
        self.wavData, self.sr = librosa.load(wavlist)

    def Overall(self, segments, SPL_THRESHOLD=55):
        """
        找出超标段落
        :param segments 音频段落列表
        :param SPL_THRESHOLD 超标阈值
        :return 超标段落索引列表
        """
        over_threshold_indices = []

        for i, segment in enumerate(segments):
            # 计算每段的平均振幅
            segment_amplitude = np.sqrt(np.mean(np.square(segment)))

            # 计算声压级 SPL
            SPL = 20 * np.log10(segment_amplitude / (2e-5))

            # 判断是否超标
            if SPL > SPL_THRESHOLD:
                over_threshold_indices.append(i)  # 使用段落的索引标记超标

        return over_threshold_indices

    def getUsefulWav(self, FRAME_SIZE = 256, HOP_LENGTH = 128):
        """
        获取有用声片段
        :param FRAME_SIZE 窗口
        :param HOP_LENGTH 帧移
        :return
        """
        rms_sign = librosa.feature.rms(y=self.wavData, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        rms_med = np.median(rms_sign)
        frames = range(len(rms_sign))
        t = librosa.frames_to_time(frames, sr=self.sr, hop_length=HOP_LENGTH)
        # 找到有效区间的开始和结束时间
        start_frames = np.argwhere(rms_sign > 2 * rms_med)
        stop_frames = np.argwhere(rms_sign > 1.5 * rms_med)
        start = t[start_frames[0][0]] if start_frames.size > 0 else None
        stop = t[stop_frames[-1][0]] if stop_frames.size > 0 else None

        # 计算有效区间的总长度并切割
        if start is not None and stop is not None:
            effective_duration = stop - start
            num_full_segments = int(effective_duration)
            remainder = effective_duration - num_full_segments
            threshold = 0.5

            # 切割每个完整的1秒片段
            segments = []
            for i in range(num_full_segments):
                start_sample = int((start + i) * self.sr)
                end_sample = int((start + (i+1)) * self.sr)
                segment = self.wavData[start_sample:end_sample]
                segments.append(segment)

            # 处理最后一个不完整的片段
            if remainder > threshold:
                # 如果长度超过阈值，则补齐
                start_sample = int((start + num_full_segments) * self.sr)
                end_sample = int((start + num_full_segments + 1) * self.sr)
                segment = self.wavData[start_sample:end_sample]
                # 如果补齐需要填充静音
                if len(segment) < self.sr:
                    segment = np.pad(segment, (0, self.sr - len(segment)), 'constant')
                segments.append(segment)
            return start, segments
        else:
            return []

    def getMfcc(self, segments):
        """
        获取MFCC特征
        :param segments: 输入的音频段列表
        :return:
        all_mfcc, all_mfcc_m, all_mfcc_scale ：音频文件的MFCC特征、平均值和缩放后的MFCC特征
        """
        # MFCC的特征数
        n_mfcc = 40
        all_mfcc = np.empty((0, n_mfcc, 45))
        all_mfcc_m = np.empty((0, n_mfcc))
        all_mfcc_scale = np.empty((0, n_mfcc, 45))
        for segment in segments:
            # 从音频文件中加载时间序列和采样率,最大时间限制为1s
            mfcc = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=n_mfcc)
            mfcc_m = np.mean(mfcc, axis=1).T
            if mfcc.shape[1] < 45:
                padding = np.zeros((n_mfcc, 45 - mfcc.shape[1]))
                mfcc = np.concatenate([mfcc, padding], axis=1)

            # 将MFCC特征和相关数据添加到相应的存储变量
            all_mfcc = np.vstack((all_mfcc, [mfcc]))
            all_mfcc_m = np.vstack((all_mfcc_m, [mfcc_m]))

            # 数据标准化 (X-X_mean)/X_std
            mfcc_scale = scale(mfcc)
            all_mfcc_scale = np.vstack((all_mfcc_scale, [mfcc_scale]))

        return all_mfcc, all_mfcc_m, all_mfcc_scale

    def mappingChinese(self, map_json, pre):
        """
        映射性
        :param map_json 映射文件选择
        :return 映射对应命名
        """
        with open(map_json, 'r', encoding='utf-8') as file:
            mapping = json.load(file)
        return [mapping[str(num)] for num in pre]


def func():
    # 开启cpu模式，调用Le97.12%识别率模型
    device= "cpu"
    # 加载识别模型
    model = SoundClassificationNet(30)
    # 设置模型调用模式
    model=model.to(device)
    model.load_state_dict(torch.load('model/model_Le9712.pth'))
    # 设置为评估模式
    model.eval()

    strat = time.time()
    # wavlist 加载原始音频，此处使用 使用：A5TDG030/2023-03-10-23-47-34.wav进行测试
    wav_list = ['A5TDG030_2023-03-16-10-39-53_2023-03-16-10-39-59.wav']
    # 将声音分段，变换为n个有效段落 此处加载1个原始音频，调用自编Features库
    features = Features(wav_list[0])
    # 利用getUsefulWav方法，获得有用声音的其实部位以及切取段落
    useful_time_start, segments = features.getUsefulWav()
    # 利用Overall方法，获取超标声音段落
    ind_over_segments = features.Overall(segments)
    # 计算超标段其特征 本模型利用的为all_mfcc_scale特征，故仅需保留标准化特征
    _, _, all_mfcc_scale = features.getMfcc([segments[i] for i in ind_over_segments])
    # 测试输出结果
    inputs = torch.tensor(all_mfcc_scale, dtype=torch.float32)
    # 预测超标声音类型
    pre = []
    with torch.no_grad():
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        pre.extend(predicted.cpu().numpy())

    # 输出预测结果，利用mappingChinese函数，通过载入一份json文件，将其映射为对应中文类型
    pre_chinese = features.mappingChinese('tableJson.json', pre)
    t = time.time() - strat
    #print(time.time() - strat)
    # 打印
    soundstr = '超标声音区间及类型如下：\n'
    #print('超标声音区间及类型如下：')
    for oind, pind in zip(ind_over_segments, pre_chinese):
        soundstr += f"{useful_time_start+oind:.2f}s ~ {useful_time_start+oind+1:.2f}s: {pind}" + '\n'
        #print(f"{useful_time_start+oind:.2f}s ~ {useful_time_start+oind+1:.2f}s: {pind}")
    #print('Time:' + str(t) + 's' + '\n' + soundstr)
    return 'Time:'+str(t)+'s'+'\n'+soundstr

