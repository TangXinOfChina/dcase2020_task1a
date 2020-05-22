import librosa
import os
import random
import numpy as np
from itertools import islice

sample_rate = 44100

folder_name = "/dockerdata/zhuhongning/data/dcase_audio/"
file_name = os.listdir(folder_name)
label_dict = dict(airport=0, bus=1, metro=2, metro_station=3, park=4, public_square=5, shopping_mall=6, street_pedestrian=7, street_traffic=8, tram=9)
csv_file = '/dockerdata/zhuhongning/dcase2020/evaluation_setup/fold1_trainall.csv'


def pitch_shift():
    save_path = "/dockerdata/zhuhongning/data/dcase2020_pitch/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in file_name:
        y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
        n_step = random.uniform(-4, 4)
        print(n_step)
        y_pitched = librosa.effects.pitch_shift(y, sr, n_steps=n_step)
        name = save_path + file.split('.')[0] + '_pitch' + '.wav'
        librosa.output.write_wav(name, y_pitched, sr)
        # librosa.output.write_wav(save_path + file, y, sr)
        print("file: ", file, "done!")

def noise_injection():
    save_path = "/dockerdata/zhuhongning/data/dcase2020_noise/"
    noise_factor = 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in file_name:
        y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
        noise = np.random.normal(0, 1, len(y))
        augmented_data = np.where(y != 0.0, y.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
        # augmented_data = y + noise_factor * noise
        name = save_path + file.split('.')[0] + '_noise' + '.wav'
        librosa.output.write_wav(name, augmented_data, sr, norm=True)
        # librosa.output.write_wav(save_path + file, y, sr)
        print("file: ", file, "done!")


def time_stretch():
    save_path = "/dockerdata/zhuhongning/data/dcase2020_time/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in file_name:
        y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
        time_factor = random.uniform(0.5, 2)
        length = len(y)
        print(time_factor)
        y_stretch = librosa.effects.time_stretch(y, time_factor)
        if len(y_stretch) < length:
            y_stretch = np.concatenate((y_stretch, y_stretch))
            y_stretch = y_stretch[0:length]
        else:
            y_stretch = y_stretch[0:length]
        name = save_path + file.split('.')[0] + '_time' + '.wav'
        print("file: ", file, "done!")
        librosa.output.write_wav(name, y_stretch, sr)
       #  librosa.output.write_wav(save_path + file, y, sr)

def class_sort():
    class_list = []
    for i in range(10):
        ap = []
        class_list.append(ap)
    with open(csv_file, 'r') as csv_r:
        # reader = csv.reader(csv_r)
        for line in islice(csv_r, 1, None):
            file_name = line.split('\t')[0].split('/')[1]
            label = line.split('\t')[1].split('\n')[0]
            class_list[label_dict[label]].append(file_name)

    return class_list

# 随机选取一个同类音频与当前音频相加
def data_add():
    save_path = "/dockerdata/zhuhongning/data/dcase2020_add/"
    class_list = class_sort()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for label in class_list:
        length = len(label)
        print(length)
        for file in label:
            y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
            num = random.randint(0, length - 1)
            while file == label[num]:
                num = random.randint(0, length - 1)
            f1, f2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
            y2, _ = librosa.load(folder_name + label[num], mono=True, sr=sample_rate)
            y_final = y * f1 + y2 * f2
            name = save_path + file.split('.')[0] + '_add' + '.wav'
            librosa.output.write_wav(name, y_final, sr)
            print("file: ", file, "done!")


if __name__ == "__main__":
    data_add()
