import fnmatch
import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
num=1
class Aduio_DataLoader(Dataset):
     def __init__(self, data_folder, sr=16000, dimension=8193):
         self.data_folder = data_folder
         self.sr = sr
         self.dim = dimension

         # 获取音频名列表
         self.wav_list = []
         for root, dirnames, filenames in os.walk(data_folder):
             for filename in fnmatch.filter(filenames, "*.wav"):  # 实现列表特殊字符的过滤或筛选,返回符合匹配“.wav”字符列表
                 self.wav_list.append(os.path.join(root, filename))

     def __getitem__(self, item):
         # 读取一个音频文件，返回每个音频数据
         filename = self.wav_list[item]
         # print(filename)
         wb_wav, _ = librosa.load(filename, sr=self.sr)
         # sr为采样率，通过KMplayer查看sampling rate，确认过speech commands为16000

         # 取 帧
         if len(wb_wav) >= self.dim:#self.dim=8193
             # print("yes:len of wb_wav{}:{}".format(filename, len(wb_wav)))
             max_audio_start = len(wb_wav) - self.dim
             audio_start = np.random.randint(0, max_audio_start)
             wb_wav = wb_wav[audio_start: audio_start + self.dim]
         else:
             wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")
             # print("yes:len of wb_wav{}:{}".format(filename, len(wb_wav)))

         return wb_wav, filename

     def __len__(self):
         # 音频文件的总数
         return len(self.wav_list)


train_set = Aduio_DataLoader("/root/speech_command/train", sr=16000, dimension=8193)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

print("len:",len(train_loader))

for (i, data) in enumerate(train_loader):
    # print(i,data)
    wav_data, wav_name = data
    if i==10:
        print(wav_name, wav_data, wav_data.shape)


