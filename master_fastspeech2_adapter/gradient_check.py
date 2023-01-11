from torch.autograd import gradcheck
import torch
import torch.nn as nn
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2

fastspeech2 = FastSpeech2()

for name, param in fastspeech2.named_parameters():
    if param.requires_grad:
        print(name)