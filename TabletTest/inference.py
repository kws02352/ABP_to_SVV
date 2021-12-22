import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

def load_model(saved_model):
    model_module = getattr(import_module("model"), 'Net')
    model = model_module(
        num_classes = 1
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path))
    return model

def save_abp_svv_10sec(abp_data):
    abp_fft = np.fft.fft(abp_data, 1000)
    abp_fft[0] = 0
    abp_data = np.fft.ifft(abp_fft)

    tmp_data = []

    for k in range(10):
        tmp_fft = np.abs(np.fft.fft(abp_data[k:k + 100], 100))
        tmp_fft[0] = 0
        tmp_fft = np.fft.fftshift(tmp_fft)
        tmp_data.append(tmp_fft)

    tmp_dpdt = np.diff(abp_data.flatten(),1)
    tmp_dpdt = np.append(tmp_dpdt, 0)
    tmp_dpdt.flatten()

    tmp_data = np.array(tmp_data)
    fft_data = tmp_data.flatten()

    zfft = np.abs(np.fft.fft(abp_data, 4096))
    zfft[0] = 0
    zfft = np.fft.fftshift(zfft)
    zfft = zfft[len(zfft)//2-500:len(zfft)//2+500]    

    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [1000, 1])

    zfft_data = np.array(zfft)
    zfft_data = np.reshape(zfft_data, [1000, 1])

    dpdt = np.array(tmp_dpdt)
    dpdt = np.reshape(dpdt, [1000, 1])

    data = np.array(abp_data)
    data = np.reshape(data, [1000, 1])

    data = np.dstack([data, dpdt, zfft_data])
    
    return data

class TestModel():
    def __init__(self, model_dir):
        self.model = load_model(model_dir)

    @torch.no_grad()
    def inference(self, abp_data):
        self.model.eval()

        test_ABP = np.array(abp_data)
        test_ABP = test_ABP.squeeze(1)
        test_ABP = save_abp_svv_10sec(test_ABP) # (1000, 1, 3)
        test_ABP = test_ABP.transpose((1,2,0))
        
        test_ABP = torch.Tensor(test_ABP)

        preds = []
        labels_arr = []
        with torch.no_grad():
            inputs = test_ABP
            pred = self.model(inputs).cpu().numpy()
        pred = pred.astype(int)

        return pred[0]

if __name__ == '__main__':
    model_dir = './model/paper'
    data = np.load('./test_data/data_0.npy')
    abp_data = data[:100]
    print("init")
    test = TestModel(model_dir = model_dir)
    print("Calculating inference results..")
    for i in abp_data:
        result = test.inference(abp_data = i) # input: ABP(1000,)
        print(result)
    