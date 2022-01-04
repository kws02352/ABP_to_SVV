import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

def load_model(saved_model, num_classes, device):
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes = num_classes
    ).to(device)
    
    model_path = os.path.join(saved_model, 'best.pth')
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def rmse(predictions, targets):
    '''
    rmse 계산.
    '''
    return np.sqrt(((predictions - targets) ** 2).mean())

def mse(predictions, targets):
    '''
    mse 계산.
    '''
    return ((predictions - targets) ** 2).mean()

def smooth(y, box_pts):
    '''
    plotting 결과 smoothing.
    '''
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

@torch.no_grad()
def inference(model_dir, output_dir, args):
    """ 
    Parameters
    ----------
    model_dir : str
        학습 된 모델이 있는 폴더 경로.
        
    output_dir : str
        inference 결과 시각화 된 이미지 저장 경로.
    
    args
        parameters.(epoch, batch_size, ...)
        
    """
    path = '/srv/project_data/SV_sanghyun/base_ppv/test/'
    test_files = os.listdir(path)
    
    cnt = 0
    
    rmse_arr = []
    mse_arr = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = load_model(model_dir, args.num_classes, device)
    model.eval()
    
    f = open('./file_list.txt')
    lines = []
    while True:
        line =  f.readline()
        if not line: break
        line = line.strip()
        if line.split(' ')[1] == 'test':
            lines.append(line.split(' ')[2])
    f.close()
    
    for test_file in test_files:
        if test_file != '.ipynb_checkpoints' and test_file.startswith('data'):
            num = test_file[5:-4]

            test_ABP = np.load('/srv/project_data/SV_sanghyun/base_ppv/test/data_'+str(num)+'.npy', allow_pickle = True)
            test_ABP = test_ABP.transpose((0,2,1))
            test_SVV = np.load('/srv/project_data/SV_sanghyun/base_ppv/test/svv_'+str(num)+'.npy', allow_pickle = True)
            test_SVV = test_SVV.astype(int)
            
            test_ABP = torch.Tensor(test_ABP)
            test_SVV = torch.Tensor(test_SVV)

            test_set = TensorDataset(test_ABP, test_SVV)

            test_loader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=False,
            )
            
            print("Calculating inference results..")
            preds = []
            labels_arr = []
            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    pred = model(inputs)
                    preds.extend(pred.cpu().numpy())
                    labels_arr.extend(labels.cpu().numpy())
            preds = np.array(preds)
            # preds = smooth(preds, 15)
            preds = preds.astype(int)
            r = np.arange(0, len(test_SVV))
            result = rmse(preds, np.array(labels_arr))
            result2 = mse(preds, np.array(labels_arr))
            print("rmse: ", result)
            print("mse: ", result2)
            rmse_arr.append(result)
            mse_arr.append(result2)
            title_name = test_file + ': ' + str(result)
            
            filename = lines[cnt]
            
            plt.figure(figsize=(40,10))
            title = filename[:-4] + ", RMSE: " + str(result) + ", MSE: " + str(result2)
            plt.title(title, fontsize=30)
            plt.xlabel('Time', fontsize=30)
            plt.xticks(fontsize=30)
            plt.ylabel('SVV(%)', fontsize=30)
            plt.yticks(fontsize=30)
            plt.plot(r, np.array(preds), color='blue', linestyle='solid', linewidth=0.5)
            plt.plot(r, np.array(labels_arr), color='red', linestyle='solid', linewidth=0.5)
            plt.savefig(output_dir + filename[:-4] + '.png')
            plt.close()
            cnt += 1
        else:
            continue

    total_rmse = sum(rmse_arr) / cnt
    total_mse = sum(mse_arr) / cnt
    print('rmse total: '+ str(total_rmse))
    print('mse total: '+ str(total_mse))
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Net', help='model type (default: LSTM)')

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/paper'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output_sep_rmse'))
    
    # Hyper-parameters
    parser.add_argument('--num_classes', default=1, help='class number')

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True) # output_dir 생성

    inference(model_dir, output_dir, args)
