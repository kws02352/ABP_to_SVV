import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
import statsmodels.api as sm
from scipy.stats import pearsonr

def bland_altman_plot(data1, data2):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff)  # Standard deviation of the difference
    
    plt.title('Bland_Altman')
    plt.xlabel("Means")
    plt.ylabel("Difference")
    plt.scatter(mean, diff, s=0.1, c='red')
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.savefig('./Bland-Altman.png')
    plt.close()
    
    f, ax = plt.subplots(1, figsize = (16,10))
    
    sm.graphics.mean_diff_plot(data2, data1, ax = ax, scatter_kwds = dict(s = 0.1) )
    plt.ylim([-30, 30])
    plt.savefig("./Bland-Altman1.png")
    plt.close()

def pearsonCorr(data1, data2):
    corr, _ = pearsonr(data1, data2)
    return corr

def quadrant(data1, data2):
    mean_pred = np.mean(data1)
    mean_svv = np.mean(data2)
    
    plt.title('Quadrant plot')
    plt.ylim(0,40)
    plt.xlim(0,40)
    plt.plot([-100,100],[-100,100],color='gray',  linestyle='--',linewidth=0.6)
    plt.axhline(y=mean_svv, color='k', linestyle='--', linewidth=1)           
    plt.axvline(x=mean_pred, color='k',linestyle='--', linewidth=1) 
    plt.xlabel('prediction')    
    plt.ylabel('SVV')
    plt.scatter(data1, data2, s=0.1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig("./quadrant.png")
    
def load_model(saved_model, num_classes, device):
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes = args.num_classes
    ).to(device)
    
    model_path = os.path.join(saved_model, 'best.pth')
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

@torch.no_grad()
def inference(model_dir, output_dir, args):
    path = '/srv/project_data/SV_sanghyun/base_ppv/test/'
    test_files = os.listdir(path)
    
    cnt = 0
    
    total_pred = []
    total_svv = []
    
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
    plus = []
    for test_file in test_files:
        if test_file != '.ipynb_checkpoints' and test_file.startswith('data'):
            num = test_file[5:-4]

            test_ABP = np.load('/srv/project_data/SV_sanghyun/base_ppv/test/data_'+str(num)+'.npy', allow_pickle = True)
            plus.append(test_ABP.shape[0])
            test_ABP = test_ABP.transpose((0,2,1))
            test_SVV = np.load('/srv/project_data/SV_sanghyun/base_ppv/test/svv_'+str(num)+'.npy', allow_pickle = True)
            
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
                    total_pred.append(pred.cpu().numpy())
                    total_svv.append(labels.cpu().numpy())
            preds = np.array(preds)
            r = np.arange(0, len(test_SVV))
            result = rmse(preds, np.array(labels_arr))
            result2 = mse(preds, np.array(labels_arr))
            rmse_arr.append(result)
            mse_arr.append(result2)
            title_name = test_file + ': ' + str(result)
            
            filename = lines[cnt]

            print(filename[:-4])
            print("rmse: ", result)
            print("mse: ", result2)
            
            plt.figure(figsize=(50,10))
            plt.title(filename[:-4])
            plt.plot(r, np.array(preds), color='blue', linestyle='solid', linewidth=0.5)
            plt.plot(r, np.array(labels_arr), color='red', linestyle='solid', linewidth=0.5)
            plt.close()
            cnt += 1
        else:
            continue
    total_rmse = sum(rmse_arr) / cnt
    total_mse = sum(mse_arr) / cnt
    print('rmse total: '+ str(total_rmse))
    print('mse total: '+ str(total_mse))
    
    total_pred = np.asarray(total_pred)
    total_svv = np.asarray(total_svv)
    
    total_preds = []
    total_svvs = []
    for i in range(len(total_pred)):
        if len(total_pred[i]) != args.batch_size or len(total_svv[i]) != args.batch_size:
            continue
        else:
            total_preds.append(total_pred[i])
            total_svvs.append(total_svv[i])
    total_preds = np.asarray(total_preds)
    total_svvs = np.asarray(total_svvs)
    total_preds = np.reshape(total_preds, (-1))
    total_svvs = np.reshape(total_svvs, (-1))
    total_preds = np.asarray(total_preds)
    total_svvs = np.asarray(total_svvs)
    
    a = total_preds.tolist()
    b = total_svvs.tolist()
    f = open('data.txt', 'w') # ICC (R 활용)
    f.write('# pred svv\n')
    for i in range(len(total_preds)):
        f.write('{0:.1f}, {1:.1f}\n'.format(a[i], b[i]))
    f.close()
    
    bland_altman_plot(total_preds, total_svvs)
    print(pearsonCorr(total_preds, total_svvs))
    quadrant(total_preds, total_svvs)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Net', help='model type (default: LSTM)')

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/paper')) # v 1.0
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/paper_ver_1_1')) # v 1.1
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output_sep_rmse'))
    
    # Hyper-parameters
    parser.add_argument('--num_classes', default=1, help='class number')

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(model_dir, output_dir, args)
