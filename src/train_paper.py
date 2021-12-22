import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from loss import create_criterion

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
        
def train(model_dir, args):
    path = '/srv/project_data/SV_sanghyun/base_ppv/train/'
    train_files = os.listdir(path)
    cnt = 0
    best_val_loss = np.inf

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
                
    val_ABP = '/srv/project_data/SV_sanghyun/base_ppv/val_data.txt'
    with open(val_ABP, 'rb') as abp:
        readList_data = pickle.load(abp)
    val_ABP = np.asarray(readList_data)
    val_ABP = val_ABP.transpose((0,2,1))

    val_SVV = '/srv/project_data/SV_sanghyun/base_ppv/val_svv.txt'
    with open(val_SVV, 'rb') as svv:
        readList_svv = pickle.load(svv)
    val_SVV = np.asarray(readList_svv)
    
    val_ABP = torch.Tensor(val_ABP)
    val_SVV = torch.Tensor(val_SVV)

    val_set = TensorDataset(val_ABP, val_SVV)

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
        
    for train_file in train_files:
        if train_file != '.ipynb_checkpoints' and train_file.startswith('data'):
            cnt += 1
            num = train_file[5:-4]
            
            # -- dataset
            train_ABP = np.load('/srv/project_data/SV_sanghyun/base_ppv/train/'+'data_'+str(num)+'.npy', allow_pickle = True)
            train_ABP = train_ABP.transpose((0,2,1))
            train_SVV = np.load('/srv/project_data/SV_sanghyun/base_ppv/train/'+'svv_'+str(num)+'.npy', allow_pickle = True)
            # train_SVV = train_SVV.astype(int)
                        
            # -- data to tensor
            train_ABP = torch.Tensor(train_ABP)
            train_SVV = torch.Tensor(train_SVV)
                        
            train_set = TensorDataset(train_ABP, train_SVV)
            
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )

            # -- model

            model_module = getattr(import_module("model"), args.model)
            model = model_module(
                num_classes = args.num_classes
            ).to(device)
            model = torch.nn.DataParallel(model)
            wandb.watch(model)

            # -- loss & metric
            criterion = create_criterion(args.criterion)  # default: MSE
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )
            scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.1)

            # -- logging
            logger = SummaryWriter(log_dir=save_dir)
            with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

            if os.path.exists('./model/paper/last.pth'):
                model.module.load_state_dict(torch.load('./model/paper/last.pth', map_location=device))
                print('============LOAD MODEL============')
                
            for epoch in range(args.epochs):
                # train loop
                model.train()
                loss_value = 0
                for idx, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    pred_labels = model(inputs)
                    loss = criterion(pred_labels, labels)

                    loss.backward()
                    optimizer.step()

                    loss_value += loss.item()
                    if (idx + 1) % args.log_interval == 0:
                        train_loss = loss_value / args.log_interval
                        current_lr = get_lr(optimizer)
                        print(
                            f"total[{cnt}/{int((len(train_files)-1)/2)}] || "
                            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                            f"training loss {train_loss:4.4} || lr {current_lr}"
                        )
                        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)

                        loss_value = 0

                        wandb.log({'loss': loss})

                scheduler.step()

                # val loop
                with torch.no_grad():
                    print("Calculating validation results...")
                    model.eval()
                    val_loss_items = []
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        loss_item = criterion(outs, labels).item()
                        val_loss_items.append(loss_item)

                    val_loss = np.sum(val_loss_items) / len(val_loader)
                    if val_loss < best_val_loss:
                        print(f"saving the best model..")
                        best_val_loss = val_loss
                        torch.save(model.module.state_dict(), "./model/paper/best.pth")
                    torch.save(model.module.state_dict(), "./model/paper/last.pth")
                    print(
                        f"[Val] loss: {val_loss:4.4} || "
                        f"best best loss: {best_val_loss:4.4}"
                    )
                    logger.add_scalar("Val/loss", val_loss, epoch)
                    wandb.log({'val_loss': val_loss})
                    wandb.log({'best_val_loss': best_val_loss})
                    print()
        else:
            continue


if __name__ == '__main__':
    wandb.init(project='', entity='') # 기입 필수 (wandb 사용시)
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='mse', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=50, help='learning rate scheduler deacy step (default: 20)') 
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--num_classes', default=1, help='class number')
    parser.add_argument('--model', type=str, default='Net', help='model type (default: LSTM)')
    parser.add_argument('--name', default='0', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)
    
    wandb.config.update(args)
    
    model_dir = args.model_dir

    train(model_dir, args)