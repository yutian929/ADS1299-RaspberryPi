# 2023/07/28 zjn: 在跑16*63的网络，用的v2
# 2023/07/29 zyt: uniform NET.py, assemble all net classes
# 2023/07/29 zjn: 创建EEG_Dataset数据集，节省预处理时间

import numpy as np
import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from NET import CSAC_NET, LINEAR_NET, ResNet18, Lenet5, CSAC_NET_udlsrn
from Dataset import EEG_Dataset

# batch_size
batch_size = 64

best_test_res = {
    'acc': 0,
    'loss': 100,
}

modelname = 'RES_fff_23_07_31.pth'

# learning_rates = [0.004, 0.005, 0.006, 0.007, 0.005, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]
# learning_rates = [0.00009, 0.0001, 0.00012, 0.00014, 0.00016, 0.00018, 0.00002, 0.00003, 0.00004, 0.00005]
learning_rates = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]

def train():
    # get dataset
    train_dataset = EEG_Dataset('dataset/zyt_fff', 'train')
    val_dataset = EEG_Dataset('dataset/zyt_fff', 'val')
    test_dataset = EEG_Dataset('dataset/zyt_fff', 'test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Use the GPU if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    learning_rate = learning_rates

    for idx, lr in enumerate(learning_rate):
        best_res = train_dnn(train_loader, val_loader, lr, device)
        model = CSAC_NET_udlsrn(ch_in=8, ch_out=512).to(device)
        # wrongmodel = CSAC_NET().to(device)
        # wrongmodel = LINEAR_NET(ch_in=8, ch_out=512)
        pretrain_model_path = 'model/' + modelname
        if os.path.exists(pretrain_model_path):
            model.load_state_dict(torch.load(pretrain_model_path))
        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)
            test_acc = total_correct / total_num
        print('Lr : {} -- TestAcc : {}\n'.format(lr, test_acc))


def train_dnn(train_loader, test_loader, lr, device):
    model = CSAC_NET_udlsrn(ch_in=8, ch_out=512)  # 使用输入数据的特征数量作为输入大小
    # wrongmodel = LINEAR_NET(ch_in=8, ch_out=512)
    model.to(device)
    epoch_num = 3000
    learning_rate = lr
    stop = 0
    stop_threshold = 2000
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    # myloss = nn.BCEWithLogitsLoss()
    myloss = nn.CrossEntropyLoss()

    global_step = 0

    for ep in range(epoch_num):
        if stop >= stop_threshold and ep > 1000:
            print(f'Stopping training for LR={lr} as the stop count exceeded the threshold.')
            break

        model.train()
        total_correct = 0
        total_num = 0
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            loss = myloss(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = total_correct / total_num


        print('Lr : {} -- Epoch : {} -- TrainLoss : {} -- TrainAcc : {}\n'.format(lr, ep, loss.cpu().data,
                                                                                  train_acc))

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for batchidx, (x, label) in enumerate(test_loader):
                x, label = x.to(device), label.to(device)
                logits = model(x)
                test_loss = myloss(logits, label)

                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
            test_acc = total_correct / total_num
            print('Lr : {} -- Epoch : {} -- Val_Loss : {} -- ValAcc : {}\n'.format(lr, ep, test_loss.cpu().data,
                                                                                    test_acc))

            global_step += 1

            # save the best results
            if best_test_res['acc'] < test_acc:
                best_test_res['acc'] = test_acc
                best_test_res['loss'] = test_loss.cpu().data
                print('update res')

                model_directory = 'model'
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)

                # save wrongmodel
                model_filename = os.path.join(model_directory, modelname)
                torch.save(model.state_dict(), model_filename)
            else:
                stop += 1
        scheduler.step()
    return best_test_res


if __name__ == '__main__':
    train()
