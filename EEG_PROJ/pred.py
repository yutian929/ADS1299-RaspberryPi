# 2023/07/28 zjn:用数据集的后64给做预测，若修改在Dataset.py中的38行

import os
import torch
from NET import CSAC_NET, ResNet18, Lenet5, CSAC_NET_fff
from train import modelname
from torch.utils.data import DataLoader
from Dataset import EEG_Dataset

def label2name(label_map, label_index):
    for name, index in label_map.items():
        if index == label_index:
            return name
    raise ValueError("Label index not found in the label map.")


# load model and dataset
# pred_modelname = modelname
# pretrain_model_path = 'model/' + pred_modelname
pretrain_model_path = 'model/CSAC_fff_23_07_31.pth'

model = CSAC_NET(ch_in=8, ch_out=512)
pred_dataset = EEG_Dataset('dataset/zyt_fff_test', mode='all', repre=True)
label_map = pred_dataset.name2label
pred_loader = DataLoader(dataset=pred_dataset, batch_size=1, shuffle=False)


if os.path.exists(pretrain_model_path):
    model.load_state_dict(torch.load(pretrain_model_path))

model.eval()
correct = 0
total_num = 0
for i, (test_data, test_target) in enumerate(pred_loader):
        target = test_target
        out = model(test_data)
        pred_idx = torch.argmax(out, 1).item()
        pred = label2name(label_map, pred_idx)
        true_target = label2name(label_map, target)
        same = (pred_idx == target).item()
        if same == True:
            correct += 1
        total_num += 1
        print('Prediction : {} -- True_Target : {} -- Same? : {}\n'.format(pred, true_target, same))

test_acc = correct/total_num
print('TestAcc : {}\n'.format(test_acc))




