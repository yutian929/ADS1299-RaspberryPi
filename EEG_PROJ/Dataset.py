import csv
import random
import pickle
import os
import glob
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
from preprocessing import preprocessing_single
import torch

class EEG_Dataset(Dataset):
    """
     :param root: location of dataset
     :param mode: way to use dataset
     :param repre: rewrite pickle and csv and repreprosessing, should be True if data is new
     """
    def __init__(self, root, mode, repre=False, ica=True):
        super(EEG_Dataset, self).__init__()

        self.root = root
        self.name2label = {}
        self.repre = repre
        self.ica = ica
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.signals, self.labels = self.load_csv('eeg_dataset.csv')

        if mode == 'train':
            self.signals = self.signals[:int(0.6*len(self.signals))]
            self.labels = self.labels[:int(0.6*len(self.labels))]

        elif mode == 'test':
            self.signals = self.signals[int(0.8*len(self.signals)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

        elif mode == 'val':
            self.signals = self.signals[int(0.6*len(self.signals)):int(0.8*len(self.signals))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]

        elif mode == 'pred':
            self.signals = self.signals[-64:]
            self.labels = self.labels[-64:]

        elif mode == 'label2name':
            self.signals = self.signals[-1:]
            self.labels = self.labels[-1:]

        elif mode == 'all':
            self.signals = self.signals
            self.labels = self.labels

    def load_csv(self, filename):
        pickle_file = os.path.join(self.root, 'eeg_dataset_preprocessed.pkl')

        if os.path.exists(pickle_file) and self.repre == False:
            with open(pickle_file, 'rb') as f:
                preprocessed_data = pickle.load(f)
            return preprocessed_data['signals'], preprocessed_data['labels']

        else:
            signals = []
            for name in self.name2label.keys():
                signals += glob.glob(os.path.join(self.root, name, '*.xls'))

            random.shuffle(signals)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for signal in signals:
                    name = signal.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([signal, label])

            signals, labels = [], []
            with open(os.path.join(self.root, filename)) as f:
                reader = csv.reader(f)
                for row in reader:
                    signal, label = row
                    label = int(label)

                    df = pd.read_excel(signal, header=None)
                    data_array = df.values
                    single_data = preprocessing_single(data_array, ica=self.ica)
                    signal_tensor = torch.from_numpy(single_data).float()
                    label_tensor = torch.tensor(label).long()

                    signals.append(signal_tensor)
                    labels.append(label_tensor)

            assert len(signals) == len(labels)

            preprocessed_data = {'signals': signals, 'labels': labels}
            with open(pickle_file, 'wb') as f:
                pickle.dump(preprocessed_data, f)

            return signals, labels


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def label2name(label_map, label_index):
    for name, index in label_map.items():
        if index == label_index:
            return name
    raise ValueError("Label index not found in the label map.")
#
# db = EEG_Dataset('dataset/zyt_eeg', 'val')
# train_loader = DataLoader(db, batch_size=64, shuffle=True)
# for x, y in train_loader:
#     print(x.shape, y.shape)