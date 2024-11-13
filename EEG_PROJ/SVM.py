import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from Dataset import EEG_Dataset

# batch_size
batch_size = 64

# get dataset
train_dataset = EEG_Dataset('dataset/zyt_fff_v2', 'train')
val_dataset = EEG_Dataset('dataset/zyt_fff_v2', 'val')
test_dataset = EEG_Dataset('dataset/zyt_fff_v2', 'test')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# flatten the data
X_train = []
y_train = []
for x, label in train_loader:
    X_train.append(x.view(x.size(0), -1).numpy())
    y_train.append(label.numpy())
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Use GridSearchCV to find the best parameters for SVM
parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)

# validate
X_val = []
y_val = []
for x, label in val_loader:
    X_val.append(x.view(x.size(0), -1).numpy())
    y_val.append(label.numpy())
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

y_val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print('Validation accuracy:', val_acc)

# test
X_test = []
y_test = []
for x, label in test_loader:
    X_test.append(x.view(x.size(0), -1).numpy())
    y_test.append(label.numpy())
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print('Test accuracy:', test_acc)