# -*- coding: utf-8 -*-
"""lstm_torch.py

## Premier modèle :: LSTM (PyTorch)

### Importer les bibliothèques nécessaires
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
from torch import nn

import sys

"""### Lire les données"""

data = pd.read_csv('/content/drive/MyDrive/inrae/icos/preprocessed_dataICOS.csv')
# data = pd.read_csv('../data/preprocessed_dataICOS.csv') si téléchargé depuis github
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.head()

"""### Préparation des données & helper functions"""

df = data[['GPP_ann', 'GPP_cycle', 'anomalies', 'cos day','sin day','cos year', 'sin year']]
df.head()

"""
fonction qui à partir du dataframe, crée des séquences sous forme de
[ [GPP_ann(t-L),GPP_cycle(t-L),anomalies(t-L),+time(t-L)], [GPP_ann(t-L+1),GPP_cycle(t-L+1),anomalies(t-L+1),+time(t-L+1)], ... ,
[GPP_ann(t),GPP_cycle(t),anomalies(t),+time(t)] ]
et des targets à t+1 : [GPP_ann(t+1), GPP_cycle(t+1), anomalies(t+1)]
avec L : longueur choisie de la séquence
l'intérêt d'avoir 3 sorties est d'évaluer chaque composante individuellemnt
"""
def inputs_outputs(sequence_length, data):
    data = np.array(data)
    sequences = []
    targets = []
    for i in range(0,len(data)-sequence_length-1):
        sequence = data[i:i+sequence_length] # variables d'intérêt de (t-L) à t + 4 remaining features(time), [GPP_ann,GPP_cycle,anomalies,+time ]
        target = data[i+sequence_length][:3] # variables d'intérêt à t+1 # [GPP_ann(t+1), GPP_cycle(t+1), anomalies(t+1)]
        sequences.append(sequence)
        targets.append(target)
    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets

sequence_length=365
sequences, targets = inputs_outputs(sequence_length, df)

print(sequences.shape, targets.shape) # seq_len=365, features=7

# diviser le dataset en données d'entrainement et de test

X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=19)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# les transformer en tensors compatibles avec torch
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print(X_train_tensor.shape, y_train_tensor.shape)
print(X_test_tensor.shape, y_test_tensor.shape)

# les loader

train_ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False)
test_ds = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

def myloss_fct(weights_matrix):
    def loss(y_true, y_pred):
        weighted_error = torch.square(y_true - y_pred)*weights_matrix # choisir les composantes à prendre en compte et les composantes à ignorer (matrice binaire)
        return torch.mean(weighted_error)
    return loss

weights_matrix_ann = [[1, 0, 0]]    # tenir compte de la composante annuelle
weights_matrix_cycle = [[0, 1, 0]]  # ............................. cycle
weights_matrix_anomalies = [[0, 0, 1]] # .......................... anomalies

# fonctions de loss personnalisés (pondérées)
loss_ann = weighted_loss(weights_matrix_ann)
loss_cycle = weighted_loss(weights_matrix_cycle)
loss_anomalies = weighted_loss(weights_matrix_anomalies)

"""### Modèle Torch

#### Définir une classe LSTM
"""

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2)

        self.linear = nn.Linear(
            in_features=hidden_dim, # entrée: le nbre de valeurs sortant du LSTM
            out_features=3)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim),
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim))

    def forward(self, input):
        lstm_out, _ = self.lstm(
            input.view(len(input), self.seq_length, -1),
            self.hidden)

        y_pred = self.linear(
            lstm_out.view(self.seq_length, len(input), self.hidden_dim)[-1])

        return y_pred

"""#### L'entrainement du modèle"""

def test(mod):
    mod.train(False)
    total_loss, nbatch = 0., 0
    for batch in test_loader:
        sequences, target = batch
        pred_target = mod(sequences)
        loss = loss_ann(target,pred_target) # compter l'erreur seulement sur la composante annuelle, puis loss_cycle et loss_anomalies
        total_loss += loss.item()
        nbatch += 1
    total_loss /= float(nbatch)
    mod.train(True)
    return total_loss

def train(mod, nepochs, learning_rate):

    optim = torch.optim.Adam(mod.parameters(), lr=learning_rate)
    test_loss_vect = np.zeros(nepochs)
    train_loss_vect = np.zeros(nepochs)

    for epoch in range(nepochs):
        mod.reset_hidden_state()
        test_loss = test(mod)
        total_loss, nbatch = 0., 0

        for batch in train_loader:
            sequences, target = batch
            optim.zero_grad()
            pred_target = mod(sequences)
            loss = loss_ann(target,pred_target) # puis loss_cycle et loss_anomalies
            total_loss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()

        total_loss /= float(nbatch)
        test_loss_vect[epoch] = test_loss
        train_loss_vect[epoch] = total_loss
        print(f'Epoch {epoch} train_loss: {total_loss} ; test_loss: {test_loss}')

    print(f'Fin Epoch {epoch} train_loss: {total_loss} ; test_loss: {test_loss}', file=sys.stderr)
    return mod.eval(), train_loss_vect, test_loss_vect

input_dim = 7
hidden_dim = 2
layers = 1
epochs = 2
lr = 0.001 # learning rate

mymodel = LSTM(input_dim, hidden_dim, sequence_length, layers)

trained_model, train_loss_history, test_loss_history = train(mymodel, epochs, lr)

"""#### Courbes loss"""

plt.plot(train_loss_history[0], label='train loss') # erreur seulement sur la composante annuelle
plt.plot(test_loss_history[0], label='test loss')
plt.title('Courbes loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""#### La phase du test"""

print(trained_model)

test_loss_total = test(trained_model)

print("Loss total sur la base de test:", test_loss_total)
