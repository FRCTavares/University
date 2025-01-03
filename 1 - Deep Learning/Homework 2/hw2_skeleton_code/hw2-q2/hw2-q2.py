#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN_Old(nn.Module):
    def __init__(self, dropout_prob=0.1, conv_bias=True):
        super(CNN_Old, self).__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512

        # Convolutional blocks WITHOUT batch normalization
        self.block1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_prob)
        )

        # Flatten instead of global average pooling
        # MLP part
        self.fc1 = nn.Linear(channels[3] * (48 // 2 // 2 // 2) * (48 // 2 // 2 // 2), fc1_out_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(fc2_out_dim, 6)  # Example: 6 classes

    def forward(self, x):
        # x shape: (B, 3, 48, 48) if already reshaped
        x = x.reshape(x.shape[0], 3, 48, -1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=None,
            maxpool=True,
            batch_norm=True,
            dropout=0.0
        ):
        super().__init__()

        # Q2.1. Initialize convolution, maxpool, activation and dropout layers 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Q2.2 Initialize batchnorm layer 
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_layer = nn.Dropout(0.1)
        
        #raise NotImplementedError

    def forward(self, x):
        # input for convolution is [b, c, w, h]
        
        # Implement execution of layers in right order
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_layer(x)

        #raise NotImplementedError

        return x


class CNN(nn.Module):
    def __init__(self, dropout_prob, maxpool=True, batch_norm=True, conv_bias=True):
        super(CNN, self).__init__()
        channels = [3, 32, 64, 128]
        fc1_out_dim = 1024
        fc2_out_dim = 512
        self.maxpool = maxpool
        self.batch_norm = batch_norm

        # Initialize convolutional blocks
        self.block1 = ConvBlock(channels[0], channels[1], 3, batch_norm=batch_norm)
        self.block2 = ConvBlock(channels[1], channels[2], 3, batch_norm=batch_norm)
        self.block3 = ConvBlock(channels[2], channels[3], 3, batch_norm=batch_norm)

        # Global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # BarchNorm in MLP block (Identify if not batch_norm)
        self.bn_mlp = nn.BatchNorm1d(channels[3]) if batch_norm else nn.Identity()
        
        # Initialize layers for the MLP block
        # For Q2.2 initalize batch normalization
        self.fc1 = nn.Linear(channels[3], fc1_out_dim)
        self.bn_fc1 = nn.BatchNorm1d(fc1_out_dim) if batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
        self.bn_fc2 = nn.BatchNorm1d(fc2_out_dim) if batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(fc2_out_dim, 6)  # Example: 6 classes

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 48, -1)

        # Implement execution of convolutional blocks 
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # For Q2.2 implement global averag pooling
        # Global average pooling
        x = self.global_avg_pool(x) # (B, 128, 1, 1)
        x = x.squeeze(-1).squeeze(-1) # (B, 128)
        
        # Flattent output of the last conv block
        #x = x.view(x.size(0), -1)

        # Implement MLP part
        # BatchNorm in MLP 
        x = self.bn_mlp(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
 

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, return_scores=True):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)

    if return_scores:
        return predicted_labels, scores
    else:
        return predicted_labels


def evaluate(model, X, y, criterion=None):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    with torch.no_grad():
        y_hat, scores = predict(model, X, return_scores=True)
        loss = criterion(scores, y)
        n_correct = (y == y_hat).sum().item()
        n_possible = float(y.shape[0])

    return n_correct / n_possible, loss


def plot(epochs, plottable, ylabel='', name=''):
    plt.figure()#plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    # Sum the number of elements for all trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #raise NotImplementedError


def plot_file_name_sufix(opt, exlude):
    """
    opt : options from argument parser
    exlude : set of variable names to exlude from the sufix (e.g. "device")

    """
    return '-'.join([str(value) for name, value in vars(opt).items() if name not in exlude])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=40, type=int)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    parser.add_argument('-no_batch_norm', action='store_true')
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz')
    parser.add_argument('-device', choices=['cpu', 'cuda', 'mps'], default='cpu')

    # New flag to select the old or new model
    parser.add_argument('-old_model', action='store_true',
                        help="Use old model without BN & global avg pooling")

    opt = parser.parse_args()
    utils.configure_seed(seed=42)

    data = utils.load_dataset(data_path=opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X.to(opt.device), dataset.dev_y.to(opt.device)
    test_X, test_y = dataset.test_X.to(opt.device), dataset.test_y.to(opt.device)

    # Pick which model
    if opt.old_model:
        print("Using old model (no BN, flatten).")
        model = CNN_Old(dropout_prob=opt.dropout).to(opt.device)
    else:
        print("Using new model (BN, global avg pooling).")
        model = CNN(
            dropout_prob=opt.dropout,
            maxpool=not opt.no_maxpool,
            batch_norm=not opt.no_batch_norm
        ).to(opt.device)

    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)
    criterion = nn.NLLLoss()

    # Training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('\nTraining epoch {}'.format(ii))
        model.train()
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(opt.device)
            y_batch = y_batch.to(opt.device)
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)
        train_mean_losses.append(mean_loss)

        val_acc, val_loss = evaluate(model, dev_X, dev_y, criterion)
        valid_accs.append(val_acc)
        print("Valid loss: %.4f" % val_loss)
        print('Valid acc: %.4f' % val_acc)

    test_acc, _ = evaluate(model, test_X, test_y, criterion)
    print('Final Test acc: %.4f' % test_acc)

    # Plot
    sufix = plot_file_name_sufix(opt, exlude={'data_path', 'device'})
    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-3-train-loss-{}-{:.2f}'.format(sufix, test_acc*100))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-3-valid-accuracy-{}-{:.2f}'.format(sufix, test_acc*100))

    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
