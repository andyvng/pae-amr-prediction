import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from models import VanillaAutoEncoder, VariationalAutoEncoder
from utils import DatasetFromDir, EarlyStopper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        help='Config file path',
                        type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as config_input:
        config = json.load(config_input)


    train_dataset = DatasetFromDir(config['working_dir'],
                                   config['train_path'])
    val_dataset = DatasetFromDir(config['working_dir'],
                                 config['val_path'])
    test_dataset = DatasetFromDir(config['working_dir'],
                                  config['test_path'])

    print(f"Train:{len(train_dataset)}\tVal:{len(val_dataset)}\tTest:{len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['model'] == 'vanila':
        model = VanillaAutoEncoder(train_dataset.__getitem__(0).shape[0]) \
                               .to(device)
    elif config['model'] == 'variational':
        model = VariationalAutoEncoder(train_dataset.__getitem__(0).shape[0],
                                       config['latent_shape']).to(device) 
    else:
        raise ValueError(f"Model not found! - {config['model']}")

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    stopper = EarlyStopper(config['patient'], config['delta'])

    train_losses = []
    val_losses = []

    for epoch in range(config["num_epochs"]):
        train_losses.append(0)
        for batch_idx, x in enumerate(train_dataloader):
            x = x.to(device)
            if config["model"] == "vanilla":
                x_hat = model(x.float())
                loss = F.binary_cross_entropy(x_hat, x.float(), reduction='sum').to(device)
            elif config["model"] == "variational":
                x_hat, mu, logvar = model(x.float())
                reconstruction_loss = F.binary_cross_entropy(x_hat, x.float(), reduction='sum').to(device)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_loss           
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses[epoch] += loss.item()
        train_losses[epoch] /= len(train_dataloader.dataset)
        val_losses.append(0)

        with torch.no_grad():
            for batch_idx, x in enumerate(val_dataloader):
                x = x.to(device)
                if config["model"] == "vanilla":
                    x_hat = model(x.float())
                    loss = F.binary_cross_entropy(x_hat, x.float(), reduction='sum').to(device)
                elif config["model"] == "variational":
                    x_hat, mu, logvar = model(x.float())
                    reconstruction_loss = F.binary_cross_entropy(x_hat, x.float(), reduction='sum').to(device)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = reconstruction_loss + kl_loss
                val_losses[epoch] += loss.item()
            val_losses[epoch] /= len(val_dataloader.dataset)

        if epoch % 10 == 0:
            print(f'Epoch: {str(epoch).zfill(4)} - Cross-Entropy loss: Train {train_losses[epoch]} - Val {val_losses[epoch]}')

        if stopper.early_stop(val_losses[epoch]):
            print(f"Stopping criteria reached! Stop at epoch {epoch}")
            break

        if epoch % 20 == 0:
            torch.save(model, os.path.join(config["output_dir"], f"{config['suffix']}_{config['model']}_epoch_{epoch}.pt"))
            plt.clf()
            fig = plt.figure(figsize=(12, 5), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(train_losses, lw=3, color='blue', label='Training loss')
            ax.plot(val_losses, lw=3, color='red', label='Validation loss')
            ax.set_title('Cross-entropy loss', size=15)
            ax.set_xlabel('Epoch', size=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.legend()
            plt.savefig(os.path.join(config["output_dir"], f"{config['suffix']}_{config['model']}_epoch_{epoch}.jpg"), dpi=150)

    torch.save(model, os.path.join(config["output_dir"], f"{config['suffix']}_{config['model']}.pt"))

    plt.clf()
    fig = plt.figure(figsize=(12, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_losses, lw=3, color='blue', label='Training loss')
    ax.plot(val_losses, lw=3, color='red', label='Validation loss')
    ax.set_title('Cross-entropy loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend()
    plt.savefig(os.path.join(config["output_dir"], f"{config['suffix']}_{config['model']}.jpg"), dpi=150)

    return                      

if __name__ == "__main__":
    main()