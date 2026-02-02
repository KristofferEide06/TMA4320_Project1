"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################

    x, y, t, T_fdm, sensor_data = generate_training_data(cfg) 
    nn_params, losses = train_nn(sensor_data, cfg)
    
    T_pred = predict_grid(nn_params, x, y, t, cfg) 
    
    print("\nGenerating NN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/NN/NN_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/NN/NN_animation.gif"
    )
    
    loss_total, loss_data, loss_ic = losses["total"], losses["data"], losses["ic"]
    epochs = cfg.num_epochs
    epochs_data = np.linspace(0, epochs - 1, epochs)
    
    plt.plot(epochs_data, loss_total, label = "L_total")
    plt.plot(epochs_data, loss_data, label = "L_data")
    plt.plot(epochs_data, loss_ic, label = "L_ic" )
    plt.yscale('log')
    
    plt.xlabel("Epoch")
    plt.ylabel("Log - Loss function value")
    plt.title("Logplot of loss functions")
    
    plt.grid()
    plt.legend()
    plt.savefig("output/NN/NNloss.pdf")
    
    
    
    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
