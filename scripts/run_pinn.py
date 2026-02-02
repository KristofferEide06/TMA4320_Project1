"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg) 
    pinn_params, losses = train_pinn(sensor_data, cfg)
    
    T_pred = predict_grid(pinn_params['nn'], x, y, t, cfg) 
    
    print("\nGenerating PINN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/PINN/PINN_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="PINN", save_path="output/PINN/PINN_animation.gif"
    )
    
    loss_total, loss_data, loss_ic, loss_bc, loss_ph = losses['total'], losses['data'], losses['ic'], losses['bc'], losses['physics']
    epochs = cfg.num_epochs
    epochs_data = np.linspace(0, epochs - 1, epochs)
    
    print(len(epochs_data))
    print(len(loss_total))
    print(len(loss_data))
    print(len(loss_ic))
    
    plt.plot(epochs_data, loss_total, label = "L_total")
    plt.plot(epochs_data, loss_data, label = "L_data")
    plt.plot(epochs_data, loss_ic, label = "L_ic" )
    plt.plot(epochs_data, loss_bc, label = "L_bc" )
    plt.plot(epochs_data, loss_ph, label = "L_ph" )
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss function value")
    plt.title("Object functions plotted by epoch number")
    
    plt.grid()
    plt.legend()
    plt.savefig("output/PINN/PINN_loss.pdf")
    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
