"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################
    
    # Update the nn_params and losses dictionary
    from tqdm import tqdm
    
    
    num_epochs = cfg.num_epochs
    adam_state = init_adam(nn_params)
    
    @jax.jit
    def step(nn_params, adam_state):
        def objective_function(nn_params):
            
            loss_data = data_loss(nn_params, sensor_data, cfg)
            loss_ic = ic_loss(nn_params, ic_epoch, cfg)
            
            return cfg.lambda_data*loss_data + cfg.lambda_ic*loss_ic, (loss_data, loss_ic)
        
        (error_total, aux), grad_total =  jax.value_and_grad(objective_function, has_aux = True)(nn_params)
        nn_params, adam_state = adam_step(nn_params, grad_total, adam_state, lr = cfg.learning_rate)
        
        return nn_params, adam_state, error_total, aux
    
    
    for _ in tqdm(range(num_epochs), desc = "Training NN"):
        ic_epoch, key = sample_ic(key, cfg)
        
        nn_params, adam_state, error_total, aux = step(nn_params, adam_state) 
        
        error_data, error_icl = aux
        losses["total"].append(error_total)
        losses["data"].append(error_data)
        losses["ic"].append(error_icl)
        
    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################

    # Update the nn_params and losses dictionary
    num_epochs = cfg.num_epochs
    
    from tqdm import tqdm
    
    @jax.jit
    def step(pinn_params, opt_state, interior_epoch, bc_epoch, ic_epoch):
        
        def objective_function(pinn_params):
            nn_params = pinn_params['nn']
            
            loss_data = data_loss(nn_params, sensor_data, cfg)
            loss_ic = ic_loss(nn_params, ic_epoch, cfg)
            loss_bc = bc_loss(pinn_params, bc_epoch, cfg)
            loss_ph = physics_loss(pinn_params, interior_epoch, cfg)
            
            return cfg.lambda_data*loss_data + cfg.lambda_ic*loss_ic + cfg.lambda_bc*loss_bc + cfg.lambda_physics*loss_ph, (loss_data, loss_ic, loss_bc, loss_ph)
        
        (error_total, aux), grad_total =  jax.value_and_grad(objective_function, has_aux = True)(pinn_params)
        
        pinn_params, opt_state = adam_step(pinn_params, grad_total, opt_state, lr = cfg.learning_rate)
        
        return pinn_params, opt_state, error_total, aux
        
    
    for _ in tqdm(range(num_epochs), desc = "Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)
        pinn_params, opt_state, error_total, aux = step(pinn_params, opt_state, interior_epoch, ic_epoch, bc_epoch)
        
        error_data, error_icl, error_bc, error_ph = aux
        
        losses["total"].append(error_total)
        losses["data"].append(error_data)
        losses["ic"].append(error_icl)
        losses["bc"].append(error_bc)
        losses["physics"].append(error_ph)


    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
