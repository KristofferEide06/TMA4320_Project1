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
    def step(nn_params : list[tuple[jnp.ndarray, jnp.ndarray]], adam_state : dict):
        """Step-function, calculates necessary parameters for step algorithm 
        
        Args:
            nn_params : Network parameters (list of (W, b) tuples)
            adam_state : Optimizer state dict with moment estimates and step count
            
        Returns:
            nn_params: Corrected network parameters
            adam_state: Updated adam_state dict
            error_total: MSE for the object function
            aux: Tuple containg MSE for the loss functions (loss_data, loss_ic)
        """
            
        def objective_function(nn_params : list[tuple[jnp.ndarray, jnp.ndarray]]):
            """Calcualtes the MSE for the objective_function"""
            
            loss_data = data_loss(nn_params, sensor_data, cfg)
            loss_ic = ic_loss(nn_params, ic_epoch, cfg)
            
            return cfg.lambda_data*loss_data + cfg.lambda_ic*loss_ic, (loss_data, loss_ic)
        
        (error_total, aux), grad_total =  jax.value_and_grad(objective_function, has_aux = True)(nn_params)
        nn_params, adam_state = adam_step(nn_params, grad_total, adam_state, lr = cfg.learning_rate)
        
        return nn_params, adam_state, error_total, aux
    
    #Step algorithm: applies the step function to each epoch to calculates the new nn_params
    for _ in tqdm(range(num_epochs), desc = "Training NN"):
        ic_epoch, key = sample_ic(key, cfg)
        
        nn_params, adam_state, error_total, aux = step(nn_params, adam_state) 
        
        error_data, error_icl = aux
        losses['total'].append(error_total)
        losses['data'].append(error_data)
        losses['ic'].append(error_icl)
        
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
    def step(pinn_params : dict, opt_state : dict, interior_epoch : jnp.ndarray, bc_epoch : jnp.ndarray, ic_epoch : jnp.ndarray):
        """Step-function, calculates loss-gradient and takes step in minimizing direction
        
        Args:
            pinn_params : Network parameters (list of (W, b) tuples and)
            adam_state : Optimizer state dict with moment estimates and step count
            epochs: positions to use for the loss functions
            
        Returns:
            nn_params: Corrected network parameters
            opt_state: Updated adam_state dict
            error_total: MSE for the object function
            aux: Tuple containg MSE for the loss functions (loss_data, loss_ic, loss_bc, loss_ph)
        """
        
        def objective_function(pinn_params : dict):
            """Calculates MSE for objective function"""
            
            nn_params = pinn_params['nn']
            
            loss_data = data_loss(nn_params, sensor_data, cfg)
            loss_ic = ic_loss(nn_params, ic_epoch, cfg)
            loss_bc = bc_loss(pinn_params, bc_epoch, cfg)
            loss_ph = physics_loss(pinn_params, interior_epoch, cfg)
            
            return cfg.lambda_data*loss_data + cfg.lambda_ic*loss_ic + cfg.lambda_bc*loss_bc + cfg.lambda_physics*loss_ph, (loss_data, loss_ic, loss_bc, loss_ph)
        
        (error_total, aux), grad_total =  jax.value_and_grad(objective_function, has_aux = True)(pinn_params)
        pinn_params, opt_state = adam_step(pinn_params, grad_total, opt_state, lr = cfg.learning_rate)
        
        return pinn_params, opt_state, error_total, aux
        
    #Step algorithm - uses the step function to train the network and updates the PINN parameters
    for _ in tqdm(range(num_epochs), desc = "Training PINN"):
        interior_epoch, key = sample_interior(key, cfg)
        ic_epoch, key = sample_ic(key, cfg)
        bc_epoch, key = sample_bc(key, cfg)
        pinn_params, opt_state, error_total, aux = step(pinn_params, opt_state, interior_epoch, ic_epoch, bc_epoch)
        
        error_data, error_icl, error_bc, error_ph = aux
        
        losses['total'].append(error_total)
        losses['data'].append(error_data)
        losses['ic'].append(error_icl)
        losses['bc'].append(error_bc)
        losses['physics'].append(error_ph)


    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################
    alpha = jnp.exp(pinn_params['log_alpha'])
    P = jnp.exp(pinn_params['log_power'])
    k = jnp.exp(pinn_params['log_k'])
    h = jnp.exp(pinn_params['log_h'])
    
    print(f'Alpha: {alpha}')
    print(f'Power : {P}')
    print(f'(k : {k})')
    print(f'h : {h}')
    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
