"""Finite Difference Method solver for the 2D heat equation."""

import numpy as np

from .config import Config


def solve_heat_equation(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 2D heat equation using implicit Euler.

    Args:
        cfg: Configuration object

    Returns:
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        t: time points (nt,)
        T: temperature solution (nt, nx, ny)
    """
    # Create grids
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    #######################################################################
    # Oppgave 3.2: Start
    #######################################################################
    T = np.zeros((cfg.nt, cfg.nx, cfg.ny))
    T [0, :, :] = cfg.T_outside
    
    A = _build_matrix(cfg, dx, dy, dt)
    q_on = True
    
    for k in range(cfg.nt-1):
        t_next = t[k+1]
        
        b = _build_rhs(cfg, T[k, :, :], X, Y, dx, dy, dt, t_next, q_on)
        T[k+1, :, :] = np.linalg.solve(A, b).reshape(cfg.nx, cfg.ny)
        
        #Find sensor temperatures
        sensor_indeces = cfg.sensor_locations
        sensor_temp = []
        
        for sx, sy in sensor_indeces: 
            i = np.argmin(np.abs(x - sx))
            j = np.argmin(np.abs(y - sy))
            temp = T[k, i, j]
            sensor_temp.append(temp)
        
        #Turn off heat sorce if temperature exceeds T_max
        max_sensor_temp = np.max(np.array(sensor_temp))
        T_max = cfg.T_max
        
        if (max_sensor_temp < T_max):
            q_on = True
        else:
            q_on = False
            

    #######################################################################
    # Oppgave 3.2: Slutt
    #######################################################################

    return x, y, t, T


def _build_matrix(cfg: Config, dx: float, dy: float, dt: float) -> np.ndarray:
    """Build the implicit Euler system matrix."""
    n = cfg.nx * cfg.ny
    A = np.zeros((n, n))

    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    def idx(i, j):
        return i * cfg.ny + j

    I, J = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")

    # Boundary masks
    left = I == 0
    right = I == cfg.nx - 1
    bottom = J == 0
    top = J == cfg.ny - 1

    # Diagonal entries
    diag = np.full((cfg.nx, cfg.ny), 1 + 2 * rx + 2 * ry)
    diag[left | right] -= rx
    diag[bottom | top] -= ry
    diag[left | right] += rx * cfg.h * dx / cfg.k
    diag[bottom | top] += ry * cfg.h * dy / cfg.k

    p = idx(I, J)
    A[p, p] = diag

    # Off-diagonals
    mask = ~left
    A[idx(I[mask], J[mask]), idx(I[mask] - 1, J[mask])] = -rx

    mask = ~right
    A[idx(I[mask], J[mask]), idx(I[mask] + 1, J[mask])] = -rx

    mask = ~bottom
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] - 1)] = -ry

    mask = ~top
    A[idx(I[mask], J[mask]), idx(I[mask], J[mask] + 1)] = -ry

    return A


def _build_rhs(cfg: Config, T_curr, X, Y, dx, dy, dt, t_next, q_on: bool = True):
    """Build right-hand side for implicit system."""
    rhs = T_curr.copy()

    # Heat source
    if q_on:
        q = np.array(cfg.heat_source(X, Y, t_next))
    else:
        q = 0
    rhs += dt * q

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs[0, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[-1, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[:, 0] += ry * (cfg.h * dy / cfg.k) * bc_term
    rhs[:, -1] += ry * (cfg.h * dy / cfg.k) * bc_term

    return rhs.flatten()
