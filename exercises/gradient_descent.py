import numpy as np


def steepest_descent(x_k, alpha_k, grad_f_k):
    """
    One iteration of simple steepest descent, where the
    descent direction is D^k = I, k = 0, 1, ...
    :param x_k: value of x at time step k, as an np.array
    :param alpha_k: value of step-size alpha at timestep k
    :param grad_f_k: Jacobian of f at x_k
    :return: x_k+1: value of x at time step k+1, as an np.array
    """
    return x_k - alpha_k * grad_f_k
