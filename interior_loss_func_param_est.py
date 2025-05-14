import torch
import numpy as np
from torch.nn import *
from problem import *
# from ADR_fem_driver import *
# from fem import data_points




#Define the loss function on the interior of the domain to calculate
#how closely the learned function solves the PDE you are trying to solve
def interior_pde_loss_func(x,y, samples_f, model,problem):
    torch.set_default_dtype(torch.float64)
    u, mu  = model(torch.cat((x.double(),y.double()), dim=1))
    # breakpoint()
    # Ensure mu_field is evaluated as a scalar field for each input point
    # If mu_field is a neural network, pass the input coordinates (x, y) to it
    # breakpoint()
    

    

    # print("mu_field norm is", torch.norm(mu_field))
    # breakpoint()
    n = len(u)

    torch.autograd.set_detect_anomaly(True)

    # u2 = u.clone()

    u_x = torch.autograd.grad(u, x,
                            grad_outputs=torch.ones_like(u),
                            retain_graph=True,
                            create_graph=True,
                            allow_unused=True)[0]

    u_x_clone = u_x.clone()

    u_xx = torch.autograd.grad(
        u_x_clone, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,

    )[0]

    u_xx_clone = u_xx.clone()

    u_y = torch.autograd.grad(
        u, y,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]

    # Multiply the scalar field mu_field_values by u_x
    mu_u_x = mu * u_x
    mu_u_y = mu * u_y

    u_yy = torch.autograd.grad(
        u_y, y,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_yy_clone = u_yy.clone()
    # breakpoint()
    # mu_u_x = mu_field*u_x
    # mu_u_y = mu_field*u_y
    # breakpoint()
    div_mu_grad_u_x = torch.autograd.grad(mu_u_x, x,
                                      grad_outputs=torch.ones_like(mu_u_x),
                                      retain_graph=True,
                                      create_graph=True,
                                      allow_unused=True)[0]
    
    div_mu_grad_u_y = torch.autograd.grad(mu_u_y, y,
                                      grad_outputs=torch.ones_like(mu_u_y),
                                      retain_graph=True,
                                      create_graph=True,
                                      allow_unused=True)[0]
    
    #THIS NEEDS TO INVOLVE MU
    if problem == 'constant_mu':
        # breakpoint()
        return torch.norm(mu*(-u_xx_clone-u_yy_clone)-samples_f)**2

    if problem == 'cont_laplace' or problem == 'simpler_cont_laplace' or problem == 'very_simple':
        return torch.norm(-(div_mu_grad_u_x+div_mu_grad_u_y)-samples_f)**2
    

    