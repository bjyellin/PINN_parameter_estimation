from torch.nn import *
from problem import *
import numpy as np
import torch

def g_true(x, y, problem):
   if problem == 'cont_laplace' or problem == 'simpler_cont_laplace' or problem == 'very_simple':
      return torch.sin(x+y)
   if problem == 'constant_mu':
      return torch.sin(torch.pi*x)*torch.sin(torch.pi*y)

def dirichlet_bdd_loss_func(x,y,model,problem, iteration, num_epochs):
    torch.set_default_dtype(torch.float64)
    
    u, mu = model(torch.cat((x,y), dim=1))
   #  breakpoint()
    g_val = (g_true(x, y, problem))
   #  breakpoint()
    return torch.norm((u-g_val))**2
   #  return torch.tensor(torch.sum((u-g_val)**2),requires_grad=True)
    