
import torch
import torch.nn as nn
# from fenics import *
# from dolfin import *
from interior_loss_func_param_est import interior_pde_loss_func
from dirichlet_boundary_loss_func_param_est import dirichlet_bdd_loss_func
# from ADR_GALS import adr_gals 

# from ADR_GALS import *
import timeit
#FIGURE OUT HOW TO IMPORT THE gals_solution.csv file I just created so I don't have to run FEM every time

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 
import meshio

import warnings

plt.rcParams.update({
    'axes.labelsize': 14,       # for xlabel and ylabel
    'axes.titlesize': 15,       # for the title
    'xtick.labelsize': 12,      # x-axis tick labels
    'ytick.labelsize': 12,      # y-axis tick labels
    'legend.fontsize': 12,      # legend
    'font.size': 12             # base font size (affects others if specific keys aren't set)
})



# def interior_pde_loss_func(x,y, samples_f, model, problem):


    
#     u = model(torch.cat((x,y), dim=1))
#     # breakpoint()
#     n = len(u)

#     torch.autograd.set_detect_anomaly(True)

#     # u2 = u.clone()

#     u_pred, mu_pred = model(torch.cat((x,y), dim=1))
#     # print("norm(mu_pred) is", torch.norm(mu_pred))

#     u_x = torch.autograd.grad(u_pred, x,
#                             grad_outputs=torch.ones_like(u_pred),
#                             retain_graph=True,
#                             create_graph=True,
#                             allow_unused=True)[0]

#     u_x_clone = u_x.clone()

#     u_xx = torch.autograd.grad(
#         u_x_clone, x,
#         grad_outputs=torch.ones_like(u_pred),
#         retain_graph=True,
#         create_graph=True,

#     )[0]

#     u_xx_clone = u_xx.clone()

#     u_y = torch.autograd.grad(
#         u_pred, y,
#         grad_outputs=torch.ones_like(u_pred),
#         retain_graph=True,
#         create_graph=True,
#         allow_unused=True
#     )[0]

#     u_y_clone = u_y.clone()

#     u_yy = torch.autograd.grad(
#         u_y_clone, y,
#         grad_outputs=torch.ones_like(u_pred),
#         retain_graph=True,
#         create_graph=True
#     )[0]

#     u_yy_clone = u_yy.clone()

#     if problem == 'cont_laplace':
#         return torch.norm(-mu_pred*(u_xx+u_yy)-samples_f)



def mu_true(x, y, problem):
    if problem == 'cont_laplace':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return x**2+y**2+1
    
    
    if problem == 'simpler_cont_laplace':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return (x**2+y**2)

    if problem == 'very_simple':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return 2*torch.ones_like(x)
    
    if problem == 'constant_mu':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return 1*torch.ones_like(x)

def u_true(x, y, problem):
    if isinstance(x,np.ndarray):
        x = torch.tensor(x, dtype=torch.float64)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float64)
    if problem == 'cont_laplace' or problem=='simpler_cont_laplace' or problem=='very_simple':
        # breakpoint()
        return torch.sin(torch.tensor(x)+torch.tensor(y))
    if problem == 'constant_mu':
        return torch.sin(torch.pi*torch.tensor(x))*torch.sin(torch.pi*torch.tensor(y))
        
def f_true(x, y, problem):
    if isinstance(x,np.ndarray):
        x = torch.tensor(x, dtype=torch.float64)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float64)
    if problem == 'cont_laplace':
        return -((2*torch.tensor(x)**2+torch.tensor(y)**2+1)*(torch.sin(torch.tensor(x)+torch.tensor(y)))+2*(torch.tensor(x)+torch.tensor(y))*torch.cos(torch.tensor(x)+torch.tensor(y)))#2*(torch.tensor(x)+torch.tensor(y))*torch.cos(torch.tensor(x)+torch.tensor(y))-2*(torch.tensor(x)**2+torch.tensor(y)**2+1)*torch.sin(torch.tensor(x)+torch.tensor(y))#(2*torch.tensor(x)**2+2*torch.tensor(y)**2+2)*(torch.sin(torch.tensor(x)+torch.tensor(y)))

    if problem == 'simpler_cont_laplace':
        return -4*(torch.tensor(x)+torch.tensor(y))*(torch.sin(torch.tensor(x)+torch.tensor(y)))-4*(torch.tensor(x)**2+torch.tensor(y)**2)*torch.cos(torch.tensor(x)+torch.tensor(y))

    if problem == 'very_simple':
        return 2*torch.sin(torch.tensor(x)+torch.tensor(y))
    
    if problem == 'constant_mu':
        return 2*torch.pi**2*torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
    

import torch

def compute_hessian_norm(u, x, model):
    """
    Computes the squared Frobenius norm of the Hessian of u w.r.t. x.

    Args:
        u: tensor of shape [N, 1], network output
        x: tensor of shape [N, d], input coordinates

    Returns:
        scalar tensor: sum over N of the Frobenius norm squared of the Hessian
    """
    N, d = x.shape
    x = x.clone().detach().requires_grad_(True)
    u_tensor, mu_tensor = model(x)
    # breakpoint()
    # u.requires_grad_()
    # breakpoint()
    hessian_norm = 0.0
    # breakpoint()
    for i in range(d):
        # breakpoint()
        grads = torch.autograd.grad(u_tensor, x, grad_outputs=torch.ones_like(u_tensor), 
                                       create_graph=True, retain_graph=True, allow_unused=True)[0]
        # breakpoint()
        grad_u_i = grads[:, i]
        # breakpoint()
        for j in range(d):
            grads2 = torch.autograd.grad(grad_u_i, x, grad_outputs=torch.ones_like(grad_u_i), 
                                            create_graph=True, retain_graph=True)[0]
            grad_u_ij = grads2[:, j]
            
            hessian_norm += torch.sum(grad_u_ij ** 2)
    # print("hessian norm is: ", hessian_norm)    
    return hessian_norm / N  # optional: average over batch

def compute_laplacian_norm(model, x_input):
    # Ensure x_input tracks gradients
    x_input = x_input.clone().detach().requires_grad_(True)
    u_pred, *_ = model(x_input)  # assumes model returns (u, ...) tuple

    N, d = x_input.shape
    laplacian = 0.0

    # First derivatives ∂u / ∂x_j
    grads = torch.autograd.grad(u_pred, x_input, grad_outputs=torch.ones_like(u_pred),
                                create_graph=True, retain_graph=True)[0]  # [N, d]

    # Second derivatives ∂²u / ∂x_j²
    for j in range(d):
        grad_u_j = grads[:, j]
        grad2 = torch.autograd.grad(grad_u_j, x_input, grad_outputs=torch.ones_like(grad_u_j),
                                    create_graph=True, retain_graph=True)[0]
        laplacian += grad2[:, j]  # only the diagonal terms (u_xx, u_yy, ...)

    # Squared L2 norm of the Laplacian over batch
    laplacian_norm = torch.sum(laplacian ** 2) / N
    return laplacian_norm


#mesh size is the size of the FEM mesh
#num_fem_points is the number of points sampled in each direction from the FEM solution 
def estimate_u(problem, step, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration, phys_weight):
    print("Starting state estimation")
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            torch.set_default_dtype(torch.float64)
            super(NeuralNetwork, self).__init__()
            

            self.flatten = torch.nn.Flatten()


            # Learnable weights (log sigma^2 form to stabilize optimization)
            # self.log_sigma_data = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=True)  # For data loss (start with smaller weight)
            # self.log_sigma_boundary = torch.nn.Parameter(torch.tensor(0), requires_grad=True)   # For boundary loss (start with smaller weight)


            #Input layer is 2 dimensional because I have (x,y) information in data 
            #Output layer is 1 dimensional because I want to output the temperature
            #at that particular point 


            # Increase the network capacity by adding more layers and neurons
            self.u_net = nn.Sequential(
                nn.Linear(2, 20),
                nn.Tanh(),
                nn.Linear(20,20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )

            self.mu_net = torch.nn.Sequential(
                torch.nn.Linear(2,5),
                torch.nn.Softplus(),
                torch.nn.Linear(5,5),
                torch.nn.Softplus(),
                torch.nn.Linear(5,5),
                torch.nn.Softplus(),
                torch.nn.Linear(5,1),
                torch.nn.Softplus()  # Ensures positive output
            )

        def initialize_weights(self, seed=42):
            torch.manual_seed(seed)
            # breakpoint()
            if int(training_loop) == 0:
                for layer in self.u_net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)  # Use Xavier initialization for Tanh
                        nn.init.zeros_(layer.bias)  # Initialize biases to zero
            
                for layer in self.mu_net:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)  # Use Xavier initialization for Tanh
                        nn.init.zeros_(layer.bias)  # Initialize biases to zero

                return self
            
            if int(training_loop) >= 1:
                
                print("Now we're setting parameters based on last run")
                import os
                
                if os.path.exists("neural_network_for_u_params.pth"):
                    checkpt = torch.load("neural_network_for_u_params.pth", map_location=lambda storage, loc: storage)
                    self.load_state_dict(checkpt)
                else:
                    print("File 'neural_network_for_u_params.pth' not found. Starting with default initialization.")
                    checkpt = None
                
                print("Just set the weights from the previous training")
                # for key in checkpt["state_dict"].keys():
                #     print(key)
                
                # breakpoint()
                
                # breakpoint()
                print("Just initialized weights")
                print("Num real data points: ", num_real_data_points)

                return self


        def forward(self, x):
            # print(">>>> Inside forward pass <<<<<<<<")
            #x = self.flatten(x)
            torch.set_default_dtype(torch.float64)
            # print("x dtype is ",x.dtype)
            u = self.u_net(x)
            mu = self.mu_net(x)
            # sigma_data = torch.exp(self.log_sigma_data)
            # sigma_boundary = torch.exp(self.log_sigma_boundary)
            
            return u, mu #, sigma_boundary
        
        def get_mu(self, x):
            return self.mu_net(x)
    
    criterion = torch.nn.MSELoss() # Choose appropriate loss for your task
    
    if problem == 'adr':
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1
        # breakpoint()
    
    else:
        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1

    def generate_boundary(num_points):
        # breakpoint()
        edges = [
                np.linspace([x_min, y_min], [x_max, y_min], int(num_points // 4)),  # bottom edge
                np.linspace([x_max, y_min], [x_max, y_max], int(num_points // 4)),   # right edge
                np.linspace([x_max, y_max], [x_min, y_max], int(num_points // 4)),   # top edge
                np.linspace([x_min, y_max], [x_min, y_min], int(num_points // 4))   # left edge
            ]
        
        # Concatenate all edge points
        points = np.vstack(edges)

        return points

    # Generate points on the boundary and interior for the boundary, interior, and data fit losses
    #Generate points on the boundary
    boundary_points = torch.tensor(generate_boundary(num_boundary_points))
    # breakpoint()
    x_boundary_points = boundary_points[:,0].reshape(-1,1)
    y_boundary_points = boundary_points[:,1].reshape(-1,1)

    #Generate points on interior for physics loss and data fit
    if num_phys_points != 0:
        x_phys = torch.linspace(x_min,x_max,num_phys_points) #We'll take the interior of these to get the points where we calculate the physics loss
        y_phys = torch.linspace(y_min,y_max,num_phys_points)

        #Create a meshgrid for all of the interior points (for physics and data)
        X_phys_with_bdd, Y_phys_with_bdd = torch.meshgrid(x_phys, y_phys)

        all_phys_points_with_bdd = torch.stack((X_phys_with_bdd.flatten(), Y_phys_with_bdd.flatten()), dim=-1)

        interior_mask_phys = ~((all_phys_points_with_bdd[:, 0] == x_max) | (all_phys_points_with_bdd[:, 0] == x_min) | 
                        (all_phys_points_with_bdd[:, 1] == y_max) | (all_phys_points_with_bdd[:, 1] == y_min))
        
        
        phys_points = all_phys_points_with_bdd[interior_mask_phys].requires_grad_()
        phys_points_x = phys_points[:,0].reshape(-1,1).requires_grad_()
        phys_points_y = phys_points[:,1].reshape(-1,1).requires_grad_()

    dx = (x_max - x_min)/(num_real_data_points+1)
    dy = (y_max - y_min)/(num_real_data_points+1)

    x_data = torch.linspace(x_min,x_max,num_real_data_points)
    y_data = torch.linspace(y_min,y_max,num_real_data_points)
    # breakpoint()
    x_test = torch.linspace(x_min,x_max,200)
    y_test = torch.linspace(y_min,y_max,200)

    #Create a meshgrid for the data fit loss
    X_data_with_bdd, Y_data_with_bdd = torch.meshgrid(x_data, y_data)

    all_data_points_with_bdd = torch.stack((X_data_with_bdd.flatten(), Y_data_with_bdd.flatten()), dim=-1)


    interior_mask_real_data = ~((all_data_points_with_bdd[:, 0] == x_max) | (all_data_points_with_bdd[:, 0] == x_min) | 
                    (all_data_points_with_bdd[:, 1] == y_max) | (all_data_points_with_bdd[:, 1] == y_min))
    
    
    real_data_points = all_data_points_with_bdd[interior_mask_real_data]
   
    real_data_points_x = real_data_points[:,0].reshape(-1,1).requires_grad_().double()
    real_data_points_y = real_data_points[:,1].reshape(-1,1).requires_grad_().double()
    
    # breakpoint()
    if len(real_data_points_x) != 0:
        dx = (torch.max(real_data_points_x) - torch.min(real_data_points_x))/len(real_data_points_x)
        dy = (torch.max(real_data_points_y) - torch.min(real_data_points_y))/len(real_data_points_y)
        area_element = dx*dy

    real_data_points_x_np = real_data_points_x.detach().numpy()
    real_data_points_y_np = real_data_points_y.detach().numpy()

    np.savetxt(f"real_data_points_x_{training_loop}.txt", real_data_points_x_np)
    np.savetxt(f"real_data_points_y_{training_loop}.txt", real_data_points_y_np)
    
    # breakpoint()



    losses = []
    boundary_losses = []
    data_fit_losses = []
    physics_losses = []
    adr_boundary_losses = []
    adr_phys_losses = []
    adr_data_fit_losses = []
    total_losses = []
    print('length of total losses: (just making sure that it starts over every new training)', len(total_losses))

    model_for_u = NeuralNetwork().initialize_weights()
    # breakpoint()
    optimizer = torch.optim.Adam(model_for_u.parameters(), lr=1e-4)  # Slightly increased learning rate for faster convergence
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    

    epoch_state = {'epoch': 0, 'counter': 0}
    stop_training = False
    def closure():
        nonlocal stop_training
        optimizer.zero_grad()
        u_pred, mu_pred = model_for_u(torch.cat((real_data_points_x,real_data_points_y), dim=1))
        # print("Inside closure: u_pred.requires_grad:", u_pred.requires_grad)
        # print("Inside closure: mu_pred.requires_grad:", mu_pred.requires_grad)
        # breakpoint()
        epoch = epoch_state['epoch']
        epoch_state['counter'] += 1
        # print("Sampling true data: ", sampling_true_data)
        torch.set_default_dtype(torch.float64)

        from scipy.interpolate import griddata
            
        #Compute and normalize losses
        if num_real_data_points != 0:
            # breakpoint()
            # u_pred, mu_pred, sigma_boundary = model_for_u(torch.cat((real_data_points_x,real_data_points_y), dim=1))
            # breakpoint()
            noise = torch.normal(0,0.1,size=(real_data_points_x.shape[0],1))*0
            # breakpoint()
            data_fit_loss = criterion(u_true(real_data_points_x,real_data_points_y,problem)+noise, u_pred)
            # breakpoint()

            pointwise_error = (u_pred - u_true(real_data_points_x,real_data_points_y,problem))**2
            weighted_error = pointwise_error * area_element
            integrated_data_fit_loss = torch.sum(weighted_error)

            # breakpoint()
            # print("data fit loss: ", data_fit_loss)
            # print("Data fit loss: ", data_fit_loss)
            # breakpoint()
            data_fit_loss = data_fit_loss#integrated_data_fit_loss#data_fit_loss/real_data_points_x.shape[0]
            # breakpoint()
            data_fit_losses.append(data_fit_loss.item())


        samples_f = f_true(phys_points_x, phys_points_y, problem)

        residual = interior_pde_loss_func(phys_points_x, phys_points_y, samples_f, model_for_u, problem)/phys_points_x.shape[0]#NORMALIZE 

        # residual.requires_grad_()

        # physics_losses.append(residual.detach().item())
        # breakpoint()
        boundary_loss = dirichlet_bdd_loss_func(boundary_points[:,0].reshape(-1,1), boundary_points[:,1].reshape(-1,1), model_for_u, problem, iteration, num_epochs)
        # print("boundary loss: ", boundary_loss)
        # breakpoint()
        # print("boundary loss: ", boundary_loss)
        boundary_loss = boundary_loss/(boundary_points[:,0].shape[0])
        boundary_losses.append(boundary_loss)

        # breakpoint()

        #Compute NTK-based weights
        losses = {"boundary": boundary_loss, "data_fit":data_fit_loss}
        
        #Compute the total loss 
        # sigma_data = torch.exp(model_for_u.log_sigma_data)
        # sigma_bdd = torch.exp(model_for_u.log_sigma_boundary)

        # regularization_term = 0.0
        # Access the network learned for mu
        # mu_network = model.mu_net
        # print("Learned network for mu:", mu_network)
        
        # for param in mu_network.parameters():
        #     regularization_term += torch.sum(param**2)

        # lambda_reg = 0.01
        
        # if epoch < int(num_epochs):
        #     total_loss =  sigma_data*normalized_interior_mismatch + sigma_bdd*boundary_loss # + residual+ lambda_reg*regularization_term
        
        # dummy_loss = torch.sum(mu_pred) * 0
        # Forward pass
        # x_test = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
        # y_test = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
        # inputs = torch.cat((x_test, y_test), dim=1)

        # u_pred, mu_pred, sigma_data, sigma_bdd = model_for_u(inputs)

        # # Dummy loss using only u_pred and mu_pred
        # dummy_loss = (u_pred**2).sum() + (mu_pred**2).sum()
        # dummy_loss.backward()

        # print("\n==== GRADIENT CHECK (Dummy Loss) ====")
        # for name, param in model_for_u.named_parameters():
        #     if param.grad is not None:
        #         print(f"✅ Dummy Loss: Gradient for {name}: {param.grad.norm()}")
        #     else:
        #         print(f"❌ Dummy Loss: No gradient for {name}")
        # breakpoint()
        epsilon = 1e-8
        mu_loss = epsilon * torch.sum(mu_pred**2)
        # breakpoint()
        # print("u_pred.requires_grad:", u_pred.requires_grad)
        # print("mu_pred.requires_grad:", mu_pred.requires_grad)
        # breakpoint()
        # total_loss = torch.exp(model_for_u.log_sigma_data) * data_fit_loss + torch.exp(model_for_u.log_sigma_boundary)*boundary_loss + mu_loss
        
        # Compute Hessian regularization
        hessian_reg = compute_hessian_norm(u_pred, torch.cat((real_data_points_x, real_data_points_y), dim=1), model_for_u)
        laplacian_reg = compute_laplacian_norm(model_for_u, torch.cat((real_data_points_x, real_data_points_y), dim=1))
        # breakpoint()
        lambda_L = 10 # Regularization for Hessian
        total_loss =  data_fit_loss + phys_weight*residual#+ 10 * boundary_loss #+ lambda_L*laplacian_reg #lambda_L*laplacian_reg#+REGULARIZATION (FRO NORM OF HESSIAN) ON SECOND DERIVATIVE of u
        # breakpoint()
        # breakpoint()
        # breakpoint()
        # breakpoint()
        # for name, param in model_for_u.named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}")
        #     else:
        #         print("This is very bad news")
        #         print(f"No gradient for {name}")
        # breakpoint()
        # print("residual, even though we're not training on the residual: ", residual)
        
            
        # elif epoch==int(num_epochs/2):
        #     #break
        #     total_loss =  sigma_data*normalized_interior_mismatch + sigma_bdd*boundary_loss + residual
        #     print("Just switched to PDE fitting to estimate the parameter")
        #     # breakpoint()
        # else:
        #     # print("got into the else statement")
        #     total_loss =  sigma_data*normalized_interior_mismatch + sigma_bdd*boundary_loss + residual 
        
        # if isinstance(optimizer, torch.optim.LBFGS) and epoch % 10 == 0:
        #     print("residual: ", residual, "boundary_loss: ", boundary_loss, "data_fit_loss: ", data_fit_loss)
        
        if len(total_losses) > 10 and abs(total_losses[-10] - total_losses[-1]) < 1e-8:
            # print("Losses are not changing significantly. Stopping training early.")
            stop_training = True
            return loss
        
        #Back propagate
        total_loss.backward(retain_graph=True)
        # for name, param in model_for_u.named_parameters():
        #     if param.grad is not None:
        #         print(f"After backward: Gradient for {name}: {param.grad.norm()}")
        #     else:
        #         print(f"After backward: No gradient for {name}. Check if the parameter is being used in the computation graph.")
        # breakpoint()
        
        epoch_state['counter'] += 1
        
        return total_loss

    #Training loop
    
    #Also try the cutoff at a lower number like 5 epochs to see if it runs past that 
    adam_lbfgs_cutoff = int(num_epochs/2)  # Allow Adam to train longer before switching to LBFGS
    tic = timeit.default_timer()
    # breakpoint()
    

    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name}: {param.grad.norm()}")
    #     else:
    #         print("This is very bad news")
    #         print(f"No gradient for {name}")

    # Print all model_for_uparameters
    for name, param in model_for_u.named_parameters():
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")

        # Add debugging code to monitor training:
    # for name, param in model_for_u.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name}: {param.grad.norm()}")
    #     else:
    #         print(f"No gradient for {name}. Check if the parameter is being used in the computation graph.")

    # Print loss values during training:
    # print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Check if the network outputs are non-zero:
    # with torch.no_grad():
    # test_input = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    # u_pred, mu_pred, sigma_data, sigma_bdd = model_for_u(test_input)
    # print(f"Test input: {test_input}, u_pred: {u_pred}, mu_pred: {mu_pred}")
    
    # breakpoint()
    threshold = 1000 #Threshold for where to check if losses are close and exit loop
    
    same_loss_count = 0
    
    #Training Loop for u
    switching_to_lbfgs_count = 0 
    # breakpoint()
    for epoch in range(num_epochs):
        epoch_state['epoch'] = epoch
        if len(total_losses) >= 50 and epoch < adam_lbfgs_cutoff:
            if abs(total_losses[-30]-total_losses[-1]) < 10**-4:
                # print("Switching to bfgs sooner than expected because Adam did all it could")
                if switching_to_lbfgs_count == 0:
                    print("Switching to LBFGS")
                    switching_to_lbfgs_count = 1
                optimizer =  torch.optim.LBFGS(model_for_u.parameters(), max_iter=20, line_search_fn='strong_wolfe')

        #Try switching optimiziers during training
        elif epoch == adam_lbfgs_cutoff:
            print("Just switched to LBFGS")
            optimizer =  torch.optim.LBFGS(model_for_u.parameters(), max_iter=20, line_search_fn='strong_wolfe')

        

        # if len(total_losses)>30:
        #     # print("Difference between 30th to last and last losses", (abs(total_losses[-30]-total_losses[-1])))
        #     if (abs(total_losses[-30]-total_losses[-1])<1e-5):
        #         print("Stopping training because losses aren't changing")
        #         break
        

        #Conditional gradient clipping
        # if isinstance(optimizer, torch.optim.Adam):
        #     torch.nn.utils.clip_grad_norm_(model_for_u.parameters(), max_norm=1.0)
 
        
        loss=optimizer.step(closure)

        prev_loss = loss

        # if epoch > threshold:
        #     print("loss: ", loss)
        #     print("prev_loss: ", prev_loss)
        #     print("Difference between subsequent losses: ",abs((loss-prev_loss).item()):
        #     if prev_loss is not None and abs((loss-prev_loss).item())<1e-8:
        #         print(abs(loss-prev_loss).item())
        #         same_loss_count += 1
        #         if same_loss_count > 3:
        #             print("Losses are not changing, stopping training")
        #             break
        #     else:
        #         same_loss_count = 0
        
        if epoch < adam_lbfgs_cutoff:
            scheduler.step()

        if epoch%10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            total_losses.append(loss.item())

        # Training Loop for mu


    #Plot each loss
    # breakpoint()
    # breakpoint()
    # physics_losses = [physics_loss.detach().numpy() for physics_loss in physics_losses]
    # plt.clf()
    # plt.semilogy(np.arange(1,len(physics_losses)+1), physics_losses, label="Physics Loss", color='blue', linestyle="-")
    # plt.semilogy(np.arange(1,len(boundary_losses)+1), boundary_losses, label="Boundary Loss", color='orange', linestyle="-")
    # plt.semilogy(np.arange(1,len(data_fit_losses)+1), data_fit_losses, label="Data Fit Loss", color='green', linestyle="-")
    # if problem == 'simpler_cont_laplace':
    #     plt.title(r"Training Losses for $\mu(x,y)=x^2+y^2$")
    # if problem == 'cont_laplace':
    #     plt.title(r"Training Losses for $\mu(x,y)=x^2+y^2+1$")
    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"training_losses_{problem}.png")
    
    #Plot what parameter distribution looks like after training
    x_shape = int(real_data_points_x.shape[0]**0.5)
    y_shape = int(real_data_points_y.shape[0]**0.5)
    X = real_data_points_x.reshape(x_shape,y_shape)
    Y = real_data_points_y.reshape(x_shape,y_shape)
    np.savetxt("X.txt", X.detach().numpy())
    np.savetxt("Y.txt", Y.detach().numpy())

    
    # mu_true_array = mu_true(real_data_points_x,real_data_points_y,problem).reshape(x_shape,y_shape)
    

    #Save u_pred and mu_field to a txt file
    # np.savetxt(f"u_pred_step{step}.txt", u_pred.detach().numpy())
    # np.savetxt(f"mu_field_step{step}.txt", mu_field.detach().numpy())
    
    

    # x_shape = int(mu_field.shape[0]**0.5)
    # y_shape = int(mu_field.shape[0]**0.5)
    #PLOT THE PREDICTED PARAMETER DISTRIBUTION
    # plt.clf()
    # plt.contourf(X.detach().numpy(),Y.detach().numpy(),mu_field.detach().numpy().reshape(x_shape,y_shape), levels=100)
    # # Set plot limits based on domain
    # plt.xlim(X.detach().numpy().min(), X.detach().numpy().max())
    # plt.ylim(Y.detach().numpy().min(), Y.detach().numpy().max())
    # if problem == 'simpler_cont_laplace':
    #     plt.title(r"Predicted Parameter Distribution when $\mu(x,y) = x^2+y^2$", fontsize=12)
    # if problem == 'cont_laplace':
    #     plt.title(r"Predicted Parameter Distribution when $\mu(x,y) = x^2+y^2+1$", fontsize=12)
    # plt.colorbar()
    # plt.tight_layout(pad=2)
    # plt.savefig(f"predicted_parameter_distribution_{problem}.png")
    
    #PLOT THE TRUE PARAMETER DISTRIBUTION
    # plt.clf()
    # plt.contourf(X.detach().numpy(),Y.detach().numpy(),mu_true(X,Y,problem).reshape(x_shape,y_shape), levels=100)
    # # Set plot limits based on domain
    # plt.xlim(X.detach().numpy().min(), X.detach().numpy().max())
    # plt.ylim(Y.detach().numpy().min(), Y.detach().numpy().max())
    # if problem == 'simpler_cont_laplace':
    #     plt.title(r"True Parameter Distribution for $\mu(x,y) = x^2+y^2$", fontsize=12)
    # if problem == 'cont_laplace':
    #     plt.title(r"True Parameter Distribution for $\mu(x,y) = x^2+y^2+1$", fontsize=12)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f"true_parameter_distribution_{problem}.png")

    #PLOT DIFFERENCE BETWEEN TRUE AND PREDICTED PARAMETER DISTRIBUTION
    # plt.clf()
    # plt.contourf(X.detach().numpy(),Y.detach().numpy(),mu_true(real_data_points_x,real_data_points_y,problem).reshape(x_shape,y_shape)-mu_field.detach().numpy().reshape(x_shape,y_shape), levels=100)
    # # Set plot limits based on domain
    # plt.xlim(X.detach().numpy().min(), X.detach().numpy().max())
    # plt.ylim(Y.detach().numpy().min(), Y.detach().numpy().max())
    # if problem == 'simpler_cont_laplace':
    #     plt.title(r"True - Pred Parameter Distribution for $\mu(x,y)=x^2+y^2$", fontsize=12)
    # if problem == 'cont_laplace':
    #     plt.title(r"True - Pred Parameter Distribution for $\mu(x,y)=x^2+y^2+1$", fontsize=12)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f"true_pred_diff_parameter_distribution_{problem}.png")
    # breakpoint()



    toc = timeit.default_timer()

    #Redefine the fine mesh
    # Create a meshgrid on a fine set of points just to plot and evaluate things
    x_vals = torch.linspace(x_min, x_max, 50)
    y_vals = torch.linspace(y_min, y_max, 50)
    fine_points_x, fine_points_y = torch.meshgrid(x_vals, y_vals, indexing='ij')

    # #Evaluate the model_for_u_for_uon the fine mesh
    u_pred, mu_pred = model_for_u(torch.cat((fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1)), dim=1))#.detach().numpy()
    # breakpoint()
    u_pred_on_mesh = torch.squeeze(u_pred)
    # mu_pred_on_mesh = torch.squeeze(mu_pred_on_mesh)
    u_pred_on_mesh_reshaped = u_pred.reshape(fine_points_x.shape[0],fine_points_y.shape[0])
    # mu_on_mesh_reshaped = mu_pred_on_mesh.reshape(fine_points_x.shape[0],fine_points_y.shape[0])

    # breakpoint()

    u_true_on_mesh = u_true(fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1), problem).reshape(fine_points_x.shape[0],fine_points_y.shape[0])
    # mu_true_on_mesh = mu_true(fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1), problem).reshape(fine_points_x.shape[0],fine_points_y.shape[0])

    # breakpoint()
    boundary_losses = [boundary_loss.detach().item() for boundary_loss in boundary_losses]
    plt.clf()
    plt.plot(boundary_losses, label="Boundary Loss", color='orange', linestyle="-")
    plt.plot(data_fit_losses, label="Data Fit Loss", color='green', linestyle="-")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig("training_losses.png")

    
    plt.clf()
    plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_pred_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='u_pred')
    plt.title('PINN Solution for u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'u_pred_parameter_estimation.png')


    plt.clf()
    plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='u_true')
    plt.title('True Solution for u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'u_true_parameter_estimation.png')

    print("The maximum difference between the predicted and true solution is: ", torch.max(torch.abs(u_pred_on_mesh_reshaped-u_true_on_mesh)))
    # breakpoint()
    plt.clf()
    plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_pred_on_mesh_reshaped.detach().numpy()-u_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='u_pred - u_true')
    plt.title('Difference between PINN and True Solution for u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'u_pred_true_diff_parameter_estimation.png')
    
    # breakpoint()
    # plt.clf()
    # plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='mu_pred')
    # plt.title('PINN Solution for mu(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'mu_pred_parameter_estimation.png')

    # plt.clf()
    # plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='mu_true')
    # plt.title('True Solution for mu(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'mu_true_parameter_estimation.png')

    # plt.clf()
    # plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_on_mesh_reshaped.detach().numpy()-mu_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='mu_pred - mu_true')
    # plt.title('Difference between PINN and True Solution for mu(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'mu_pred_true_diff_parameter_estimation.png')
    
    
    # breakpoint()
     

    from scipy.interpolate import griddata

    if problem == 'adr':
        # breakpoint()
        # adr_boundary_losses = [adr_boundary_loss.detach().item() for adr_boundary_loss in adr_boundary_losses]
        
        #ADR Losses: 
        plt.clf()
        plt.semilogy(np.arange(1,len(adr_boundary_losses)+1), adr_boundary_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Boundary Loss")
        plt.tight_layout()
        plt.savefig(f"Adv_dif_bdd_losses_training_loop_{training_loop}.png")

        
        # breakpoint()
        plt.clf()
        plt.semilogy(np.arange(1,len(adr_data_fit_losses)+1), np.array(adr_data_fit_losses))
        plt.title("Advection Diffusion Data Fit Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Interior Loss")
        plt.tight_layout()
        plt.savefig(f"Adv_dif_data_fit_losses_num_interpolation_points_training_loop_{training_loop}.png")

        # breakpoint()
        
    plt.clf()
    
    if problem == 'adr':

        true_solution = u_true(fine_points_x.reshape(-1,1),fine_points_y.reshape(-1,1), problem)
        side_length = int((fine_points_x.reshape(-1,1).shape[0])**0.5)

        true_sol_reshaped = true_solution.reshape(side_length,side_length)
        plt.clf()
        plt.imshow(model_for_u_on_mesh_reshaped, extent=(x_min,x_max,y_min,y_max), origin='lower', cmap='viridis')
        plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),model_for_u_on_mesh_reshaped, levels=100, cmap='viridis')
        cbar = plt.colorbar(label="PINN", shrink=0.8, pad=0.01)
        plt.title('Advection Diffusion Equation PINN Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"adr_PINN_training_iteration={training_loop}_mu={mu:.1f}_beta0={beta0:.1f}_beta1={beta1:.1f}.png")
        plt.close()
        
        plt.clf()
        plt.imshow(true_sol_reshaped, extent=(x_min,x_max,y_min,y_max), origin='lower', cmap='viridis')
        plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),true_sol_reshaped, levels=100, cmap='viridis')
        # Set aspect ratio to 'equal' to ensure uniform scaling
        plt.gca().set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(label="PINN", shrink=0.8, pad=0.01)
        plt.title('Advection Diffusion True Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f"adr_true_sol_mu_{mu}_beta0_{beta0}_beta1_{beta1}.png")
        plt.close() 
        # breakpoint()



        # breakpoint()
        plt.clf()
        plt.imshow(model_for_u_on_mesh_reshaped-np.asarray(true_sol_reshaped), extent=(x_min,x_max,y_min,y_max), origin='lower', cmap = 'viridis')
        plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), model_for_u_on_mesh_reshaped-np.asarray(true_sol_reshaped), levels=100,cmap='viridis')
        plt.title('Advection Diffusion model_for_u_for_u- True Solution')
        plt.colorbar(label="model_for_u- True Sol", shrink=0.8, pad=0.01)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f"adr_model_true_diff_stabilization_{stabilization_weight}_iteration={training_loop}_true_data_weight={true_data_weight}_fem_data_weight_{fem_data_weight}_mu={mu:.1f}_beta0={beta0:.1f}_beta1={beta1:.1f}_.png")
        plt.close() 

        plt.clf()

    if problem == 'two_freq':
        true_solution = u_true(fine_points_x.reshape(-1,1),fine_points_y.reshape(-1,1), problem)
        side_length = int((fine_points_x.reshape(-1,1).shape[0])**0.5)
        true_sol_reshaped = true_solution.reshape(side_length,side_length)

    plt.tight_layout() 


    if problem in ['two_freq', 'two_freq_multilevel', 'three_freq', 'three_freq_multilevel', 'high_freq']:
        true_solution = u_true(fine_points_x.reshape(-1,1),fine_points_y.reshape(-1,1), problem)
        side_length = int((fine_points_x.reshape(-1,1).shape[0])**0.5)
        true_sol_reshaped = true_solution.reshape(side_length,side_length)

    
    #Define quadrature nodes and weights for 7 point Gaussian quadrature on [0,1]x[0,1]
    if problem == 'adr':
        x = np.array([1/2, 780/989, 780/989, 780/3691, 780/3691, 3363/3421, 58/3421])
        y = np.array([1/2, 496/559, 496/4401, 496/559, 496/4401, 1/2, 1/2])
        w = [2/7, 5/36, 5/36, 5/36, 5/36, 5/63, 5/63]
        quad_points = np.array([x,y])
        quad_points = np.transpose(quad_points)

    else: 
        #Define evaluation points and weights for 7 point Gaussian quadrature on [-1,1]x[-1,1]
        quad_points = torch.zeros(7,2)
        aux1, aux2 = 1/np.sqrt(3), np.sqrt(3/5)
        aux3 = np.sqrt(14/15)
        quad_points[0] = torch.tensor([0, 0])
        quad_points[1] = torch.tensor([aux1, aux2])
        quad_points[2] = torch.tensor([-aux1, aux2])
        quad_points[3] = torch.tensor([aux1, -aux2])
        quad_points[4] = torch.tensor([-aux1, -aux2])
        quad_points[5] = torch.tensor([aux3, 0])
        quad_points[6] = torch.tensor([-aux3, 0])

        w = [8/7, 20/36, 20/36, 20/36, 20/36, 20/63, 20/63]

    #COMMENT BACK IN FOR ERROR CALCULATION
    # #Compute squared difference between model_for_uand true solution to compute L^2 error 
    # mu_true_model_diff = []
    # u_true_model_diff = []
    # for i in range(quad_points.shape[0]):
        
    #     mu_true_val = mu_true(quad_points[i][0],quad_points[i][1],problem)
    #     u_true_val = u_true(quad_points[i][0],quad_points[i][1],problem)
    #     input_tensor_for_error_calc = torch.tensor([quad_points[i,0], quad_points[i,1]]).unsqueeze(0)
    #     u_pred_val, mu_pred= model_for_u(input_tensor_for_error_calc)
        
    #     #TO DO:
    #     #I should switch this to be the difference between the true parameters 
    #     #and the computed parameters instead of the solution u
    #     u_true_model_diff.append((u_true_val.item()-u_pred_val.item())**2)
    #     # mu_true_model_diff.append((mu_true_val.item()-mu_pred_val.item())**2)
    
    # L2_error_squared_for_u = (np.dot(np.array(w), np.array(u_true_model_diff)))
    # L2_error_for_u = (L2_error_squared_for_u)**0.5 #L2 error is sqrt(integral(u_pred-u_true)^2)
    # # L2_error_squared_for_mu = (np.dot(np.array(w), np.array(mu_true_model_diff)))
    # # L2_error_for_mu = (L2_error_squared_for_mu)**0.5 #L2 error is sqrt(integral(u_pred-u_true)^2)
    
    # # breakpoint()
    # max_error_for_u = max(abs(u_pred_on_mesh_reshaped.reshape(-1,1)-u_true_on_mesh.reshape(-1,1))).item()
    # # max_error_for_mu = max(abs(mu_on_mesh_reshaped.reshape(-1,1)-mu_true_on_mesh.reshape(-1,1))).item()

    
    
    #Check if L2 error and max error relationship makes sense
    # breakpoint()
    u_pred, mu_pred = model_for_u(torch.cat((real_data_points_x,real_data_points_y), dim=1))
    bestParams = model_for_u.state_dict()
    torch.save(model_for_u.u_net.state_dict(), 'u_net_from_A.pth')
    # torch.save(bestParams, 'neural_network_for_u_params.pth')
    print("END OF FIRST TRAINING:")
    print(bestParams)    
    # model_fine_mesh, sigma_data, sigma_bdd = (torch.cat((fine_points_x.reshape(-1,1).to(torch.float64),fine_points_y.reshape(-1,1).to(torch.float64)),dim=1))
    # true_sol_fine_mesh = u_true(fine_points_x.reshape(-1,1).to(torch.float64),fine_points_y.reshape(-1,1).to(torch.float64), problem)
    # squared_diff = torch.square(model_fine_mesh - true_sol_fine_mesh)
    
    return X, Y, u_pred, u_pred_on_mesh_reshaped, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np#, L2_error_for_u, max_error_for_u


























    # if problem == 'nonlinear' or problem == 'discontinuous' or problem == 'adr':
    #     residual = 0
    #     for i in range(evaluation_points.shape[0]):
    #         residual = residual + weights[i]*true_model_diff[i]
    #     # breakpoint()
    #     print("Integrated error is ", integrated_error)

    # breakpoint()
    #Save parameters after the first training:

    

    #Add code to write to an output file 

    # with open("output.txt",'a') as f:

    #     if problem == 'nonlinear' or problem == 'discontinuous':
    #         f.write("\n"+"2nd Order {problem} Random Initialization: "+str(toc-tic)+str(" seconds to train"))
    #         f.write("\n"+"Sampling true data "+str(sampling_true_data))
    #         f.write("\n"+"Sampling fem data: "+str(sampling_fem_data))
    #         # f.write("\n"+"max error: "+ str(max_error))
    #         # f.write("\n"+"mean error: "+ str(mean_error))
    #         f.write("\n"+str("L^2 error ")+str(np.sqrt(residual.detach()))+"\n")
    # breakpoint()
    # return np.sqrt(residual.detach().item())

    # breakpoint()

# problem = 'adr'
# mesh_size=10
# num_fem_points=10
# mu=1
# beta0=1
# beta1=1
# num_real_data_points = 10
# num_phys_points = 100
# num_boundary_points = 50
# num_epochs=100
# physics_weight=1
# boundary_weight=50
# fem_data_weight=1
# true_data_weight=0
# stabilization_weight = 1
# order=2
# training_loop=0

# #Figure out the best way to decrease the stabilization weight in multilevel
# stabilized_1(problem, mesh_size, num_fem_points, num_real_data_points, num_phys_points, num_boundary_points, mu, beta0, beta1, num_epochs, physics_weight, boundary_weight, fem_data_weight, true_data_weight, stabilization_weight, order, training_loop)

#Problems implemented 
# 'discontinuous' (Has FEM Data, Has true data)
# 'nonlinear' (No FEM Data, Has true data)
# 'adr' (No FEM Data, No true data)

# problem = 'discontinuous'

#True data: data_source = 0
#FEM data: data_source = 1
#No data: data_source = 2
# run(problem, num_interpolation_points=20, data_source=0)

#Automated plotting (run this once I get the plots all looking nice)
# for sampling_true_data in range(2):
#     print("sampling true data ", sampling_true_data)
#     for num_interpolation_points in range(5,50,5):
        
#         if sampling_true_data == 0: 
#             print("Sampling fem data")
#             residual = run(problem, sampling_true_data, num_interpolation_points)
#             fem_data_residuals.append(residual)

#         if sampling_true_data == 1:
#             print("Sampling true data")
#             residual = run(problem, sampling_true_data, num_interpolation_points)
#             noisy_data_residuals.append(residual)
        
        

# breakpoint()
    # breakpoint()

    #Error computations

    # max_error = max(np.abs(fem_interpolated_reshaped-model_on_mesh_reshaped))
    # mean_error = np.mean(np.abs(fem_interpolated_reshaped-model_on_mesh_reshaped))

    # print("Max error is ", max_error)
    # print("Mean error is ", mean_error)


    #Analyze the Hessian matrix 

    # # After training or at checkpoints, compute the Hessian
    # def compute_hessian(x, y, samples_f,model):
    #     # Forward pass
    #     outputs = model(inputs)
    #     loss = interior_loss_func(x, y, samples_f, model)

    #     # Compute first derivatives
    #     grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    #     hessian = []
    #     for g in grads:
    #         row = []
    #         for i in range(len(g)):
    #             # Compute second derivatives
    #             second_derivative = torch.autograd.grad(g[i], model.parameters(), retain_graph=True)
    #             row.append(second_derivative)
    #         hessian.append(row)

    #     return hessian

    # # Example usage
    # inputs = fem_data[:,:2]
    # targets = fem_data[:,2]
    # samples_f = f_true(inputs[:,0],inputs[:,1])
    # hessian = compute_hessian(x, y, samples_f, model)

    # Debugging tips for why the network might be returning zero:

    # 1. Check the initialization of weights and biases:
    # Ensure that the weights and biases are initialized properly. Poor initialization can lead to vanishing gradients or the network being stuck in a poor local minimum.

    # 2. Verify the loss function:
    # Ensure that the loss function is correctly implemented and that the gradients are being computed properly. You can add print statements to check the loss values during training.

    # 3. Check the learning rate:
    # A learning rate that is too high or too low can cause the network to fail to learn. Experiment with different learning rates.

    # 4. Verify the data:
    # Ensure that the input data and target values are correctly prepared and normalized if necessary. Incorrect data can lead to poor training results.

    # 5. Check the optimizer:
    # Ensure that the optimizer is correctly configured and that the parameters of the network are being updated during training.

    # 6. Monitor gradients:
    # Check if the gradients are vanishing or exploding. You can print the gradient norms of the parameters to debug this issue.

    # 7. Debug the forward pass:
    # Print intermediate outputs of the network to ensure that the forward pass is working as expected.

    # 8. Check for over-regularization:
    # If you are using regularization techniques (e.g., weight decay), ensure that they are not too strong, as this can prevent the network from learning.

