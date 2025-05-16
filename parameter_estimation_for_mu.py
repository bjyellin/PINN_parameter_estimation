import torch
import torch.nn as nn
import os
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



# def interior_pde_loss_func(x,y, samples_f, model_for_mu, problem):


    
#     u = model_for_mu(torch.cat((x,y), dim=1))
#     # breakpoint()
#     n = len(u)

#     torch.autograd.set_detect_anomaly(True)

#     # u2 = u.clone()

#     u_pred, mu_pred = model_for_mu(torch.cat((x,y), dim=1))
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
        return x.clone().detach()**2+y.clone().detach()**2+1
    
    if problem == 'simpler_cont_laplace':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return (x.clone().detach()**2+y.clone().detach()**2)

    if problem == 'very_simple':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return 2*torch.ones_like(x.clone().detach())
    
    if problem == 'constant_mu':
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float64)
        return 1*torch.ones_like(x.clone().detach())

def u_true(x, y, problem):
    if isinstance(x,np.ndarray):
        x = torch.tensor(x, dtype=torch.float64)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float64)
    if problem == 'cont_laplace' or problem=='simpler_cont_laplace' or problem=='very_simple':
        # breakpoint()
        return torch.sin(torch.tensor(x)+torch.tensor(y))
    if problem == 'constant_mu':
        return torch.sin(torch.pi*x)*torch.sin(torch.pi*y)
        #return torch.sin(torch.pi*x.clone().detach())*torch.sin(torch.pi*y.clone().detach())
        
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
        return 2*torch.pi**2*torch.sin(torch.pi*x.clone().detach())*torch.sin(torch.pi*y.clone().detach())


#mesh size is the size of the FEM mesh
#num_fem_points is the number of points sampled in each direction from the FEM solution 
def estimate_mu(problem, step, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration):
    print("Starting parameter estimation")
    # print("mu_iteration is ", mu_iteration)
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            torch.set_default_dtype(torch.float64)
            super(NeuralNetwork, self).__init__()
            

            self.flatten = torch.nn.Flatten()


            # Learnable weights (log sigma^2 form to stabilize optimization)
            self.log_sigma_data = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=True)  # For data loss
            self.log_sigma_boundary = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=True)   # For boundary loss


            #Input layer is 2 dimensional because I have (x,y) information in data 
            #Output layer is 1 dimensional because I want to output the temperature
            #at that particular point 
            
            
            # Load the parameters learned from parameter_estimation_for_u
            self.u_net = nn.Sequential(
                nn.Linear(2, 20),
                nn.Tanh(),
                nn.Linear(20,20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )
            # u_net_checkpoint = torch.load("/Users/benjaminyellin/Desktop/BensStuff/EmoryAppliedMath/Year5/Research/PINNs/FEM_Initialization/neural_network_for_u_params.pth", map_location=lambda storage, loc: storage)
            # # Extract only the u_net parameters
            # u_net_state = {k.replace("u_net.", ""): v for k, v in u_net_checkpoint.items() if k.startswith("u_net.")}
            
            # # Load them into self.u_net
            # self.u_net.load_state_dict(u_net_state)

            #Make this network simpler in this file and u file 
            self.mu_net = torch.nn.Sequential(
                torch.nn.Linear(2,20),
                torch.nn.Softplus(),
                torch.nn.Linear(20,5),
                torch.nn.Softplus(),
                torch.nn.Linear(5,5),
                torch.nn.Softplus(),
                torch.nn.Linear(5,1),
                torch.nn.Softplus()  # Ensures positive output
            )
            # print("mu_iteration = ", mu_iteration)
            # if mu_iteration == 0:
                #Load only u_net from stage A
            if os.path.exists("u_net_from_A.pth"):
                print("Loading u_net from stage A")
                self.u_net.load_state_dict(torch.load("u_net_from_A.pth"))
            else:
                self.u_net.load_state_dict(torch.load("u_net".pth))
                
               
            # else:
            #     print("mu_iteration is not 0")
            #     breakpoint()
            #     #Load both u_net and mu_net from previous training
            #     if os.path.exists("u_net.pth"):
            #         print("Loading u_net from previous training C")
            #         self.u_net.load_state_dict(torch.load("u_net.pth"))
            #         breakpoint()
            #     else:
            #         raise FileNotFoundError("Expected u_net.pth file not found.")

            #     if os.path.exists("mu_net.pth"):
            #         print("Loading mu_net from previous training C")
            #         self.mu_net.load_state_dict(torch.load("mu_net.pth"))
            #     else:
            #         raise FileNotFoundError("Expected mu_net.pth file not found.")
            
            self.log_sigma_data = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)  # For data loss
            self.log_sigma_boundary = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)   # For physics residual loss
            # breakpoint()
            # full_state = (u_net_checkpoint["state_dict"])

            # from collections import OrderedDict
            # u_net_state = OrderedDict()

            # for k, v in full_state.items():
            #     if k.startswith('u_net'):
            #         u_net_state[k.replace("u_net.","")] = v
            # self.u_net.load_state_dict(u_net_state)
            # breakpoint()
            

        #Now I need to fix this function. I don't want to randomly initialize the u weights. I want to keep the imported ones from the previous line
        def initialize_weights(self, seed=42):
            # breakpoint()
            torch.manual_seed(seed)
            # Initialize mu_net weights with Xavier initialization
            for layer in self.mu_net:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

            return self
        
        # def forward(self, x):
        #     # print("INSIDE FORWARD: returning 4 values")
        #     raise RuntimeError("You hit MY forward()")
        #     #x = self.flatten(x)
        
        def forward(self, x):
            # print("INSIDE FORWARD: returning 4 values")
            # raise RuntimeError("You hit MY forward()")
            #x = self.flatten(x)
            torch.set_default_dtype(torch.float64)
            # print("x dtype is ",x.dtype)
            # breakpoint()
            mu = self.mu_net(x)
            u = self.u_net(x)
            sigma_data = torch.exp(self.log_sigma_data)
            sigma_boundary = torch.exp(self.log_sigma_boundary)
            # breakpoint()
            # print("mu is ", mu)
            # print("sigma data is ", sigma_data)
            # print("sigma bdd is ", sigma_boundary)

            return u, mu#, sigma_boundary
        
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
        phys_points_x = phys_points[:,0].reshape(-1,1).requires_grad_().double()
        phys_points_y = phys_points[:,1].reshape(-1,1).requires_grad_().double()

    
    x_data = torch.linspace(x_min,x_max,num_real_data_points)
    y_data = torch.linspace(y_min,y_max,num_real_data_points)

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

    
    #Import the trained u parameters 
    # u = torch.load("neural_network_for_u_params.pth", map_location=lambda storage, loc: storage)
    # breakpoint()

    epoch_state = {'epoch': 0, 'counter': 0}
    stop_training = False
    def closure():
        nonlocal stop_training
        epoch = epoch_state['epoch']
        epoch_state['counter'] += 1
        # print("Sampling true data: ", sampling_true_data)
        torch.set_default_dtype(torch.float64)

        from scipy.interpolate import griddata
            
        #Compute and normalize losses
        if num_real_data_points != 0:
            # breakpoint()
            u_pred, mu_pred = model_for_mu(torch.cat((real_data_points_x,real_data_points_y), dim=1))
            # breakpoint()
            noise = torch.normal(0,0.1,size=(real_data_points_x.shape[0],1))*0
            # breakpoint()
            data_fit_loss = criterion(u_true(real_data_points_x,real_data_points_y,problem)+noise, u_pred)/real_data_points_x.shape[0]
            # breakpoint()
            # normalized_interior_mismatch = data_fit_loss/real_data_points_x.shape[0]
            data_fit_losses.append(data_fit_loss.detach().item())


        samples_f = f_true(phys_points_x, phys_points_y, problem)

        residual = interior_pde_loss_func(phys_points_x, phys_points_y, samples_f, model_for_mu, problem)/phys_points_x.shape[0]#NORMALIZE 
        # print("residual: ",residual)
        residual.requires_grad_()

        physics_losses.append(residual.detach().item())
        
        boundary_loss = dirichlet_bdd_loss_func(boundary_points[:,0].reshape(-1,1),boundary_points[:,1].reshape(-1,1), model_for_mu, problem, iteration, num_epochs)/boundary_points[:,0].shape[0]
        # breakpoint()
        boundary_loss = boundary_loss#/(boundary_points[:,0].shape[0])
        boundary_losses.append(boundary_loss.detach().item())

        # breakpoint()

        #Compute NTK-based weights
        losses = {"physics": residual, "boundary": boundary_loss, "data_fit":data_fit_loss}
        
        #Compute the total loss 
        sigma_data = torch.exp(model_for_mu.log_sigma_data)
        sigma_bdd = torch.exp(model_for_mu.log_sigma_boundary)

        regularization_term = 0.0
        # Access the network learned for mu
        mu_network = model_for_mu.mu_net
        # print("Learned network for mu:", mu_network)
        
        for param in mu_network.parameters():
            regularization_term += torch.sum(param**2)

        lambda_reg = 0.01
        
        # if epoch < int(num_epochs):
        #     total_loss =  sigma_data*normalized_interior_mismatch + sigma_bdd*boundary_loss # + residual+ lambda_reg*regularization_term
        
        # if epoch < int(num_epochs/2):
        #     total_loss = sigma_data * normalized_interior_mismatch + sigma_bdd*boundary_loss
        #     # print("residual, even though we're not training on the residual: ", residual)
        # if epoch == int(num_epochs/2):
        #     print("just started using the residual")
        # if epoch>=int(num_epochs/2):

        #Regularization 
        lam = 1e-2
        # print("mu_pred.requires grad", mu_pred.requires_grad)
        # breakpoint()
        points = torch.cat((real_data_points_x, real_data_points_y), dim=1).requires_grad_()

        #Computing the norm of the Hessian
        mu_grad_x = torch.autograd.grad(mu_pred, real_data_points_x, grad_outputs=torch.ones_like(mu_pred), retain_graph=True, create_graph=True)[0]
        mu_grad_y = torch.autograd.grad(mu_pred, real_data_points_y, grad_outputs=torch.ones_like(mu_pred), retain_graph=True, create_graph=True)[0]

        mu_hess_xx = torch.autograd.grad(mu_grad_x, real_data_points_x, grad_outputs=torch.ones_like(mu_grad_x), retain_graph=True, create_graph=True)[0]
        mu_hess_yy = torch.autograd.grad(mu_grad_y, real_data_points_y, grad_outputs=torch.ones_like(mu_grad_y), retain_graph=True, create_graph=True)[0]
        
        mu_hess = mu_hess_xx + mu_hess_yy
        # breakpoint()
        # print("residual: ", residual)
        mu_grad_norm_sq = torch.norm(mu_grad_x)**2 + torch.norm(mu_grad_y)**2

        #Try to push mu towards what we know it is (lambda = 1 everywhere)
        def mu_true(x, y, problem):
            if problem == 'cont_laplace':
                return x**2 + y**2 + 1
            if problem == 'simpler_cont_laplace':
                return x**2 + y**2
            if problem == 'very_simple':
                return 2*torch.ones_like(x)
            if problem == 'constant_mu':
                return torch.ones_like(x)

        bias_term = torch.norm(mu_pred - mu_true(real_data_points_x, real_data_points_y, problem))**2
        
        total_loss =  residual + 1000*boundary_loss + lam*mu_grad_norm_sq  #+ bias_term #+ data_fit_loss #+ boundary_loss
        # breakpoint()
            # print("residual in the second half of training: ", residual)
            
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
        if len(total_losses) > 10 and abs(total_losses[-10]-total_losses[-1])<1e-10:
            print("Losses are not changing significantly. Stopping training early.")
            stop_training = True
            return total_loss
        

        # print("Residual:", residual.item())
        # print("Boundary loss:", boundary_loss.item())
        # print("mu_grad_norm_sq:", mu_grad_norm_sq.item())
        # print("bias_term:", bias_term.item())

        #Back propagate
        total_loss.backward(retain_graph=True)

        # for name, param in model_for_mu.named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}")
        #     else:
        #         print("This is very bad news")
        #         print(f"No gradient for {name}")

        
        epoch_state['counter'] += 1
        # if len(total_losses) > 10 and abs(total_losses[-10] - total_losses[-1]) < 1e-8:
        #     print("Losses are not changing significantly. Stopping training early.")
        #     return loss
        return total_loss

    #Training loop
    
    #Also try the cutoff at a lower number like 5 epochs to see if it runs past that 
    adam_lbfgs_cutoff = int(num_epochs/2)
    tic = timeit.default_timer()
    
    print("Beginning of second training")
    
    model_for_mu = NeuralNetwork().initialize_weights()
    print(model_for_mu)
    for name, param in model_for_mu.mu_net.named_parameters():
        print(name, param.shape)
        print(param)
    # breakpoint()
    # breakpoint()
    optimizer = torch.optim.Adam(model_for_mu.mu_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # breakpoint()
    

    # Print all model_for_mu parameters
    for name, param in model_for_mu.named_parameters():
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
    
    # breakpoint()
    threshold = 1000 #Threshold for where to check if losses are close and exit loop
    
    same_loss_count = 0

    #Training Loop for u
    switching_to_lbfgs_count = 0 
    for epoch in range(num_epochs):
        epoch_state['epoch'] = epoch
        if len(total_losses) >= 50 and epoch < adam_lbfgs_cutoff:
            if abs(total_losses[-30]-total_losses[-1]) < 10**-2:
                # print("Switching to bfgs sooner than expected because Adam did all it could")
                if switching_to_lbfgs_count == 0:
                    print("Switching to LBFGS")
                    switching_to_lbfgs_count = 1
                optimizer =  torch.optim.LBFGS(model_for_mu.mu_net.parameters(), max_iter=20, line_search_fn='strong_wolfe')

        #Try switching optimizers during training
        elif epoch == adam_lbfgs_cutoff:
            print("Just switched to LBFGS")
            optimizer =  torch.optim.LBFGS(model_for_mu.mu_net.parameters(), max_iter=20, line_search_fn='strong_wolfe')

        

        # if len(total_losses)>30:
        #     # print("Difference between 30th to last and last losses", (abs(total_losses[-30]-total_losses[-1])))
        #     if (abs(total_losses[-30]-total_losses[-1])<1e-5):
        #         print("Stopping training because losses aren't changing")
        #         break
        

        #Conditional gradient clipping
        if isinstance(optimizer, torch.optim.Adam):
            torch.nn.utils.clip_grad_norm_(model_for_mu.parameters(), max_norm=1.0)
 
        optimizer.zero_grad()
        loss=optimizer.step(closure)
        if stop_training:
            break

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
    plt.clf()
    plt.semilogy(np.arange(1,len(physics_losses)+1), physics_losses, label="Physics Loss", color='blue', linestyle="-")
    plt.semilogy(np.arange(1,len(boundary_losses)+1), boundary_losses, label="Boundary Loss", color='orange', linestyle="-")
    plt.semilogy(np.arange(1,len(data_fit_losses)+1), data_fit_losses, label="Data Fit Loss", color='green', linestyle="-")
    if problem == 'simpler_cont_laplace':
        plt.title(r"Training Losses for $\mu(x,y)=x^2+y^2$")
    if problem == 'cont_laplace':
        plt.title(r"Training Losses for $\mu(x,y)=x^2+y^2+1$")
    if problem == 'constant_mu':
        plt.title(r"Training Losses for $\mu(x,y)=1$")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_losses_{problem}.png")
    
    #Plot what parameter distribution looks like after training
    x_shape = int(real_data_points_x.shape[0]**0.5)
    y_shape = int(real_data_points_y.shape[0]**0.5)
    X = real_data_points_x.reshape(x_shape,y_shape)
    Y = real_data_points_y.reshape(x_shape,y_shape)
    np.savetxt("X.txt", X.detach().numpy())
    np.savetxt("Y.txt", Y.detach().numpy())

    def mu_true(x, y, problem):
        if problem == 'cont_laplace':
            if isinstance(x,np.ndarray):
                x = torch.tensor(x, dtype=torch.float64)
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float64)
            return x.clone().detach()**2+y.clone().detach()**2+1
        
        if problem == 'simpler_cont_laplace':
            if isinstance(x,np.ndarray):
                x = torch.tensor(x, dtype=torch.float64)
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float64)
            return (x.clone().detach()**2+y.clone().detach()**2)

        if problem == 'very_simple':
            if isinstance(x,np.ndarray):
                x = torch.tensor(x, dtype=torch.float64)
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float64)
            return 2*torch.ones_like(x.clone().detach())
        
        if problem == 'constant_mu':
            if isinstance(x,np.ndarray):
                x = torch.tensor(x, dtype=torch.float64)
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float64)
            return 1*torch.ones_like(x.clone().detach())

    
    mu_true_array = mu_true(real_data_points_x,real_data_points_y,problem).reshape(x_shape,y_shape)
    mu, u = model_for_mu(torch.cat((real_data_points_x,real_data_points_y), dim=1))

    #Save u_pred and mu_field to a txt file
    np.savetxt(f"u_pred_step{step}.txt", u.detach().numpy())
    np.savetxt(f"mu_field_step{step}.txt", mu.detach().numpy())
    
    

    x_shape = int(mu.shape[0]**0.5)
    y_shape = int(u.shape[0]**0.5)
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

    # #Evaluate the model_for_mu on the fine mesh
    u_pred_on_mesh, mu_pred_on_mesh = model_for_mu(torch.cat((fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1)), dim=1))#.detach().numpy()
    # breakpoint()
    u_pred_on_mesh = torch.squeeze(u_pred_on_mesh)
    mu_pred_on_mesh = torch.squeeze(mu_pred_on_mesh)
    u_pred_on_mesh_reshaped = u_pred_on_mesh.reshape(fine_points_x.shape[0],fine_points_y.shape[0])
    mu_on_mesh_reshaped = mu_pred_on_mesh.reshape(fine_points_x.shape[0],fine_points_y.shape[0])

    # breakpoint()

    u_true_on_mesh = u_true(fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1), problem).reshape(fine_points_x.shape[0],fine_points_y.shape[0])
    mu_true_on_mesh = mu_true(fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1), problem).reshape(fine_points_x.shape[0],fine_points_y.shape[0])

    # breakpoint()
    # plt.clf()
    # plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_pred_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='u_pred')
    # plt.title('PINN Solution for u(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'u_pred_parameter_estimation.png')


    # plt.clf()
    # plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='u_true')
    # plt.title('True Solution for u(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'u_true_parameter_estimation.png')

    # plt.clf()
    # plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),u_pred_on_mesh_reshaped.detach().numpy()-u_true_on_mesh.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='u_pred - u_true')
    # plt.title('Difference between PINN and True Solution for u(x,y)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.tight_layout()
    # plt.savefig(f'u_pred_true_diff_parameter_estimation.png')
    
    
    plt.clf()
    plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis', vmin=mu_on_mesh_reshaped.min(),vmax=mu_on_mesh_reshaped.max())
    plt.colorbar(label='mu_pred')
    plt.title('PINN Solution for mu(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'mu_pred_parameter_estimation.png')

    plt.clf()
    plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_true_on_mesh.detach().numpy(), levels=1, cmap='viridis', vmin=mu_true_on_mesh.min(),vmax=mu_true_on_mesh.max())
    plt.colorbar(label='mu_true')
    plt.title('True Solution for mu(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'mu_true_parameter_estimation.png')

    plt.clf()
    plt.contourf(fine_points_x.detach().numpy(),fine_points_y.detach().numpy(),mu_on_mesh_reshaped.detach().numpy()-mu_true_on_mesh.detach().numpy(), levels=100, cmap='viridis', vmin=(mu_on_mesh_reshaped-mu_true_on_mesh).min(), vmax=(mu_on_mesh_reshaped-mu_true_on_mesh).max())
    plt.colorbar(label='mu_pred - mu_true')
    plt.title('Difference between PINN and True Solution for mu(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'mu_pred_true_diff_parameter_estimation.png')
    
    
    # Compute the residual of the PDE: mu * Laplace(u) - f
    def compute_pde_residual(x, y, model, problem):
        
        u_pred, mu_pred = model(torch.cat((x, y), dim=1))
        # u_pred = torch.tensor(u_pred, requires_grad=True)
        # breakpoint()
        # Compute gradients
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        
        # Compute Laplace(u)
        laplace_u = u_xx + u_yy
        
        # Compute the residual
        f = f_true(x, y, problem)
        residual = -mu_pred * laplace_u - f
        
        return residual

    

    #Compute the residual with the learned mu and true u to see how good the learned mu is
    def isolate_mu_error(x, y, model, problem):
        
        u_pred, mu_pred = model(torch.cat((x, y), dim=1))

        # mu_pred = torch.tensor(mu_pred, requires_grad=True)
        mu_pred_reshaped = mu_pred.reshape(int(mu_pred.shape[0]**0.5), int(mu_pred.shape[0]**0.5))
        
        # u_pred = torch.tensor(u_pred, requires_grad=True)
        u_true_vals = u_true(x, y, problem)

        #USE THIS TO COMPUTE THE RESIDUAL WITH THE TRUE U AND TRUE MU TO JUST MAKE SURE THE RESIDUAL PLOT LOOKS RIGHT WITH REAL ANSWERS 
        u_true_reshaped = u_true_vals.reshape(int(u_true_vals.shape[0]**0.5), int(u_true_vals.shape[0]**0.5))
        

        mu_true_vals = mu_true(x, y, problem)
        # breakpoint()
        # Compute gradients
        u_x = torch.autograd.grad(u_true_vals, x, grad_outputs=torch.ones_like(u_true_vals), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u_true_vals, y, grad_outputs=torch.ones_like(u_true_vals), retain_graph=True, create_graph=True)[0]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        
        # Compute Laplace(u) with automatic differentiation
        # laplace_u = u_xx + u_yy

        #Compute Laplacian by hand
        laplace_u = -f_true(x, y, problem)
        
        # breakpoint()
        # Compute the residual
        f = f_true(x, y, problem)
        # breakpoint()
        residual = -mu_pred * laplace_u - f
        # breakpoint()
        # breakpoint()
        x = x.reshape(int(x.shape[0]**0.5), int(x.shape[0]**0.5))
        y = y.reshape(int(y.shape[0]**0.5), int(y.shape[0]**0.5))
        # breakpoint()
        residual = residual.reshape(int(x.shape[0]), int(y.shape[0]))


        # breakpoint()
        fake_residual = -1.185*torch.ones_like(mu_pred)*laplace_u - f
        fake_residual = fake_residual.reshape(int(x.shape[0]), int(y.shape[0]))


        # breakpoint()
        plt.clf()
        plt.contourf(x.detach().numpy(), y.detach().numpy(), residual.detach().numpy(), levels=100, cmap='viridis')
        plt.colorbar(label='Residual')
        plt.title('Residual of the PDE: ||-mu * Laplace(u) - f|| with learned mu')
        plt.tight_layout()
        plt.savefig(f'residual_of_pde_with_learned_mu_and_true_u.png')

        plt.clf()
        plt.contourf(x.detach().numpy(), y.detach().numpy(), fake_residual.detach().numpy(), levels=100, cmap='viridis')
        plt.colorbar(label='Fake Residual with Average Mu')
        plt.title('Fake Residual of the PDE: ||mu * Laplace(u) - f|| with Average Mu')
        plt.tight_layout()
        plt.savefig(f'residual_of_pde_with_fake_mu_and_true_u.png')

        plt.clf()


        plt.clf()
        plt.contourf(x.detach().numpy(), y.detach().numpy(), mu_pred_reshaped.detach().numpy(), levels=100, cmap='viridis')
        plt.colorbar(label='mu_pred')
        plt.title('Learned mu(x,y)')
        plt.tight_layout()
        plt.savefig(f'learned_mu.png')

        # breakpoint()
        return residual, fake_residual


    # Evaluate the residual on a fine mesh
    fine_points_x = fine_points_x.reshape(-1, 1).requires_grad_()
    fine_points_y = fine_points_y.reshape(-1, 1).requires_grad_()

    residual = compute_pde_residual(fine_points_x, fine_points_y, model_for_mu, problem)
    residual_reshaped = residual.reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5))
    # breakpoint()
    residual, fake_residual = isolate_mu_error(fine_points_x, fine_points_y, model_for_mu, problem)
    
    

    residual_with_true_mu_reshaped = residual.reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5))
    # breakpoint()
    # Plot the residual with the learned mu and learned u
    plt.clf()
    plt.contourf(fine_points_x.detach().numpy().reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5)), fine_points_y.detach().numpy().reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5)), residual_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='Residual')
    plt.title('Residual of the PDE: ||mu * Laplace(u) - f|| with learned mu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'pde_residual_mu_laplace_u_minus_f_learned_mu_and_u.png')

    #Plot the residual with the learned mu and true u
    plt.clf()
    plt.contourf(fine_points_x.detach().numpy().reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5)), fine_points_y.detach().numpy().reshape(int(fine_points_x.shape[0]**.5), int(fine_points_y.shape[0]**0.5)), residual_with_true_mu_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='Residual')
    plt.title('Residual of the PDE: ||mu * Laplace(u) - f|| with true mu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'pde_residual_mu_laplace_u_minus_f_learned_u_true_mu.png')



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
        plt.savefig(f"Adv_dif_bdd_losses_training_loop_{training_loop}_phys_weight_{physics_weight}.png")

        
        # breakpoint()
        plt.clf()
        plt.semilogy(np.arange(1,len(adr_data_fit_losses)+1), np.array(adr_data_fit_losses))
        plt.title("Advection Diffusion Data Fit Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Interior Loss")
        plt.tight_layout()
        plt.savefig(f"Adv_dif_data_fit_losses_num_interpolation_points_training_loop_{training_loop}_phys_weight_{physics_weight}.png")

        # breakpoint()
        
    plt.clf()
    
    if problem == 'adr':

        true_solution = u_true(fine_points_x.reshape(-1,1),fine_points_y.reshape(-1,1), problem)
        side_length = int((fine_points_x.reshape(-1,1).shape[0])**0.5)

        true_sol_reshaped = true_solution.reshape(side_length,side_length)
        plt.clf()
        plt.imshow(model_for_mu_on_mesh_reshaped, extent=(x_min,x_max,y_min,y_max), origin='lower', cmap='viridis')
        plt.contourf(fine_points_x.numpy(),fine_points_y.numpy(),model_for_mu_on_mesh_reshaped, levels=100, cmap='viridis')
        cbar = plt.colorbar(label="PINN", shrink=0.8, pad=0.01)
        plt.title('Advection Diffusion Equation PINN Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"adr_PINN_training_iteration={training_loop}_phys_weight={physics_weight}_mu={mu:.1f}_beta0={beta0:.1f}_beta1={beta1:.1f}.png")
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
        plt.imshow(model_for_mu_on_mesh_reshaped-np.asarray(true_sol_reshaped), extent=(x_min,x_max,y_min,y_max), origin='lower', cmap = 'viridis')
        plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), model_for_mu_on_mesh_reshaped-np.asarray(true_sol_reshaped), levels=100,cmap='viridis')
        plt.title('Advection Diffusion model_for_mu - True Solution')
        plt.colorbar(label="model_for_mu - True Sol", shrink=0.8, pad=0.01)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(f"adr_model_for_mu_true_diff_stabilization_{stabilization_weight}_iteration={training_loop}_phys_weight={physics_weight}_true_data_weight={true_data_weight}_fem_data_weight_{fem_data_weight}_mu={mu:.1f}_beta0={beta0:.1f}_beta1={beta1:.1f}_.png")
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

    #Compute squared difference between model_for_mu and true solution to compute L^2 error 
    mu_true_model_for_mu_diff = []
    u_true_model_for_mu_diff = []
    for i in range(quad_points.shape[0]):
        
        mu_true_val = mu_true(quad_points[i][0],quad_points[i][1],problem)
        u_true_val = u_true(quad_points[i][0],quad_points[i][1],problem)
        input_tensor_for_error_calc = torch.tensor([quad_points[i,0], quad_points[i,1]]).unsqueeze(0)
        u_pred, mu_pred  = model_for_mu(input_tensor_for_error_calc)
        
        #TO DO:
        #I should switch this to be the difference between the true parameters 
        #and the computed parameters instead of the solution u
        u_true_model_for_mu_diff.append((u_true_val.item()-u_pred.item())**2)
        mu_true_model_for_mu_diff.append((mu_true_val.item()-mu_pred.item())**2)
    
    L2_error_squared_for_u = (np.dot(np.array(w), np.array(u_true_model_for_mu_diff)))
    L2_error_for_u = (L2_error_squared_for_u)**0.5 #L2 error is sqrt(integral(u_pred-u_true)^2)
    L2_error_squared_for_mu = (np.dot(np.array(w), np.array(mu_true_model_for_mu_diff)))
    L2_error_for_mu = (L2_error_squared_for_mu)**0.5 #L2 error is sqrt(integral(u_pred-u_true)^2)
    
    # breakpoint()
    max_error_for_u = max(abs(u_pred_on_mesh_reshaped.reshape(-1,1)-u_true_on_mesh.reshape(-1,1))).item()
    max_error_for_mu = max(abs(mu_on_mesh_reshaped.reshape(-1,1)-mu_true_on_mesh.reshape(-1,1))).item()

    
    
    #Check if L2 error and max error relationship makes sense
    # breakpoint()

    bestParams = model_for_mu.state_dict()
    # torch.save({'state_dict': bestParams,}, 'neural_network_for_mu_params.pth')
    torch.save(model_for_mu.mu_net.state_dict(), 'mu_net_from_B.pth')

    model_for_mu_fine_mesh = model_for_mu(torch.cat((fine_points_x.reshape(-1,1).to(torch.float64),fine_points_y.reshape(-1,1).to(torch.float64)),dim=1))
    true_sol_fine_mesh = u_true(fine_points_x.reshape(-1,1).to(torch.float64),fine_points_y.reshape(-1,1).to(torch.float64), problem)
    # squared_diff = torch.square(model_for_mu_fine_mesh - true_sol_fine_mesh)
    
    return X, Y, u_pred, mu_pred, u_pred_on_mesh_reshaped, mu_on_mesh_reshaped, mu_true_array, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np, L2_error_for_u, L2_error_for_mu, max_error_for_u, max_error_for_mu


























    # if problem == 'nonlinear' or problem == 'discontinuous' or problem == 'adr':
    #     residual = 0
    #     for i in range(evaluation_points.shape[0]):
    #         residual = residual + weights[i]*true_model_for_mu_diff[i]
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

    # max_error = max(np.abs(fem_interpolated_reshaped-model_for_mu_on_mesh_reshaped))
    # mean_error = np.mean(np.abs(fem_interpolated_reshaped-model_for_mu_on_mesh_reshaped))

    # print("Max error is ", max_error)
    # print("Mean error is ", mean_error)


    #Analyze the Hessian matrix 

    # # After training or at checkpoints, compute the Hessian
    # def compute_hessian(x, y, samples_f,model_for_mu):
    #     # Forward pass
    #     outputs = model_for_mu(inputs)
    #     loss = interior_loss_func(x, y, samples_f, model_for_mu)

    #     # Compute first derivatives
    #     grads = torch.autograd.grad(loss, model_for_mu.parameters(), create_graph=True)

    #     hessian = []
    #     for g in grads:
    #         row = []
    #         for i in range(len(g)):
    #             # Compute second derivatives
    #             second_derivative = torch.autograd.grad(g[i], model_for_mu.parameters(), retain_graph=True)
    #             row.append(second_derivative)
    #         hessian.append(row)

    #     return hessian

    # # Example usage
    # inputs = fem_data[:,:2]
    # targets = fem_data[:,2]
    # samples_f = f_true(inputs[:,0],inputs[:,1])
    # hessian = compute_hessian(x, y, samples_f, model_for_mu)

