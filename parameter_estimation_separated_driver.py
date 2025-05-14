from parameter_estimation_for_u import estimate_u
from parameter_estimation_for_mu import estimate_mu
from parameter_estimation_for_u_and_mu import estimate_u_and_mu

import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import os

# plt.rcParams.update({
#     'axes.labelsize': 14,       # for xlabel and ylabel
#     'axes.titlesize': 16,       # for the title
#     'xtick.labelsize': 12,      # x-axis tick labels
#     'ytick.labelsize': 12,      # y-axis tick labels
#     'legend.fontsize': 12,      # legend
#     'font.size': 12             # base font size (affects others if specific keys aren't set)
# })

# # problem = 'simpler_cont_laplace'
# # level = 'single'
# # num_levels = 1
# # mesh_size = 10
# # num_real_data_points = 200
# # num_phys_points = 200
# # num_boundary_points = 1000
# # mu=1
# # beta0=4
# # beta1 = 4
# # num_epochs = 2000
# # physics_weight = 10
# # boundary_weight = 30
# # fem_data_weight = 0
# # true_data_weight = 1
# # stabilization_weight = 0
# # order = 2
# # training_loop = 0


parser = argparse.ArgumentParser(description="Run different test cases")
parser.add_argument('--mode', choices=['estimate_parameters', 'estimate_u', 'estimate_mu', 'estimate_u_and_mu'])
parser.add_argument('--problem', choices=['simpler_cont_laplace', 'cont_laplace', 'very_simple', 'constant_mu']) 
parser.add_argument('--level', choices=['single', 'multilevel'], default='single')
parser.add_argument('--num_levels', type=int, default=1)
parser.add_argument('--num_real_data_points', type=int, default=200)
parser.add_argument('--num_phys_points', type=int, default=200)
parser.add_argument('--num_boundary_points', type=int, default=1000)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--physics_weight', type=float, default=0)
parser.add_argument('--boundary_weight', type=float, default=500)
parser.add_argument('--true_data_weight', type=float, default=1)
parser.add_argument('--order', type=float, default=2)
parser.add_argument('--training_loop', type=int, default=0)
parser.add_argument('--step', type=int, default=0)
parser.add_argument('--iteration', type=int, default=0)
parser.add_argument('--mu_iteration', type=int, default=0)

# First pass to check conditions
known_args, remaining_argv = parser.parse_known_args()

# Add conditionally required arguments
# if known_args.testing == "stabilized_fem_data":
#     parser.add_argument("--stabilization_weight", type=int, choices=[0, 1])
#     parser.add_argument("--num_fem_points", type=int, required=True)


# Final argument parsing
args = parser.parse_args()

problem = args.problem
level=args.level
num_levels = args.num_levels
num_real_data_points = args.num_real_data_points
num_phys_points=args.num_phys_points
num_boundary_points=args.num_boundary_points
num_epochs = args.num_epochs
physics_weight = args.physics_weight
boundary_weight = args.boundary_weight
true_data_weight = args.true_data_weight
order = args.order
training_loop = args.training_loop
step = args.step
iteration = args.iteration

#STILL NEED TO INCREASE THE NUMBER OF POINTS EACH ITERATION OF THE LOOP FOR MULTILEVEL (DO LIKE I DID IN THE OTHER FILE) 


def learn_u(problem,level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration, phys_weight):
    
    num_real_data_values = num_real_data_points#[int(i) for i in np.round(np.linspace(5, num_real_data_points, num_levels))]
    num_phys_values = num_phys_points#[int(i) for i in np.round(np.linspace(5, num_phys_points, num_levels))]
    num_boundary_values = num_boundary_points#[int(i) for i in np.round(np.linspace(200, num_boundary_points, num_levels))]
    # breakpoint()
    L2_errors_for_u = []
    max_errors_for_u = []
    L2_errors_for_mu = []
    max_errors_for_mu = []

    X, Y, u_pred, u_on_mesh_reshaped, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np = estimate_u(problem, step, num_real_data_values, num_phys_values, num_boundary_values, num_epochs, order, training_loop, iteration, phys_weight)
    # breakpoint()
    # for i in range(num_levels):
    #     # breakpoint()
    #     X, Y, u_pred, u_on_mesh_reshaped, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np = estimate_u(problem, step, num_real_data_values[i], num_phys_values[i], num_boundary_values[i], num_epochs, order, training_loop, iteration)
        


        # L2_errors_for_u.append(L2_error_for_u)
        # max_errors_for_u.append(max_error_for_u)

        # L2_errors_for_mu.append(L2_error_for_mu)
        # max_errors_for_mu.append(max_error_for_mu)

        # if level=='multilevel':
        #     training_loop += 1    

    np.savetxt(f"X_{step}.txt", X.detach().numpy())
    np.savetxt(f"Y_{step}.txt", Y.detach().numpy())
    np.savetxt(f"u_pred_{step}.txt", u_pred.detach().numpy())
    # np.savetxt(f"mu_field_{step}.txt", mu_field.detach().numpy())
    np.savetxt(f"real_data_points_x_np_{step}.txt", real_data_points_x_np)
    np.savetxt(f"real_data_points_y_np_{step}.txt", real_data_points_y_np)
        
    # np.savetxt("L2_errors_for_u.txt", L2_errors_for_u)
    # np.savetxt("max_errors_for_u.txt", max_errors_for_u)
    # np.savetxt("L2_errors_for_mu.txt", L2_errors_for_mu)
    # np.savetxt("max_errors_for_mu.txt", max_errors_for_mu)

    step += 1
    # breakpoint()
    plt.clf()
    # breakpoint()
    plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), u_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='u')
    plt.title('u on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'u_on_mesh_{step}.png')

    # plt.clf()
    # plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), mu_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='mu')
    # plt.title('mu on Mesh')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(f'mu_on_mesh_{step}.png')

    # breakpoint()
    # plt.clf()
    # plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), mu_true_array.numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='mu_true')
    # plt.title('mu True on Mesh')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(f'mu_true_on_mesh_{step}.png')


    # breakpoint()
    

    # # Plotting the errors
    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_mu, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_mu, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Parameter Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Parameter Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'parameter_estimation_errors_{level}.png')

    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_u, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_u, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Solution Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Solution Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'solution_estimation_errors_{level}.png')

    return L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu


def learn_mu(problem,level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration, mu_iteration):
    
    num_real_data_values = num_real_data_points#[int(i) for i in np.round(np.linspace(5, num_real_data_points, num_levels))]
    num_phys_values = num_phys_points#[int(i) for i in np.round(np.linspace(5, num_phys_points, num_levels))]
    num_boundary_values = num_boundary_points#[int(i) for i in np.round(np.linspace(200, num_boundary_points, num_levels))]
    # breakpoint()
    L2_errors_for_u = []
    max_errors_for_u = []
    L2_errors_for_mu = []
    max_errors_for_mu = []

    X, Y, u_pred, mu_field, u_pred_on_mesh_reshaped, mu_on_mesh_reshaped, mu_true_array, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np, L2_error_for_u, L2_error_for_mu, max_error_for_u, max_error_for_mu = estimate_mu(problem, step, num_real_data_values, num_phys_values, num_boundary_values, num_epochs, order, training_loop, iteration)
    # for i in range(num_levels):
    #     X, Y, u_pred, mu_field, u_pred_on_mesh_reshaped, mu_on_mesh_reshaped, mu_true_array, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np, L2_error_for_u, L2_error_for_mu, max_error_for_u, max_error_for_mu = estimate_mu(problem, step, num_real_data_values[i], num_phys_values[i], num_boundary_values[i], num_epochs, order, training_loop, iteration)
        

    #     L2_errors_for_u.append(L2_error_for_u)
    #     max_errors_for_u.append(max_error_for_u)

        # L2_errors_for_mu.append(L2_error_for_mu)
        # max_errors_for_mu.append(max_error_for_mu)

    if level=='multilevel':
        training_loop += 1    

    np.savetxt(f"X_{step}.txt", X.detach().numpy())
    np.savetxt(f"Y_{step}.txt", Y.detach().numpy())
    np.savetxt(f"u_pred_{step}.txt", u_pred.detach().numpy())
    # np.savetxt(f"mu_field_{step}.txt", mu_field.detach().numpy())
    np.savetxt(f"real_data_points_x_np_{step}.txt", real_data_points_x_np)
    np.savetxt(f"real_data_points_y_np_{step}.txt", real_data_points_y_np)
        
    np.savetxt("L2_errors_for_u.txt", L2_errors_for_u)
    np.savetxt("max_errors_for_u.txt", max_errors_for_u)
    # np.savetxt("L2_errors_for_mu.txt", L2_errors_for_mu)
    # np.savetxt("max_errors_for_mu.txt", max_errors_for_mu)

    step += 1
    # breakpoint()
    # plt.clf()
    # # breakpoint()
    # plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), u_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='u')
    # plt.title('u on Mesh')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(f'u_on_mesh_{step}.png')

    plt.clf()
    # breakpoint()
    fine_points_x = fine_points_x.reshape(int(fine_points_x.shape[0]**.5), int(fine_points_x.shape[0]**0.5))
    fine_points_y = fine_points_y.reshape(int(fine_points_y.shape[0]**.5), int(fine_points_y.shape[0]**0.5))
    plt.contourf(fine_points_x.detach().numpy(), fine_points_y.detach().numpy(), mu_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='mu')
    plt.title('mu on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'mu_on_mesh_{step}.png')

    print("mu_on_mesh_reshaped shape", mu_on_mesh_reshaped.shape)
    
    # breakpoint()
    real_data_points_x_np = real_data_points_x_np.reshape(int(real_data_points_x_np.shape[0]**.5), int(real_data_points_x_np.shape[0]**0.5))
    real_data_points_y_np = real_data_points_y_np.reshape(int(real_data_points_y_np.shape[0]**.5), int(real_data_points_y_np.shape[0]**0.5))
    plt.clf()
    plt.contourf(real_data_points_x_np, real_data_points_y_np, mu_true_array.numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='mu_true')
    plt.title('mu True on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'mu_true_on_mesh_{step}.png')
    # breakpoint()

    # breakpoint()
    

    # # Plotting the errors
    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_mu, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_mu, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Parameter Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Parameter Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'parameter_estimation_errors_{level}.png')

    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_u, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_u, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Solution Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Solution Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'solution_estimation_errors_{level}.png')

    return L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu

def learn_u_and_mu(problem,level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration, mu_iteration):
    num_real_data_values = num_real_data_points#[int(i) for i in np.round(np.linspace(5, num_real_data_points, num_levels))]
    num_phys_values = num_phys_points#[int(i) for i in np.round(np.linspace(5, num_phys_points, num_levels))]
    num_boundary_values = num_boundary_points#[int(i) for i in np.round(np.linspace(200, num_boundary_points, num_levels))]
    # breakpoint()
    L2_errors_for_u = []
    max_errors_for_u = []
    L2_errors_for_mu = []
    max_errors_for_mu = []

    X, Y, u_pred, mu_field, u_pred_on_mesh_reshaped, mu_on_mesh_reshaped, mu_true_array, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np, L2_error_for_u, L2_error_for_mu, max_error_for_u, max_error_for_mu = estimate_u_and_mu(problem, step, num_real_data_values, num_phys_values, num_boundary_values, num_epochs, order, training_loop, iteration)
    # for i in range(num_levels):
    #     X, Y, u_pred, mu_field, u_pred_on_mesh_reshaped, mu_on_mesh_reshaped, mu_true_array, fine_points_x, fine_points_y, real_data_points_x_np, real_data_points_y_np, L2_error_for_u, L2_error_for_mu, max_error_for_u, max_error_for_mu = estimate_mu(problem, step, num_real_data_values[i], num_phys_values[i], num_boundary_values[i], num_epochs, order, training_loop, iteration)
        

    #     L2_errors_for_u.append(L2_error_for_u)
    #     max_errors_for_u.append(max_error_for_u)

        # L2_errors_for_mu.append(L2_error_for_mu)
        # max_errors_for_mu.append(max_error_for_mu)

    if level=='multilevel':
        training_loop += 1    

    np.savetxt(f"X_{step}.txt", X.detach().numpy())
    np.savetxt(f"Y_{step}.txt", Y.detach().numpy())
    np.savetxt(f"u_pred_{step}.txt", u_pred.detach().numpy())
    # np.savetxt(f"mu_field_{step}.txt", mu_field.detach().numpy())
    np.savetxt(f"real_data_points_x_np_{step}.txt", real_data_points_x_np)
    np.savetxt(f"real_data_points_y_np_{step}.txt", real_data_points_y_np)
        
    np.savetxt("L2_errors_for_u.txt", L2_errors_for_u)
    np.savetxt("max_errors_for_u.txt", max_errors_for_u)
    # np.savetxt("L2_errors_for_mu.txt", L2_errors_for_mu)
    # np.savetxt("max_errors_for_mu.txt", max_errors_for_mu)

    step += 1
    # breakpoint()
    # plt.clf()
    # # breakpoint()
    # plt.contourf(fine_points_x.numpy(), fine_points_y.numpy(), u_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    # plt.colorbar(label='u')
    # plt.title('u on Mesh')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(f'u_on_mesh_{step}.png')

    plt.clf()
    # breakpoint()
    fine_points_x = fine_points_x.reshape(int(fine_points_x.shape[0]**.5), int(fine_points_x.shape[0]**0.5))
    fine_points_y = fine_points_y.reshape(int(fine_points_y.shape[0]**.5), int(fine_points_y.shape[0]**0.5))
    plt.contourf(fine_points_x.detach().numpy(), fine_points_y.detach().numpy(), mu_on_mesh_reshaped.detach().numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='mu')
    plt.title('mu on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'mu_on_mesh_{step}.png')

    print("mu_on_mesh_reshaped shape", mu_on_mesh_reshaped.shape)
    
    # breakpoint()
    real_data_points_x_np = real_data_points_x_np.reshape(int(real_data_points_x_np.shape[0]**.5), int(real_data_points_x_np.shape[0]**0.5))
    real_data_points_y_np = real_data_points_y_np.reshape(int(real_data_points_y_np.shape[0]**.5), int(real_data_points_y_np.shape[0]**0.5))
    plt.clf()
    plt.contourf(real_data_points_x_np, real_data_points_y_np, mu_true_array.numpy(), levels=100, cmap='viridis')
    plt.colorbar(label='mu_true')
    plt.title('mu True on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'mu_true_on_mesh_{step}.png')
    # breakpoint()

    # breakpoint()
    

    # # Plotting the errors
    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_mu, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_mu, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Parameter Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Parameter Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'parameter_estimation_errors_{level}.png')

    # plt.figure(figsize=(10, 5))
    # plt.semilogy(range(1, num_levels + 1), L2_errors_for_u, label='L2 Error')
    # plt.semilogy(range(1, num_levels + 1), max_errors_for_u, label='Max Error')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # if level == 'single':
    #     plt.title('Solution Estimate Errors vs. Level (Single Level)')
    # if level == 'multilevel':
    #     plt.title('Solution Estimate Errors vs. Level (Multilevel)')
    
    # plt.title('Parameter Estimate Errors vs. Resolution')
    # plt.xlabel('Resolution')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig(f'solution_estimation_errors_{level}.png')

    return L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu


if args.mode == 'estimate_u':
    # breakpoint()
    L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu = learn_u(problem, level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, num_epochs, order, training_loop, iteration)

if args.mode == 'estimate_mu':
    L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu = learn_mu(problem, level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, num_epochs,  order, training_loop, iteration)

if args.mode == 'estimate_u_and_mu':
    print("Starting training for u with only data fitting")
    L2_errors_for_u, max_errors_for_u, L2_errors_for_mu, max_errors_for_mu = learn_u(problem, level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, int(num_epochs/50), order, training_loop, iteration, phys_weight=0)
    print("just finished training for u")
    
    print("Starting training for mu")
    for i in range(3): 
        L2_errors_for_u_2, max_errors_for_u_2, L2_errors_for_mu_2, max_errors_for_mu_2 = learn_mu(problem, level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, int(num_epochs), order, training_loop, iteration, mu_iteration=i)
        print("just finished training for mu")

        # print("About to start joint training using learned u and learned mu")
        # L2_errors_for_u_3, max_errors_for_u_3, L2_errors_for_mu_3, max_errors_for_mu_3 = learn_u_and_mu(problem, level, step, num_levels, num_real_data_points, num_phys_points, num_boundary_points, int(num_epochs), order, training_loop, iteration, mu_iteration=i+1)

