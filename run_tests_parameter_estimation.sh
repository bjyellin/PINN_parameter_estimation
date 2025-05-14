#!/bin/bash

# Run the tests for the parameter estimation

EXPERIMENT="estimate_u_and_mu_single"

#Separated approach: First learn u with data and BCs, then learn mu with the pde
#Change these parameters so they're consistent with "estimate_u_and_mu_single"
if [ "$EXPERIMENT" = "estimate_u_single" ]; then
python parameter_estimation_separated_driver.py --mode estimate_u --problem constant_mu --level single --num_levels 1 --num_real_data_points 60 --num_phys_points 400 --num_boundary_points 800 --num_epochs 3000 --order 2 --training_loop 0
fi

#Change these parameters so they're consistent with "estimate_u_and_mu_single"
if [ "$EXPERIMENT" = "estimate_mu_single" ]; then
python parameter_estimation_separated_driver.py --mode estimate_mu --problem constant_mu --level single --num_levels 3 --num_phys_points 100 --num_boundary_points 800 --num_epochs 1000 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "estimate_u_and_mu_single" ]; then
python parameter_estimation_separated_driver.py --mode estimate_u_and_mu --problem constant_mu --level single --num_levels 1 --boundary_weight 500 --physics_weight 0 --true_data_weight 1 --num_phys_points 60 --num_boundary_points 800 --num_epochs 3000 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "estimate_u_and_mu_multilevel" ]; then
python parameter_estimation_separated_driver.py --mode estimate_u_and_mu --problem constant_mu --level multilevel --num_levels 1 --boundary_weight 500 --physics_weight 0 --true_data_weight 1 --num_phys_points 40 --num_boundary_points 800 --num_epochs 1 --order 2 --training_loop 0
fi



if [ "$EXPERIMENT" = "very_simple_single" ]; then
python parameter_estimation_driver.py --mode estimate_parameters --problem simpler_cont_laplace --level single --num_levels 1 --num_phys_points 100 --num_boundary_points 800 --num_epochs 4000 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "simpler_cont_laplace_single" ]; then
python parameter_estimation_driver.py --mode estimate_parameters --problem simpler_cont_laplace --level single --num_levels 1 --num_phys_points 100 --num_boundary_points 800 --num_epochs 100 --physics_weight 1 --boundary_weight 500 --true_data_weight 1 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "simpler_cont_laplace_multilevel" ]; then
python parameter_estimation_driver.py --mode estimate_parameters --problem simpler_cont_laplace --level multilevel --num_levels 5 --num_phys_points 40 --num_boundary_points 800 --num_epochs 3000 --physics_weight 1 --boundary_weight 500 --true_data_weight 1 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "cont_laplace_single" ]; then
python parameter_estimation_driver.py --mode estimate_parameters --problem cont_laplace --level single --num_levels 5 --num_phys_points 40 --num_boundary_points 800 --num_epochs 1 --physics_weight 1 --boundary_weight 500 --true_data_weight 1 --order 2 --training_loop 0
fi

if [ "$EXPERIMENT" = "cont_laplace_multilevel" ]; then
python parameter_estimation_driver.py --mode estimate_parameters --problem cont_laplace --level multilevel --num_levels 5 --num_phys_points 40 --num_boundary_points 800 --num_epochs 8000 --physics_weight 1 --boundary_weight 500 --true_data_weight 1 --order 2 --training_loop 0
fi

