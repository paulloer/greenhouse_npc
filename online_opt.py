import pandas as pd
import numpy as np
import torch
import os
import time
from datetime import datetime, timedelta
torch.manual_seed(42)
np.random.seed(42)

from utilities import SimpleTemporalFusionTransformer, standardize, destandardize, device
from optimizer import optimize_U, plot_optimization, weights, bounds, split_with_transition_points
from open_meteo import get_weather_forecast
from opc_client import gh_client
from constants import ARGUMENTS, OPC_PARAMETERS, URL, TIME_CONTROL, TIME_LEARNING, TIME_FORECAST, COLUMNS, OPC_CONTROL_PARAM_DICT, mean_values, std_values


def online_opt(shared_measurements, shared_vent_postion_targets, shared_model, lock_m, lock_v, lock_p):

    hours_m, hours_p, stride, N_m, N_p = ARGUMENTS.hours_m, ARGUMENTS.hours_p, ARGUMENTS.stride, ARGUMENTS.N_m, ARGUMENTS.N_p

    # greenhouse specific parameter names
    state_features = ['Temperature_inside', 'Humidity_inside']
    control_features_GH13 = ['Vent_S1_Roof_1', 'Vent_S1_Roof_2', 'Vent_S1_Roof_3', 
                                    'Vent_S1_Side_N', 'Vent_S1_Side_NW', 'Vent_S1_Side_S', 'Vent_S1_Side_SW', 
                                    'Vent_S2_Roof_1', 'Vent_S2_Roof_2', 'Vent_S2_Roof_3',
                                    'Vent_S2_Side_E', 'Vent_S2_Side_N', 'Vent_S2_Side_S']
    control_features_GH13 = [control_features_GH13[4]]  # roof vents' encoders are not functioning right now
    disturbance_features = ['Temperature_outside', 'Humidity_outside', 'Radiation_inside', 'Radiation_outside', 'Wind_speed_outside']

    # Create instance of transformer and load weights
    model = SimpleTemporalFusionTransformer(
        len(state_features), len(control_features_GH13), len(disturbance_features),
        hidden_dim=ARGUMENTS.hidden_size,
        num_heads=ARGUMENTS.attention_head_size,
        num_layers=ARGUMENTS.num_lstm_layers, 
        # dropout_prob=ARGUMENTS.dropout_prob
        dropout_prob=0
    ).to(device)

    optimizer_weights = weights(Q=[10, 0], 
                                C=[1, 0], 
                                R=1, 
                                alpha_wind=1e3, 
                                alpha_window=1e3)
        
    window_opening_time = 5*60 # sec
    Ts = 30  # time step of data
    window_ub =  Ts/window_opening_time*100  # max 10% per 30 seconds
    wind_ub = 30

    optimizer_bounds = bounds(temp_target=25, # TODO change these values to the current climate
                              temp_lower=20, 
                              temp_upper=30, 
                              humid_target=70, 
                              humid_lower=50, 
                              humid_upper=90, 
                              window_ub=window_ub, 
                              wind_ub=wind_ub)

    # Initial setup
    last_forecast_time = datetime.min
    update_interval_forecast = timedelta(seconds=TIME_FORECAST)
    last_control_time = datetime.min
    update_interval_control = timedelta(seconds=TIME_CONTROL)
    last_learning_time = datetime.min
    update_interval_learning = timedelta(seconds=TIME_LEARNING)

    with lock_p:
        model_path = shared_model['path']
        mean_values = shared_model['mean_values']
        std_values = shared_model['std_values']
    print(f'Loading model state dict from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device.type), weights_only=True))
    model.to(device)
    model.train()

    X_cl, U_cl, P_cl = [], [], []
    
    # min number of samples required:
    n_samples_target = N_m*stride

    optimal_U = None
    
    while True:
        current_time = datetime.now()
        current_time_fmt = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if forecast update is needed
        if current_time - last_forecast_time >= update_interval_forecast:
            weather_forecast = get_weather_forecast()
            last_forecast_time = current_time
            print(f"Updated forecast at {current_time_fmt}")

        with lock_p:
            if shared_model['path'] != model_path:
                model_path = shared_model['path']
                mean_values = shared_model['mean_values']
                std_values = shared_model['std_values']
                print(f'Loading model state dict from {model_path}')
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device.type), weights_only=True))
                model.to(device)
                model.train()

        tic = time.time()
        if current_time - last_control_time >= update_interval_control:
            last_control_time = datetime.now()

            # do the optimisation
            with lock_m:
                n_samples = len(shared_measurements)
            if n_samples < n_samples_target:
                print(f"Not enough data for prediction available. Status: [{n_samples}/{n_samples_target}]")
            else:
                with lock_m:
                    measurements = pd.DataFrame(shared_measurements[-n_samples_target:], columns=COLUMNS)

                indices = sorted(list(range(len(measurements)-1, -1, -stride)))
                X_m = measurements[state_features].iloc[indices].to_numpy()
                U_m = measurements[control_features_GH13].iloc[indices].to_numpy()
                P_m = measurements[disturbance_features].iloc[indices].to_numpy()
                # print(f"X_m: {X_m}")
                # print(f"U_m: {U_m}")
                idx = weather_forecast[weather_forecast['Date'].str.startswith((current_time+timedelta(seconds=TIME_CONTROL)).strftime('%Y-%m-%d %H:%M'))].index[0]
                P_p = weather_forecast[idx:idx+N_p*stride:stride]
                P_p = P_p.drop('Date', axis=1).to_numpy()

                parameter = np.array(P_p.squeeze())

                X_m[:,0] = standardize(X_m[:,0], 'Temperature_inside', mean_values, std_values)
                X_m[:,1] = standardize(X_m[:,1], 'Humidity_inside', mean_values, std_values)
                U_m = standardize(U_m, 'Vent_S1_Roof_1', mean_values, std_values)
                for i,name in enumerate(disturbance_features):
                    P_m[:,i] = standardize(P_m[:,i], name, mean_values, std_values) 
                    P_p[:,i] = standardize(P_p[:,i], name, mean_values, std_values)

                X_m, U_m, P_m, P_p = torch.tensor(X_m, dtype=torch.float32), torch.tensor(U_m, dtype=torch.float32), torch.tensor(P_m, dtype=torch.float32), torch.tensor(P_p, dtype=torch.float32)
                X_m, U_m, P_m, P_p = X_m.unsqueeze(0), U_m.unsqueeze(0), P_m.unsqueeze(0), P_p.unsqueeze(0)

                if optimal_U is not None:
                    U_initial = torch.tensor(standardize(optimal_U.squeeze(), 'Vent_S1_Roof_1', mean_values, std_values), dtype=torch.float32, requires_grad=True).to(device)
                    U_initial = torch.nn.Parameter(U_initial.unsqueeze(-1)).to(device)
                else:
                    U_initial = np.random.uniform(0, 100, N_p)
                    # U_initial = np.zeros(N_p) + 50
                    U_initial = torch.tensor(standardize(U_initial, 'Vent_S1_Roof_1', mean_values, std_values), dtype=torch.float32, requires_grad=True).to(device)
                    U_initial = torch.nn.Parameter(U_initial.unsqueeze(-1)).to(device)
                # Optimization loop
                num_iters = 3000
                learning_rate = 1e-2
    
                optimal_X, optimal_U, _, _ = optimize_U('Vent_S1_Roof_1',
                                                        model, device, 
                                                        optimizer_weights, optimizer_bounds,
                                                        X_m, U_m, P_m, U_initial, P_p, hours_p, N_p,
                                                        '.', ARGUMENTS.config_string, 
                                                        stride=stride,
                                                        num_iters=num_iters, learning_rate=learning_rate, 
                                                        print_loss=True,
                                                        patience=250)
                
                optimal_X[:,0] = destandardize(optimal_X[:,0], 'Temperature_inside', mean_values, std_values)
                optimal_X[:,1] = destandardize(optimal_X[:,1], 'Humidity_inside', mean_values, std_values)
    
                optimal_U = destandardize(optimal_U, 'Vent_S1_Roof_1', mean_values, std_values)


                plot_optimization(optimal_X, optimal_U, optimizer_bounds, parameter, hours_p, N_p, '.', filename=f'optimization_result_{num_iters}iters_{ARGUMENTS.model_name}')

    
                # if device.type != 'cpu':
                #     u_next = optimal_U[0].cpu().numpy()
                # else:
                #     u_next = optimal_U[0].numpy()

                u_next = max(0, min(optimal_U[0], 100))
                print(f'Next target control is {u_next}')

                with lock_v:
                    shared_vent_postion_targets.append(u_next)
            toc = time.time()
            print(f'Time elapsed for online optimization: {(toc-tic):.3f}')
            
        time.sleep(0.1)




