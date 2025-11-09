import torch
import torch.nn as nn
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from constants import ARGUMENTS, TIME_LEARNING, CSV_PATH
from utilities import train, test, plot_loss, SimpleTemporalFusionTransformer, GreenhouseDatasetHandler, device

def online_learning(shared_model, lock_p):

    hours_m, hours_p, stride, N_m, N_p = ARGUMENTS.hours_m, ARGUMENTS.hours_p, ARGUMENTS.stride, ARGUMENTS.N_m, ARGUMENTS.N_p
    model_name, model_path = ARGUMENTS.model_name, ARGUMENTS.model_path

    with lock_p:
        shared_model['path'] = model_path

    state_features = ['Temperature_inside', 'Humidity_inside']
    control_features_GH13 = ['Vent_S1_Roof_1', 'Vent_S1_Roof_2', 'Vent_S1_Roof_3', 
                                    'Vent_S1_Side_N', 'Vent_S1_Side_NW', 'Vent_S1_Side_S', 'Vent_S1_Side_SW', 
                                    'Vent_S2_Roof_1', 'Vent_S2_Roof_2', 'Vent_S2_Roof_3',
                                    'Vent_S2_Side_E', 'Vent_S2_Side_N', 'Vent_S2_Side_S']
    control_features_GH13 = [control_features_GH13[0]]
    disturbance_features = ['Temperature_outside', 'Humidity_outside', 'Radiation_inside', 'Radiation_outside', 'Wind_speed_outside']

    model = SimpleTemporalFusionTransformer(
        len(state_features), len(control_features_GH13), len(disturbance_features),
        hidden_dim=ARGUMENTS.hidden_size,
        num_heads=ARGUMENTS.attention_head_size,
        num_layers=ARGUMENTS.num_lstm_layers, 
        dropout_prob=ARGUMENTS.dropout_prob
    ).to(device)

    last_time = datetime.min
    update_interval = timedelta(seconds=TIME_LEARNING)

    while True:
        current_time = datetime.now()
        current_time_fmt = current_time.strftime('%Y-%m-%d %H:%M:%S')

        if current_time - last_time >= update_interval:
            last_time = datetime.now()

            data = pd.read_csv(CSV_PATH, delimiter=',')

            # always train on newest data, validate and test on oldest data
            split_index_train = int(0.2 * len(data))
            split_index_val = int(0.1 * len(data)) #TODO these are not useable data
            split_index_test = int(0.0 * len(data))
            dataset = GreenhouseDatasetHandler(data=data, 
                                                train_data=data.iloc[split_index_train:],
                                                val_data=data.iloc[split_index_val:split_index_train], 
                                                test_data=data.iloc[split_index_test:split_index_val], 
                                                state_features=state_features, 
                                                control_features=control_features_GH13,
                                                disturbance_features=disturbance_features, 
                                                seq_len=N_m, pred_len=N_p, stride=stride, batch_size=ARGUMENTS.batch_size)
            
            shared_model['mean_values'] = dataset.mean_values
            shared_model['std_values'] = dataset.std_values

            print(f'Loading model state dict from {ARGUMENTS.model_path}')
            model.load_state_dict(torch.load(ARGUMENTS.model_path, map_location=torch.device(device.type), weights_only=True))
            model.to(device)
            model.train()


            criterion = nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = torch.optim.SGD(model.parameters(), lr=ARGUMENTS.learning_rate, weight_decay=ARGUMENTS.weight_decay)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=ARGUMENTS.num_epochs)

            if not os.path.isfile(model_path):
                train_loss, val_loss = train(model, dataset.train_loader, dataset.val_loader, criterion, optimizer, num_epochs=ARGUMENTS.num_epochs, scheduler=scheduler)

                # update model path to save checkpoint
                model_path = f"{ARGUMENTS.save_dir}/{model_name}_{current_time_fmt}.pth"
                ARGUMENTS.model_path = model_path
                print(f'Saving {model_name} state dict to {model_path}')
                torch.save(model.state_dict(), model_path)
                plot_loss(train_loss, val_loss, f'{ARGUMENTS.save_dir}/loss_plot_{model_name}_{current_time_fmt}.png')
            else:
                print(f'Loading {model_name} state dict from {model_path}')
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device.type), weights_only=True))

            test(model, dataset, ARGUMENTS.save_dir, f'{model_name}_{current_time_fmt}')


        time.sleep(3600)