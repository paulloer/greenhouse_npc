import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import os
torch.manual_seed(42)
np.random.seed(42)

from constants import STAND_VALUES

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')
num_workers = 128 if device.type == "cuda" else 0
print(f'Using {num_workers} workers')

class GreenhouseDatasetHandler():
    def __init__(self, data, train_data, val_data, test_data, state_features, control_features, disturbance_features, seq_len, pred_len, stride, batch_size, forecast=False, closed_loop=False):

        self.stride = stride
        forecast_features = []
        for feature in disturbance_features:
            forecast_features.append(feature + '_forecast')
            
        if forecast:
            self.features = state_features + control_features + disturbance_features + forecast_features
            self.forecast_features = forecast_features
        else:
            self.features = state_features + control_features + disturbance_features

        data = data.copy()
        train_data = train_data.copy()
        val_data = val_data.copy()
        test_data = test_data.copy()

        self.mean_values = {feature: np.mean(data[feature]) for feature in self.features}
        self.std_values = {feature: np.sqrt(sum((data[feature] - self.mean_values[feature])**2) / len(data[feature])) for feature in self.features}
        
        for feature in self.features:
            data[feature] = (data[feature]-self.mean_values[feature])/self.std_values[feature]
            train_data[feature] = (train_data[feature]-self.mean_values[feature])/self.std_values[feature]
            val_data[feature] = (val_data[feature]-self.mean_values[feature])/self.std_values[feature]
            test_data[feature] = (test_data[feature]-self.mean_values[feature])/self.std_values[feature]
        
        # create datasets
        self.full_dataset = GreenhouseDataset(data, state_features, control_features, disturbance_features, forecast_features, seq_len, pred_len, stride, forecast, closed_loop)
        self.train_dataset = GreenhouseDataset(train_data, state_features, control_features, disturbance_features, forecast_features, seq_len, pred_len, stride, forecast, closed_loop)
        self.val_dataset = GreenhouseDataset(val_data, state_features, control_features, disturbance_features, forecast_features, seq_len, pred_len, stride, forecast, closed_loop)
        self.test_dataset = GreenhouseDataset(test_data, state_features, control_features, disturbance_features, forecast_features, seq_len, pred_len, stride, forecast, closed_loop)

        # data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=1 if device.type == 'cuda' and num_workers > 0 else None)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

class GreenhouseDataset(Dataset):
    def __init__(self, data, state_features, control_features, disturbance_features, forecast_features, seq_length, pred_length, stride, forecast, closed_loop):
        self.data = data
        self.state_features = state_features
        self.control_features = control_features
        self.disturbance_features = disturbance_features
        self.forecast_features = forecast_features
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.stride = stride
        self.forecast = forecast
        self.closed_loop = closed_loop

    def __len__(self):
        return len(self.data) - (self.seq_length + self.pred_length) * self.stride - 1

    def __getitem__(self, idx):
        sequence_begin = idx
        sequence_end = sequence_begin+self.seq_length*self.stride
        prediction_begin = sequence_end - self.stride + 1
        prediction_end = prediction_begin+self.pred_length*self.stride

        if self.closed_loop:
            X_m = self.data[self.state_features].iloc[sequence_begin : sequence_end - self.stride + 1].values
        else:
            X_m = self.data[self.state_features].iloc[sequence_begin : sequence_end : self.stride].values
        U_m = self.data[self.control_features].iloc[sequence_begin : sequence_end : self.stride].values
        P_m = self.data[self.disturbance_features].iloc[sequence_begin : sequence_end : self.stride].values

        X_p = self.data[self.state_features].iloc[prediction_begin : prediction_end : self.stride].values
        U_p = self.data[self.control_features].iloc[prediction_begin : prediction_end : self.stride].values
        if self.forecast:
            P_p = self.data[self.forecast_features].iloc[prediction_begin : prediction_end : self.stride].values
        else:
            P_p = self.data[self.disturbance_features].iloc[prediction_begin : prediction_end : self.stride].values
        return torch.tensor(X_m, dtype=torch.float32), torch.tensor(U_m, dtype=torch.float32), torch.tensor(P_m, dtype=torch.float32), torch.tensor(X_p, dtype=torch.float32), torch.tensor(U_p, dtype=torch.float32), torch.tensor(P_p, dtype=torch.float32)
    

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_prob):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        # self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.dropout(attn_output) + x  # residual
        # x = self.norm(x)  # layer norm
        return x 

class SimpleTemporalFusionTransformer(nn.Module):
    def __init__(self, state_dim, control_dim, param_dim, hidden_dim, num_heads, num_layers=1, dropout_prob=0.3):
        super().__init__()
        self.lstm_encoder = nn.LSTM(state_dim+control_dim+param_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(control_dim+param_dim, hidden_dim, num_layers, batch_first=True)
        self.self_attention = TemporalSelfAttention(hidden_dim, num_heads, dropout_prob)
        self.fc = nn.Linear(hidden_dim, state_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X_m, U_m, P_m, U_p, P_p):
        past, future_known = torch.cat([X_m, U_m, P_m], dim=-1), torch.cat([U_p, P_p], dim=-1)

        # _, (h, c) = self.lstm_encoder(past)
        # decoder_output, _ = self.lstm_decoder(future_known, (h, c))
        # decoder_output = self.dropout(decoder_output)
        # # decoder_output = torch.tanh(decoder_output) 

        # attn_out = self.self_attention(decoder_output)
        # return self.fc(attn_out)

        # Encode past, decode future
        encoder_output, (h, c) = self.lstm_encoder(past)
        decoder_output, _ = self.lstm_decoder(future_known, (h, c))

        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        lstm_output = self.dropout(lstm_output)
        
        attn_out = self.self_attention(lstm_output)

        # Final prediction and slice out only decoder portion
        out = attn_out[:, -decoder_output.shape[1]:, :]
        return self.fc(out)
    

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=10, scheduler=None):
    train_loss, val_loss = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        for i, (X_m, U_m, P_m, X_p, U_p, P_p) in enumerate(train_loader):
            X_m, U_m, P_m, X_p, U_p, P_p = X_m.to(device), U_m.to(device), P_m.to(device), X_p.to(device), U_p.to(device), P_p.to(device)

            # Forward pass
            X_p_pred = model(X_m, U_m, P_m, U_p, P_p)
            loss = criterion(X_p_pred, X_p)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping
            optimizer.step()
            running_train_loss += loss.item()
        
        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss.append(epoch_train_loss)

        # Step the scheduler after each epoch (for LinearLR)
        if scheduler:
            scheduler.step()

        # Validation phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_m, U_m, P_m, X_p, U_p, P_p in val_loader:
                X_m, U_m, P_m, X_p, U_p, P_p = X_m.to(device), U_m.to(device), P_m.to(device), X_p.to(device), U_p.to(device), P_p.to(device)

                X_p_pred = model(X_m, U_m, P_m, U_p, P_p)
                loss = criterion(X_p_pred, X_p)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss.append(epoch_val_loss)

        # Print epoch statistics
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if scheduler:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6e}")
            else:  
                print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}")

        # Check early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0  # Reset counter if validation loss improves
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            break
          
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return train_loss, val_loss


def test(model, handler: GreenhouseDatasetHandler, save_dir: str, filename: str, return_data=False):
    model.eval()
    errors = []
    predictions, actuals = [], []
    future_controls, future_parameters = [], []

    with torch.no_grad():
        for X_m, U_m, P_m, X_p, U_p, P_p in tqdm(handler.test_loader, 'Inference'):
            X_m, U_m, P_m, X_p, U_p, P_p = X_m.to(device), U_m.to(device), P_m.to(device), X_p.to(device), U_p.to(device), P_p.to(device)
            X_p_pred = model(X_m, U_m, P_m, U_p, P_p)
            errors.append(X_p_pred.squeeze(0).cpu().numpy().astype(np.float32) - X_p.squeeze(0).cpu().numpy().astype(np.float32))
            predictions.append(X_p_pred.squeeze(0).cpu().numpy().astype(np.float32))
            actuals.append(X_p.squeeze(0).cpu().numpy().astype(np.float32))
            future_controls.append(U_p.squeeze(0).cpu().numpy().astype(np.float32))
            future_parameters.append(P_p.squeeze(0).cpu().numpy().astype(np.float32))

    errors = np.array(errors).astype(np.float32)
    errors = np.array((destandardize(errors[:, :, 0], 'Temperature_inside', handler.mean_values, handler.std_values, std_only=True),
                       destandardize(errors[:, :, 1], 'Humidity_inside', handler.mean_values, handler.std_values, std_only=True))).transpose(1,2,0).reshape(-1,2)
    
    for i in range(len(predictions)):
        predictions[i] = np.array((destandardize(predictions[i][:, 0], 'Temperature_inside', handler.mean_values, handler.std_values),
                                   destandardize(predictions[i][:, 1], 'Humidity_inside', handler.mean_values, handler.std_values))).T
        actuals[i] = np.array((destandardize(actuals[i][:, 0],'Temperature_inside', handler.mean_values, handler.std_values),
                               destandardize(actuals[i][:, 1],'Humidity_inside', handler.mean_values, handler.std_values))).T
        future_controls[i] = np.array((destandardize(future_controls[i], handler.test_dataset.control_features[0], handler.mean_values, handler.std_values))).T
        future_parameters[i] = np.array((destandardize(future_parameters[i][:, 0], handler.test_dataset.disturbance_features[0], handler.mean_values, handler.std_values),
                                         destandardize(future_parameters[i][:, 1], handler.test_dataset.disturbance_features[1], handler.mean_values, handler.std_values),
                                         destandardize(future_parameters[i][:, 2], handler.test_dataset.disturbance_features[2], handler.mean_values, handler.std_values),
                                         destandardize(future_parameters[i][:, 3], handler.test_dataset.disturbance_features[3], handler.mean_values, handler.std_values),
                                         destandardize(future_parameters[i][:, 4], handler.test_dataset.disturbance_features[4], handler.mean_values, handler.std_values))).T

    print(f"Predictions: {len(predictions)}")
    N_p = int(len(predictions[0][:, 0]))
    stride = handler.stride
    hours_pred = int(len(predictions)/120) # samples every 30 s, TODO: replace with global variables

    mpl.rcParams.update(
    {
        "font.size":       10,         # control font sizes of different elements
        "axes.labelsize":  10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Plot predictions vs actuals
    fig, axs = plt.subplots(2, 1, figsize=(8/2.54, 10/2.54), sharex=True)
    linewidth = .8

    for i in range(len(predictions)):
        if i % int(N_p*stride) == 0:
            n = int(i/(N_p*stride))
            x = np.linspace(n*N_p, (n+1)*N_p-1, N_p)
            axs[0].plot(x, actuals[i][:, 0], color='black', linewidth=linewidth)
            axs[0].plot(x, predictions[i][:, 0], color='purple', linewidth=linewidth)
            axs[1].plot(x, actuals[i][:, 1], color='black', linewidth=linewidth)
            axs[1].plot(x, predictions[i][:, 1], color='blue', linewidth=linewidth)
    
    axs[0].set_ylabel('$\\theta$ [$^\circ$C]')
    axs[0].legend()
    axs[0].grid()
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('$\\phi$ [\\%]')
    axs[1].legend()
    axs[1].grid()

    fig.align_ylabels()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}.pgf')
    plt.savefig(f'{save_dir}/{filename}.png', dpi=300)
    plt.close()


    mpl.use("pdf")

    fig, axs = plt.subplots(5, 1, figsize=(8/2.54, 10/2.54), sharex=True)
    linewidth = .3

    for i in range(len(predictions)):
        x = np.arange(N_p) * stride + i
        axs[0].plot(x, actuals[i][:, 0], color='black', linewidth=linewidth, alpha=0.5)
        axs[0].plot(x, predictions[i][:, 0], color='purple', linewidth=linewidth, alpha=0.5)
        axs[0].plot(x, future_parameters[i][:, 0], color='magenta', linewidth=linewidth, alpha=0.5)
        
        axs[1].plot(x, actuals[i][:, 1], color='black', linewidth=linewidth, alpha=0.5)
        axs[1].plot(x, predictions[i][:, 1], color='blue', linewidth=linewidth, alpha=0.5)
        axs[1].plot(x, future_parameters[i][:, 1], color='cyan', linewidth=linewidth, alpha=0.5)

        axs[2].plot(x, future_controls[i][0], color='limegreen', linewidth=linewidth, alpha=1)
        axs[3].plot(x, future_parameters[i][:, 2], color='gold', linewidth=linewidth, alpha=1)
        axs[3].plot(x, future_parameters[i][:, 3], color='darkorange', linewidth=linewidth, alpha=1)
        axs[4].plot(x, future_parameters[i][:, 4], color='darkcyan', linewidth=linewidth, alpha=1)

    axs[0].set_ylabel('$\\theta$ [$^\circ$C]')
    axs[0].grid()
    axs[0].legend(['$\\theta_{\\mathrm{in}}$', '$\\hat{\\theta}_{\\mathrm{in}}$', '$\\theta_{\\mathrm{out}}$'], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelspacing=0.2, handlelength=.5)
    axs[1].set_ylabel('$\\phi$ [\\%]')
    axs[1].grid()
    axs[1].legend(['$\\phi{\\mathrm{in}}$', '$\\hat{\\phi}_{\\mathrm{in}}$', '$\\phi_{\\mathrm{out}}$'], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelspacing=0.2, handlelength=.5)
    axs[2].set_ylabel('$u$ [\\%]')
    axs[2].set_yticks([0,20,40,60,80,100])
    axs[2].set_yticklabels([0,20,40,60,80,100])
    axs[2].grid()
    axs[2].legend(['$u_1$'], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelspacing=0.2, handlelength=.5)
    axs[3].set_ylabel('$R$ [W/m$^2$]')
    axs[3].grid()
    axs[3].legend(['$R_{\\mathrm{in}}$', '$R_{\\mathrm{out}}$'], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelspacing=0.2, handlelength=.5)
    axs[4].set_ylabel('$v$ [m/s]')
    axs[4].grid()
    axs[4].legend(["$v_{\\mathrm{max}}$"], loc="center left", bbox_to_anchor=(1, .5), frameon=False, labelspacing=0.2, handlelength=.5) 

    if hours_pred <= 24:
        h = hours_pred + int(N_p * stride / 120)
        axs[-1].set_xlabel("Time [h]")
        
        even_hours = np.arange(0, h + 1, 2)
        tick_positions = even_hours * 120

        axs[-1].set_xticks(tick_positions)
        axs[-1].set_xticklabels(even_hours)
    else:
        axs[-1].set_xlabel("Time [days]")
        axs[-1].set_xticks(np.linspace(0, int(hours_pred/24), int(hours_pred/24)+1, dtype=int)*24*120)
        axs[-1].set_xticklabels(np.linspace(0, int(hours_pred/24), int(hours_pred/24)+1, dtype=int))

    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}_all.png', dpi=300)
    plt.close()

    mpl.use("pgf")
    mpl.rcParams.update(
    {
        "font.size":       10,         # control font sizes of different elements
        "axes.labelsize":  10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # RMSE
    rmse_T, rmse_H, std_T, std_H = np.sqrt(np.mean(errors[:,0]**2)), np.sqrt(np.mean(errors[:,1]**2)), np.std(errors[:,0]), np.std(errors[:,1])

    print(f"RMS Temperature Error of {rmse_T:.4f} with standard deviation {std_T:.4f}")
    print(f"RMS Humidity Error of {rmse_H:.4f} with standard deviation {std_H:.4f}")

    f = open(f'{save_dir}/errors.txt', "a")
    f.write(f"\n\n{filename}: RMS Temperature Error of {rmse_T:.4f} with standard deviation {std_T:.4f}")
    f.write(f"\n{filename}: RMS Humidity Error of {rmse_H:.4f} with standard deviation {std_H:.4f}")
    f.close()

    if not return_data:
        return rmse_T, rmse_H, std_T, std_H
    else:
        return predictions, actuals, errors
        

    

def plot_loss(train_loss, val_loss, filename):
    plt.semilogy(range(len(train_loss)), train_loss, color='black', label='Training')
    plt.semilogy(range(len(val_loss)), val_loss, color='grey', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.close()

def learn(model, model_name, model_path, dataset, num_epochs, learning_rate, weight_decay, save_dir, config_string):
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)

    if not os.path.isfile(model_path):
        train_loss, val_loss = train(model, dataset.train_loader, dataset.val_loader, criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler)
        print(f'Saving {model_name} state dict to {model_path}')
        torch.save(model.state_dict(), model_path)
        plot_loss(train_loss, val_loss, f'{save_dir}/loss_plot_{model_name}.png')
    else:
        print(f'Loading {model_name} state dict from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device.type), weights_only=True))

    # Inference
    return test(model, dataset, save_dir, f'{model_name}')


def transfer_learning_with_noise(model, model_name, model_path_source, model_path_target, dataset, num_epochs, learning_rate, weight_decay, save_dir, config_string, noise):
    if not os.path.isfile(model_path_target):
        # Load the pre-trained model
        model.load_state_dict(torch.load(model_path_source, map_location=torch.device(device.type), weights_only=True))
    
        # add noise
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()).to(device) * noise)
        
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Use a smaller learning rate
    
        # fine-tuning
        train_loss, val_loss = train(model, dataset.train_loader, dataset.val_loader, criterion, optimizer, num_epochs=num_epochs)
        torch.save(model.state_dict(), model_path_target)
        plot_loss(train_loss, val_loss, f'{save_dir}/{model_name}_loss_plot.png')
    else:
        print(f'Loading model state dict from {model_path_target}')
        model.load_state_dict(torch.load(model_path_target, map_location=torch.device(device.type), weights_only=True))
    
    return test(model, dataset, save_dir, model_name)
    

def standardize(variable, name: str, mean_values: dict, std_values: dict, std_only=False):
    if std_only:
        return variable  / std_values[name]
    else:
        return (variable - mean_values[name]) / std_values[name]

def destandardize(variable, name: str, mean_values: dict, std_values: dict, std_only=False):
    if std_only:
        return variable * std_values[name]
    else:
        return variable * std_values[name] + mean_values[name]