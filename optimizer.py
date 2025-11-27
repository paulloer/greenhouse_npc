import torch
import numpy as np
from datetime import datetime
from constants import STAND_VALUES

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("pgf")
mpl.rcParams.update(
{
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       10,         # control font sizes of different elements
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

class weights():
    def __init__(self, Q, C, R, alpha_window, alpha_wind):
        self.Q = Q
        self.C = C
        self.R = R
        self.alpha_window = alpha_window
        self.alpha_wind = alpha_wind


class bounds():
    def __init__(self, temp_target, temp_lower, temp_upper, humid_target, humid_lower, humid_upper, window_ub, wind_ub):
        self.temp_target = temp_target
        self.temp_lower = temp_lower
        self.temp_upper = temp_upper
        self.humid_target = humid_target
        self.humid_lower = humid_lower 
        self.humid_upper = humid_upper
        self.window_ub = window_ub
        self.wind_ub = wind_ub

def optimize_U(control_name, model, device, weights, bounds, X_m, U_m, P_m, U_opt, P_p, hours_p, N_p, save_dir, config_str, stride=20, num_iters = 5000, learning_rate=1e-3, print_loss=False, patience=200):

    Radiation_outside = destandardize(P_p[0][:, 3], 'Radiation_outside') 
    is_night = (Radiation_outside < 50).float()
    is_day = (Radiation_outside.cpu() >= 50).float()
    Wind_speed_outside = destandardize(P_p[0][:, 4], 'Wind_speed_outside')

    temp_targets = bounds.temp_target - 5 + is_day * 5
    temp_targets = standardize(temp_targets, "Temperature_inside")
    
    humid_targets = standardize(bounds.humid_target, "Humidity_inside")
    humid_targets = torch.full_like(temp_targets, humid_targets)
    
    x_target = torch.stack((temp_targets, humid_targets), dim=1).view(-1).to(device)


    # Soft Constraints
    x_min = torch.tensor([
        standardize(bounds.temp_lower, "Temperature_inside"),
        standardize(bounds.humid_lower, "Humidity_inside")
    ] * N_p).view(2 * N_p).to(device)
    x_max = torch.tensor([
        standardize(bounds.temp_upper, "Temperature_inside"),
        standardize(bounds.humid_upper, "Humidity_inside")
    ] * N_p).view(2 * N_p).to(device)
    
    # Hard Constraints
    u_min = torch.tensor([standardize(0, control_name)]).to(device)
    u_max = torch.tensor([standardize(100, control_name)]).to(device)

    window_ub = standardize(bounds.window_ub, control_name, std_only=True)  # applied to difference, no mean substraction
    window_ub_pred = standardize(bounds.window_ub*stride, control_name, std_only=True)  # applied to difference, no mean substraction
    wind_ub = bounds.wind_ub

    
    # Optimizer
    # optimizer = torch.optim.Adam([U_opt], lr=learning_rate)
    optimizer = torch.optim.SGD([U_opt], lr=learning_rate)
    # optimizer = torch.optim.LBFGS([U_opt], lr=1e-2, max_iter=20)


    
    best_loss = float('inf')
    no_improvement = 0
    best_U = U_opt.clone()
    opt_loss = []
    
    for i in range(num_iters):
        optimizer.zero_grad()
        
        X_opt = model(X_m, U_m, P_m, U_opt.unsqueeze(0), P_p)
        X_opt = X_opt.squeeze()
        
        loss = 0
        
        # tracking
        loss += (weights.Q[0] * torch.sum((X_opt[:,0] - x_target[0]) ** 2) + 
                 weights.Q[1] * torch.sum((X_opt[:,1] - x_target[1]) ** 2))

        # soft constraints
        loss += (weights.C[0] * torch.sum(torch.relu(X_opt[:,0] - x_max[0]) ** 2 + torch.relu(x_min[0] - X_opt[:,0]) ** 2) + 
                 weights.C[1] * torch.sum(torch.relu(X_opt[:,1] - x_max[1]) ** 2 + torch.relu(x_min[1] - X_opt[:,1]) ** 2))
        loss = loss.unsqueeze(0)
        
        # smoothing
        loss += weights.R * (U_m[0][-1] - U_opt[0])**2 / stride  # change in 30 seconds
        loss += weights.R * torch.sum((U_opt[:-1] - U_opt[1:]) ** 2)  # change in 10 minutes

        # # discrete control values
        # allowed_values  = np.linspace(0.0, 100.0, 11).tolist() 
        # allowed_values = (allowed_values - dataset.mean_values[control_name]) / dataset.std_values[control_name]
        # for j in range(len(U_opt)):  # Indices of the variables to constrain
        #     smoothing_loss += R * torch.prod(torch.abs(torch.tensor([U_opt[j] - v for v in allowed_values])))

        # hard constraints
        # windows between 0 and 100 %
        loss += weights.alpha_window * torch.sum(torch.relu(U_opt - u_max) ** 2 + torch.relu(u_min - U_opt) ** 2)
        # maximum opening rate
        # print(f'last control {destandardize(U_m[0][-1], control_name)}, next opt control {destandardize(U_opt[0], control_name)}')
        loss += weights.alpha_window * torch.sum(torch.relu(torch.abs(U_m[0][-1] - U_opt[0]) - window_ub) ** 2)  # change in 30 seconds
        loss += weights.alpha_window * torch.sum(torch.relu(torch.abs(U_opt[:-1] - U_opt[1:]) - window_ub_pred) ** 2)  # change in stride minutes, e.g. 5 or 10 min
        # windows closed at night
        # loss += weights.alpha_window * torch.sum(torch.mul(is_night, torch.relu(destandardize(U_opt.squeeze(), control_name)/100)))
        # close windows when windy
        loss += weights.alpha_wind * torch.sum(torch.mul(torch.relu(Wind_speed_outside - wind_ub) ** 2, torch.relu(destandardize(U_opt.squeeze(), control_name)/100)))
        
        if int(loss.item()) < best_loss:
            best_loss = int(loss.item())
            with torch.no_grad():
                best_U.copy_(U_opt)
            no_improvement = 0
        else:
            no_improvement += 1

        
        if no_improvement >= patience:
            #print(f"Early stopping due to no improvement in loss. Best Loss in iteration {i-patience}: {best_loss:.0f}")
            U_opt = best_U
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([U_opt], max_norm=10.0)
        # U_opt.data = torch.clamp(U_opt, min=u_min, max=u_max)
        optimizer.step()

        opt_loss.append(loss.item())
        if i % 100 == 0 and print_loss:
            print(f"Iteration {i}, Loss: {loss.item():.0f}")


    if print_loss:
        plt.semilogy(opt_loss)
        plt.ylabel('Cost / Loss')
        plt.xlabel('Iterations')
        plt.savefig(f"{save_dir}/optimization_loss_{num_iters}iters_{config_str}.png", dpi=300)
        plt.close()

    # Final optimized control sequence
    optimal_X_stand = X_opt.detach().cpu().numpy()
    optimal_U_stand = U_opt.detach().cpu().numpy()

    return optimal_X_stand, optimal_U_stand, i-patience, best_loss


def split_with_transition_points(x, y, lower, upper):
    x_in, y_in = [], []
    x_out, y_out = [], []

    def is_in_bounds(v):
        return lower < v < upper

    i = 0
    while i < len(x) - 1:
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        in0, in1 = is_in_bounds(y0), is_in_bounds(y1)

        if in0 and in1:
            x_in.append(x0)
            y_in.append(y0)
        elif not in0 and not in1:
            x_out.append(x0)
            y_out.append(y0)
        else:
            # Interpolate crossing
            if y1 != y0:
                if y1 > upper or y0 > upper:
                    bound = upper
                else:
                    bound = lower

                alpha = (bound - y0) / (y1 - y0)
                x_cross = x0 + alpha * (x1 - x0)
                y_cross = bound

                if in0:
                    # from in to out: add x0, y0 to in; x_cross to both, but omit from next segment
                    x_in.append(x0)
                    y_in.append(y0)
                    x_in.append(x_cross)
                    y_in.append(y_cross)

                    x_out.append(np.nan)
                    y_out.append(np.nan)
                    x_out.append(x_cross)
                    y_out.append(y_cross)
                else:
                    # from out to in: add x0, y0 to out; x_cross to both, but omit from next segment
                    x_out.append(x0)
                    y_out.append(y0)
                    x_out.append(x_cross)
                    y_out.append(y_cross)

                    x_in.append(np.nan)
                    y_in.append(np.nan)
                    x_in.append(x_cross)
                    y_in.append(y_cross)
        i += 1

    # Append final point
    if is_in_bounds(y[-1]):
        x_in.append(x[-1])
        y_in.append(y[-1])
    else:
        x_out.append(x[-1])
        y_out.append(y[-1])

    return (
        np.array(x_in), np.array(y_in),
        np.array(x_out), np.array(y_out)
    )




def plot_optimization(optimal_X, optimal_U, bounds, parameter, hours_p, N_p, save_dir, filename):
    colors = ['purple', 'blue', 'limegreen', 'limegreen', 'magenta', 'cyan', 'gold', 'darkorange', 'darkcyan']
    linewidth=1

    Radiation_outside = parameter[:,3]
    is_day = Radiation_outside >= 50
    temp_targets = bounds.temp_target - 5 + is_day * 5

    # Calculate y1 and y2 based on the temperature bounds
    #y1 = np.where((optimal_X[:,0] >= bounds.temp_upper) | (optimal_X[:,0] <= bounds.temp_lower), optimal_X[:,0], np.nan)
    #y2 = np.where((optimal_X[:,0] < bounds.temp_upper) & (optimal_X[:,0] > bounds.temp_lower), optimal_X[:,0], np.nan)

    # Calculate y3 and y4 based on the humidity bounds
    #y3 = np.where((optimal_X[:,1] >= bounds.humid_upper) | (optimal_X[:,1] <= bounds.humid_lower), optimal_X[:,1], np.nan)
    #y4 = np.where((optimal_X[:,1] < bounds.humid_upper) & (optimal_X[:,1] > bounds.humid_lower), optimal_X[:,1], np.nan)

    fig, axs = plt.subplots(5, 1, figsize=(15/2.54, 20/2.54), sharex=True)

    x = np.linspace(0, N_p-1, N_p, dtype=int)
    
    x_in, y_in, x_out, y_out = split_with_transition_points(x, optimal_X[:,0], bounds.temp_lower, bounds.temp_upper)
    
    axs[0].fill_between(x, bounds.temp_lower, bounds.temp_upper, color=colors[0], alpha=0.2, label='$\\mathcal{T}$')
    axs[0].plot(x_in, y_in, color='purple', label='$\\theta^*_{\\mathrm{in}}\\in\\mathcal{T}$', linewidth=1)
    axs[0].plot(x_out, y_out, color='red', label='$\\theta^*_{\\mathrm{in}}\\notin\\mathcal{T}$', linewidth=1)
    #axs[0].fill_between(x, bounds.temp_lower, bounds.temp_upper, color=colors[0], alpha=0.2, label='$\\mathcal{T}$')
    #axs[0].plot(x, y1, color='red', label='$\\theta^*_{\\mathrm{in}}\\notin\\mathcal{T}$', linewidth=linewidth)
    #axs[0].plot(x, y2, color=colors[0], label='$\\theta^*_{\\mathrm{in}}\\in\\mathcal{T}$', linewidth=linewidth)
    axs[0].plot(x, temp_targets, '--', color=colors[0], label="$\\theta_{\\mathrm{ref}}$")
    axs[0].plot(x, parameter[:,0], color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
    axs[0].set_ylabel('$\\theta$ [$^\circ$C]')

    x_in, y_in, x_out, y_out = split_with_transition_points(x, optimal_X[:,1], bounds.humid_lower, bounds.humid_upper)
    
    axs[1].fill_between(x, bounds.humid_lower, bounds.humid_upper, color=colors[1], alpha=0.2, label='$\\mathcal{T}$')
    axs[1].plot(x_in, y_in, color=colors[1], label='$\\theta^*_{\\mathrm{in}}\\in\\mathcal{T}$', linewidth=1)
    axs[1].plot(x_out, y_out, color='red', label='$\\theta^*_{\\mathrm{in}}\\notin\\mathcal{T}$', linewidth=1)
    #axs[1].fill_between(x, bounds.humid_lower, bounds.humid_upper, color=colors[1], alpha=0.2, label='$\\mathcal{H}$')
    #axs[1].plot(x, y3, color='red', label='$\\phi^*_{\\mathrm{in}}\\notin\\mathcal{H}$', linewidth=linewidth)
    #axs[1].plot(x, y4, color=colors[1], label='$\\phi^*_{\\mathrm{in}}\\in\\mathcal{H}$', linewidth=linewidth)
    #axs[1].plot(x, np.zeros(N_p) + bounds.humid_target, '--', color=colors[1], label="$\\phi_{\\mathrm{ref}}$")
    axs[1].plot(x, parameter[:,1], color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
    axs[1].set_ylabel('$\\phi$ [\\%]')

    axs[2].plot(x, parameter[:,2], color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
    axs[2].plot(x, parameter[:,3], color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
    axs[2].set_ylabel('$R$ [W/m$^2$]')

    axs[3].plot(x, optimal_U, color=colors[2], label='${u^*}$', linewidth=linewidth)
    axs[3].set_ylabel('$u$ [\\%]')
    axs[3].set_yticks([0,20,40,60,80,100])
    axs[3].set_yticklabels([0,20,40,60,80,100])

    axs[4].plot(x, parameter[:,4], color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
    axs[4].plot(x, np.zeros(N_p) + bounds.wind_ub, '--', color=colors[8], label="$v_{\\mathrm{max}}$")
    axs[4].set_ylabel('$v$ [m/s]')

    for i in range(5):
        axs[i].grid()
        axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
        axs[i].set_xlim(x.min(), x.max())

    if hours_p <= 12:
        axs[-1].set_xlabel("Time [h]")
        axs[-1].set_xticks(np.linspace(0, N_p, hours_p+1, dtype=int))
        axs[-1].set_xticklabels(np.linspace(0, hours_p, hours_p+1, dtype=int))
    else:
        axs[-1].set_xlabel("Time [h]")
        axs[-1].set_xticks(np.linspace(0, N_p, int(hours_p/12)+1, dtype=int))
        axs[-1].set_xticklabels(np.linspace(0, int(hours_p/12), int(hours_p/12)+1, dtype=int)*12)
    fig.align_ylabels()
    plt.tight_layout()

    # plt.subplots_adjust(right=0.8)  # Make space on the right for the legends
    current_time = datetime.now()
    plt.savefig(f"{save_dir}/optimization_results/{filename}_{current_time.strftime('%Y-%m-%d %H:%M:%S')}.pgf")
    plt.savefig(f"{save_dir}/optimization_results/{filename}_{current_time.strftime('%Y-%m-%d %H:%M:%S')}.png", dpi=300)
    plt.close()




    # mpl.rcParams.update(
    # {
    #     "font.size":       6,         # control font sizes of different elements
    #     "axes.labelsize":  6,
    #     "legend.fontsize": 6,
    #     "xtick.labelsize": 6,
    #     "ytick.labelsize": 6,
    # })

    # # Plot each feature in its own subplot
    # fig, axs = plt.subplots(5, 1, figsize=(7/2.54, 7/2.54), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})
    # linewidth=.8

    # x_in, y_in, x_out, y_out = split_with_transition_points(x, optimal_X[:,0], bounds.temp_lower, bounds.temp_upper)
    
    # axs[0].fill_between(x, bounds.temp_lower, bounds.temp_upper, color=colors[0], alpha=0.2, label='$\\mathcal{T}$')
    # axs[0].plot(x_in, y_in, color='purple', label='$\\theta^*_{\\mathrm{in}}\\in\\mathcal{T}$', linewidth=1)
    # axs[0].plot(x_out, y_out, color='red', label='$\\theta^*_{\\mathrm{in}}\\notin\\mathcal{T}$', linewidth=1)
    # axs[0].plot(x, temp_targets, '--', color=colors[0], label="$\\theta_{\\mathrm{ref}}$")
    # axs[0].plot(x, parameter[:,0], color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
    # axs[0].set_ylabel('$\\theta$ [$^\circ$C]')

    # x_in, y_in, x_out, y_out = split_with_transition_points(x, optimal_X[:,1], bounds.humid_lower, bounds.humid_upper)
    
    # axs[1].fill_between(x, bounds.humid_lower, bounds.humid_upper, color=colors[1], alpha=0.2, label='$\\mathcal{T}$')
    # axs[1].plot(x_in, y_in, color=colors[1], label='$\\theta^*_{\\mathrm{in}}\\in\\mathcal{T}$', linewidth=1)
    # axs[1].plot(x_out, y_out, color='red', label='$\\theta^*_{\\mathrm{in}}\\notin\\mathcal{T}$', linewidth=1)
    # #axs[1].plot(x, np.zeros(N_p) + bounds.humid_target, '--', color=colors[1], label="$\\phi_{\\mathrm{ref}}$")
    # axs[1].plot(x, parameter[:,1], color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
    # axs[1].set_ylabel('$\\phi$ [\\%]')

    # axs[2].plot(x, parameter[:,2], color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
    # axs[2].plot(x, parameter[:,3], color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
    # axs[2].set_ylabel('$R$ [W/m$^2$]')

    # axs[3].plot(x, optimal_U, color=colors[2], label='${u^*}$', linewidth=linewidth)
    # axs[3].set_ylabel('$u$ [\\%]')
    # axs[3].set_yticks([0,25,50,75,100])
    # axs[3].set_yticklabels([0,25,50,75,100])

    # axs[4].plot(x, parameter[:,4], color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
    # axs[4].plot(x, np.zeros(N_p) + bounds.wind_ub, '--', color=colors[8], label="$v_{\\mathrm{max}}$")
    # axs[4].set_ylabel('$v$ [m/s]')

    # # fig.tight_layout(rect=[0, 0, 0.85, 1]) 
    # plt.subplots_adjust(bottom=0.12, top=1, left=0.17, right=0.8)

    # for i in range(5):
    #     axs[i].grid()
    #     # axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    #     axs[i].set_xlim(x.min(), x.max())
        
    # if hours_p <= 12:
    #     axs[-1].set_xlabel("Time [h]")
    #     axs[-1].set_xticks(np.linspace(0, N_p, hours_p+1, dtype=int))
    #     axs[-1].set_xticklabels(np.linspace(0, hours_p, hours_p+1, dtype=int))
    # else:
    #     axs[-1].set_xlabel("Time [h]")
    #     axs[-1].set_xticks(np.linspace(0, N_p, int(hours_p/12)+1, dtype=int))
    #     axs[-1].set_xticklabels(np.linspace(0, int(hours_p/12), int(hours_p/12)+1, dtype=int)*12)
    
    # handles, labels = [], []
    # for ax in axs:
    #     for handle, label in zip(*ax.get_legend_handles_labels()):
    #         handles.append(handle)
    #         labels.append(label)
    # fig.legend(
    #     handles, labels, loc='center left', bbox_to_anchor=(.8, .56), ncol=1, frameon=False, labelspacing=0.35, handlelength=.5
    # )

    # fig.align_ylabels()

    # # plt.subplots_adjust(right=0.8)  # Make space on the right for the legends
    # plt.savefig(f"{save_dir}/{filename}_presentation.pgf")
    # plt.savefig(f"{save_dir}/{filename}_presentation.png", dpi=300)
    # plt.close()

    # mpl.rcParams.update(
    # {
    #     "font.size":       10,         # control font sizes of different elements
    #     "axes.labelsize":  10,
    #     "legend.fontsize": 9,
    #     "xtick.labelsize": 9,
    #     "ytick.labelsize": 9,
    # })

# TODO: make the normalization dynamic as they would be updated with a new model after online learning
def standardize(variable, name: str, std_only=False):
    if std_only:
        return variable  / STAND_VALUES.std_values[name]
    else:
        return (variable - STAND_VALUES.mean_values[name]) / STAND_VALUES.std_values[name]

def destandardize(variable, name: str, std_only=False):
    if std_only:
        return variable * STAND_VALUES.std_values[name]
    else:
        return variable * STAND_VALUES.std_values[name] + STAND_VALUES.mean_values[name]

    

