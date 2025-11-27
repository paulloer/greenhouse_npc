import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
import copy
from sklearn.metrics.pairwise import cosine_similarity

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

# filename = 'GH_Data_2025-11-19'
filename = 'GH_Data'
log_hours = 20


# Load the dataset
# data = pd.read_csv("GH_Ventilation_Tests_2024_11_18-2024_11_24.csv", delimiter=' ')
data = pd.read_csv(filename + ".csv", delimiter=',')
# data = pd.read_csv(filename + ".csv", delimiter=' ')
data = data.rename(columns={'Temperature_interior_S1': 'Temperature_inside', 'Humidity_interior_S1': 'Humidity_inside', 
                         'Temperature_exterior': 'Temperature_outside', 'Humidity_exterior': 'Humidity_outside', 'Rad_Global_interior_North': 'Radiation_inside', 'Rad_Global_exterior': 'Radiation_outside', 'Wind_Speed_exterior': 'Wind_speed_outside'})
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d %H:%M:%S")
data.set_index('Date', inplace=True)

colors = ['purple', 'blue', 'turquoise', 'darkgreen', 'magenta', 'cyan', 'gold', 'darkorange', 'darkcyan']
linewidth=1

control_features = [
            'Vent_S1_Roof_1', 'Vent_S1_Roof_2', 'Vent_S1_Roof_3', 
            'Vent_S1_Side_N', 'Vent_S1_Side_NW', 'Vent_S1_Side_S', 'Vent_S1_Side_SW', 
            'Vent_S2_Roof_1', 'Vent_S2_Roof_2', 'Vent_S2_Roof_3',
            'Vent_S2_Side_E', 'Vent_S2_Side_N', 'Vent_S2_Side_S'
]
palette = sns.color_palette('magma', n_colors=len(control_features))


# Plot each feature in its own subplot
step = 20

# fig, axs = plt.subplots(5, 1, figsize=(15/2.54, 20/2.54), sharex=True)

# axs[0].plot(data['Temperature_inside'].to_numpy()[::step], color=colors[0], label='$\\theta_{\\mathrm{in}}$', linewidth=linewidth)
# axs[0].plot(data['Temperature_outside'].to_numpy()[::step], color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
# axs[0].set_ylabel('$\\theta$ [$^\circ$C]')

# axs[1].plot(data['Humidity_inside'].to_numpy()[::step], color=colors[1], label='$\\phi_{\\mathrm{in}}$', linewidth=linewidth)
# axs[1].plot(data['Humidity_outside'].to_numpy()[::step], color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
# axs[1].set_ylabel('$\\phi$ [\\%]')

# axs[2].plot(data['Radiation_inside'].to_numpy()[::step], color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
# axs[2].plot(data['Radiation_outside'].to_numpy()[::step], color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
# axs[2].set_ylabel('$R$ [W/m$^2$]')

# axs[3].plot(data['Vent_S1_Roof_1'].to_numpy()[::step], color='limegreen', linewidth=linewidth, label='$u_{1}$')
# axs[3].set_ylabel('$u$ [\\%]')

# axs[4].plot(data['Wind_speed_outside'].to_numpy()[::step], color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
# axs[4].set_ylabel('$v$ [m/s]')

# for i in range(5):
#     axs[i].grid()
#     axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, markerscale=0.5, labelspacing=0.35, handlelength=.5)
#     axs[i].set_xlim(0, 7*(120/step)*24*7)
    
# axs[-1].set_xticks(np.arange(8) *(120/step)*24*7)
# axs[-1].set_xticklabels(np.arange(8))
# axs[-1].set_xlabel('Time [weeks]')

# fig.align_ylabels()
# plt.tight_layout()

# # Save plots
# plt.savefig(filename + '.pgf')
# plt.savefig(filename + '.png', dpi=300)
# plt.close()


# mpl.rcParams.update(
# {
#     "font.size":       6,         # control font sizes of different elements
#     "axes.labelsize":  6,
#     "legend.fontsize": 6,
#     "xtick.labelsize": 6,
#     "ytick.labelsize": 6,
# })

# fig, axs = plt.subplots(5, 1, figsize=(7/2.54, 7/2.54), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})
# linewidth=.8

# axs[0].plot(data['Temperature_inside'].to_numpy()[::step], color=colors[0], label='$\\theta_{\\mathrm{in}}$', linewidth=linewidth)
# axs[0].plot(data['Temperature_outside'].to_numpy()[::step], color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
# axs[0].set_ylabel('$\\theta$ [$^\circ$C]')

# axs[1].plot(data['Humidity_inside'].to_numpy()[::step], color=colors[1], label='$\\phi_{\\mathrm{in}}$', linewidth=linewidth)
# axs[1].plot(data['Humidity_outside'].to_numpy()[::step], color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
# axs[1].set_ylabel('$\\phi$ [\\%]')

# axs[2].plot(data['Radiation_inside'].to_numpy()[::step], color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
# axs[2].plot(data['Radiation_outside'].to_numpy()[::step], color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
# axs[2].set_ylabel('$R$ [W/m$^2$]')

# axs[3].plot(data['Vent_S1_Roof_1'].to_numpy()[::step], color='limegreen', linewidth=linewidth, label='$u_{1}$')
# axs[3].set_ylabel('$u$ [\\%]')

# axs[4].plot(data['Wind_speed_outside'].to_numpy()[::step], color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
# axs[4].set_ylabel('$v$ [m/s]')

# for i in range(5):
#     axs[i].grid()
#     axs[i].set_xlim(0, 7*(120/step)*24*7)

# handles, labels = [], []
# for ax in axs:
#     for handle, label in zip(*ax.get_legend_handles_labels()):
#         handles.append(handle)
#         labels.append(label)
# fig.legend(
#     handles, labels, loc='center left', bbox_to_anchor=(.8, .56), ncol=1, frameon=False, labelspacing=0.35, handlelength=.5
# )

# axs[-1].set_xticks(np.arange(8) *(120/step)*24*7)
# axs[-1].set_xticklabels(np.arange(8))
# axs[-1].set_xlabel('Time [weeks]')

# fig.align_ylabels()
# plt.subplots_adjust(bottom=0.12, top=1, left=0.17, right=0.8)

# # Save plots
# plt.savefig(filename + '_presentation.pgf')
# plt.savefig(filename + '_presentation.png', dpi=300)
# plt.close()


mpl.rcParams.update(
{
    "font.size":       8,         # control font sizes of different elements
    "axes.labelsize":  8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, axs = plt.subplots(5, 1, figsize=(7/2.54, 7/2.54), sharex=True, gridspec_kw = {'hspace':0.15, 'wspace':0})
linewidth=.8

# axs[0].plot(data['Temperature_inside'].to_numpy()[::step], color=colors[0], label='$\\theta_{\\mathrm{in}}$', linewidth=linewidth)
axs[0].plot(data['Temperature_inside'].to_numpy(), color=colors[0], label='$\\theta_{\\mathrm{in}}$', linewidth=linewidth)
axs[0].plot(data['Temperature_outside'].to_numpy(), color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
axs[0].set_ylabel('$\\theta$ [$^\circ$C]')
axs[0].set_yticks([15,20,25,30])
axs[0].set_yticklabels([15,20,25,30])

# setpoint
x_ref = list(np.linspace(0, log_hours*120, log_hours*120+1))
y_ref = [15] * int(((8.2*120+1))) + [20] * int(((3.1*120+1)))
y_ref += [25] * ((len(x_ref)-len(y_ref)))
# y_ref = [20] * (log_hours*120+1)
axs[0].plot(x_ref, y_ref, color='purple', label='$\\theta_{\\mathrm{ref}}$', linewidth=linewidth, linestyle='dashed')

axs[1].plot(data['Humidity_inside'].to_numpy(), color=colors[1], label='$\\phi_{\\mathrm{in}}$', linewidth=linewidth)
axs[1].plot(data['Humidity_outside'].to_numpy(), color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
axs[1].set_ylabel('$\\phi$ [\\%]')

axs[2].plot(data['Radiation_inside'].to_numpy(), color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
axs[2].plot(data['Radiation_outside'].to_numpy(), color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
axs[2].set_ylabel('$R$ [W/m$^2$]')

axs[3].plot(data['Vent_S1_Side_NW'].to_numpy(), color='limegreen', linewidth=linewidth, label='$u_{1}$')
axs[3].set_ylabel('$u$ [\\%]')
axs[3].set_yticks([0,50,100])
axs[3].set_yticklabels([0,50,100])
axs[3].set_ylim(-1,101)

axs[4].plot(data['Wind_speed_outside'].to_numpy(), color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
axs[4].set_ylabel('$v$ [km/h]')

# v_wind_max
y_v_wind_max = [30] * (log_hours*120+1)
axs[4].plot(x_ref, y_v_wind_max, color=colors[8], label='$v_{\\mathrm{max}}$', linewidth=linewidth, linestyle='dashed')

for i in range(5):
    axs[i].grid()
    axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, markerscale=0.5, labelspacing=0.35, handlelength=.5)
    axs[i].set_xlim(0, (log_hours-1)*120)

if log_hours>=24:
    axs[-1].set_xticks(np.arange(int(log_hours/24)+1)*120*24)
    axs[-1].set_xticklabels(np.arange(int(log_hours/24)+1))
    axs[-1].set_xlabel('Time [days]')
else:
    axs[-1].set_xticks(np.arange(log_hours)*120)
    axs[-1].set_xticklabels(np.arange(log_hours))
    axs[-1].set_xlabel('Time [hours]')

fig.align_ylabels()
plt.subplots_adjust(bottom=0.125, top=1, left=0.19, right=0.8)


# Save plots
plt.savefig(filename + '_presentation.pgf')
plt.savefig(filename + '_presentation.png', dpi=500)
plt.close()



# filename_forecast = 'GH_Dataset_2024-12-06_2025-01-22_forecast'
# data_forecast = pd.read_csv(filename_forecast + ".csv")
# data_forecast = data_forecast.iloc[:int(3/48 * len(data))].copy()
# print(len(data_forecast))

# # Plot forecast
# fig, axs = plt.subplots(5, 1, figsize=(15/2.54, 20/2.54), sharex=True)

# axs[0].plot(data_forecast.index, data_forecast['Temperature_inside'], color=colors[0], label='$\\theta_{\\mathrm{in}}$', linewidth=linewidth)
# axs[0].plot(data_forecast.index, data_forecast['Temperature_outside_forecast'], color=colors[4], label='$\\theta_{\\mathrm{out}}$', linewidth=linewidth)
# axs[0].set_ylabel('$\\theta$ [$^\circ$C]')

# axs[1].plot(data_forecast.index, data_forecast['Humidity_inside'], color=colors[1], label='$\\phi_{\\mathrm{in}}$', linewidth=linewidth)
# axs[1].plot(data_forecast.index, data_forecast['Humidity_outside_forecast'], color=colors[5], label='$\\phi_{\\mathrm{out}}$', linewidth=linewidth)
# axs[1].set_ylabel('$\\phi$ [\\%]')

# axs[2].plot(data_forecast.index, data_forecast['Radiation_inside_forecast'], color=colors[6], label='$R_{\\mathrm{in}}$', linewidth=linewidth)
# axs[2].plot(data_forecast.index, data_forecast['Radiation_outside_forecast'], color=colors[7], label='$R_{\\mathrm{out}}$', linewidth=linewidth)
# axs[2].set_ylabel('$R$ [W/m$^2$]')

# #for i in range(len(control_features)):
# #    axs[3].plot(data_forecast.index, data_forecast[control_features[i]], color=palette[i], alpha=0.5, linewidth=linewidth)  # label=control_features[i], 
# axs[3].plot(data_forecast.index, data_forecast['Vent_S1_Roof_1'], color='limegreen', linewidth=linewidth, label='$u_{1}$')
# axs[3].set_ylabel('$u$ [\\%]')

# axs[4].plot(data_forecast.index, data_forecast['Wind_speed_outside_forecast'], color=colors[8], label='$v_{\\mathrm{wind}}$', linewidth=linewidth)
# axs[4].set_ylabel('$v$ [m/s]')

# for i in range(5):
#     axs[i].grid()
#     axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, markerscale=0.5)
#     axs[i].set_xlim(data_forecast.index.min(), data_forecast.index.max())
    

# # Set x-axis label for the last subplot
# axs[-1].set_xlabel("Time")

# axs[-1].set_xticks(np.arange(4) *2*60*24)
# axs[-1].set_xticklabels(np.arange(4))
# axs[-1].set_xlabel('Time [days]')

# fig.align_ylabels()
# plt.tight_layout()

# plt.savefig(filename_forecast + '.pgf')
# plt.savefig(filename_forecast + '.png', dpi=300)
# plt.close()





# Plot each feature in its own subplot
fig, axs = plt.subplots(len(control_features), 1, figsize=(15/2.54, 40/2.54), sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})

for i in range(len(control_features)):
    axs[i].plot(data.index, data[control_features[i]], color='blue', linewidth=linewidth/2, label=control_features[i])  
    axs[i].set_ylabel('$u$ [\\%]')
    axs[i].grid()
    axs[i].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, markerscale=0.5)
    axs[i].set_xlim(data.index.min(), data.index.max())
    

# Set x-axis label for the last subplot
axs[-1].set_xlabel("Time")

# Format x-axis ticks to show days
date_formatter = DateFormatter("%m/%d/%Y")
axs[-1].xaxis.set_major_formatter(date_formatter)
plt.tight_layout()

plt.savefig(filename + '_controls.png', dpi=300)
plt.close()


sum = 0
for i in range(len(data)):
    if data['Vent_S1_Roof_1'][i] == data['Vent_S1_Roof_2'][i]:
        sum += 1
print(f'N samples is {len(data)}')
print(f'Similarity between Vent_S1_Roof_1 and Vent_S1_Roof_2 is {sum/len(data)}')

# Select only the relevant columns from the dataframe
data_subset = data[control_features]

# Compute similarity based on identical values
num_rows = len(data_subset)
similarity_matrix = np.zeros((len(control_features), len(control_features)))

for i, col1 in enumerate(control_features):
    for j, col2 in enumerate(control_features):
        similarity_matrix[i, j] = np.sum(data_subset[col1] == data_subset[col2]) / num_rows

# Convert to DataFrame for better visualization
conf_matrix = pd.DataFrame(similarity_matrix, columns=control_features, index=control_features)

# Mask the upper triangle
#mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

# Plot the lower triangle similarity matrix
plt.figure(figsize=(15/2.54, 0.8*15/2.54))
#sns.heatmap(conf_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
u_labels = ['$u_{\\mathrm{1}}$', '$u_{\\mathrm{2}}$', '$u_{\\mathrm{3}}$', '$u_{\\mathrm{4}}$', '$u_{\\mathrm{5}}$', '$u_{\\mathrm{6}}$', '$u_{\\mathrm{7}}$', '$u_{\\mathrm{8}}$', '$u_{\\mathrm{9}}$', '$u_{\\mathrm{10}}$', '$u_{\\mathrm{11}}$', '$u_{\\mathrm{12}}$', '$u_{\\mathrm{13}}$']
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, xticklabels=u_labels, yticklabels=u_labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
#plt.title("Similarity Matrix for " + filename)
plt.tight_layout()
plt.savefig(filename + '_controls_similarity.pgf')
plt.savefig(filename + '_controls_similarity.png', dpi=300)
