from dataclasses import dataclass
import os

TIME_FORECAST = 3600
TIME_SENSOR = 30
TIME_CONTROL = 30
TIME_LEARNING = 24*3600

CSV_PATH = "GH_Data.csv"

# Hyperparameters for time
hours_m = 1
hours_p = 12
stride = 10
N_m = hours_m * int(120/stride)  # Past timesteps
N_p = hours_p * int(120/stride)  # Future timesteps

# Hyperparameters for transformer
learning_rate = 1e-3
hidden_size = 128
attention_head_size = 8
num_epochs = 200
batch_size = 16
weight_decay = 1e-3
num_lstm_layers = 1
dropout_prob = 0.5
config_string = f"{N_m}_Nm_{N_p}_Np_lr_{learning_rate}_hs_{hidden_size}_ahs_{attention_head_size}_nll_{num_lstm_layers}_bs_{batch_size}_do_{dropout_prob}_e_{num_epochs}_stride_{stride}"
# save_dir_transformer = os.getcwd() + f"/one_shot_train_test_opt/simple_tft_{config_string}"
# save_dir_transformer = os.getcwd() + f"/one_shot_train_test_opt/transformer_no_tanh_with_past_with_norm_{config_string}"
# save_dir_transformer = os.getcwd() + f"/one_shot_train_test_opt/transformer_no_tanh_with_norm_{config_string}"
save_dir = os.getcwd() + f"/transformer_no_tanh_with_past_{config_string}"
# model_name = 'model_W20_0.001lr_100e'
# model_name = 'model_W24_0.001lr_100e'
# model_name = 'model_A25_0.001lr_200e'
model_name = 'model_transfer_A25_to_J25_0.01n_0.001lr_200e'
# model_name = 'model_transfer_W24_to_A25_0.01n_0.0001lr_200e'
# model_name = 'model_transfer_W20_to_W24_0.01n_0.0001lr_100e'
model_path = f"{save_dir}/{model_name}.pth"

@dataclass
class arguments:
    hours_m: int
    hours_p: int
    stride: int
    N_m: int
    N_p: int

    hidden_size: int
    attention_head_size: int
    num_lstm_layers: int
    num_epochs: int
    learning_rate: float
    batch_size: int
    weight_decay: float
    dropout_prob: float

    config_string: str
    save_dir: str
    model_name: str
    model_path: str

ARGUMENTS = arguments(hours_m, hours_p, stride, N_m, N_p, hidden_size, attention_head_size, num_lstm_layers, num_epochs, learning_rate, batch_size, weight_decay, dropout_prob, config_string, save_dir, model_name, model_path)

@dataclass
class Data:
    mean_values: dict
    std_values: dict

# # from dataset W20
# mean_values_W20 = {'Temperature_inside': 18.501110883619486, 'Humidity_inside': 76.43854710715095, 'Lateral_vents_opening': 11.987736790912713, 'Temperature_outside': 16.002862260363955, 'Humidity_outside': 64.00996561981053, 'Radiation_inside': 60.640828274829566, 'Radiation_outside': 137.9653291051628, 'Wind_speed_outside': 1.8502185035806773}
# std_values_W20 = {'Temperature_inside': 5.509722365652311, 'Humidity_inside': 14.697952052068693, 'Lateral_vents_opening': 22.820823627983984, 'Temperature_outside': 4.03241253426733, 'Humidity_outside': 15.775418272704076, 'Radiation_inside': 98.35048940385843, 'Radiation_outside': 213.96149089707887, 'Wind_speed_outside': 1.8237571525213807}

# # from dataset W24
# mean_values = {'Temperature_inside': 13.84089299932915, 
#                 'Humidity_inside': 88.91561369492504, 
#                 'Vent_S1_Roof_1': 39.221529600223704, 
#                 'Temperature_outside': 12.491215005419411, 
#                 'Humidity_outside': 62.771677219004545, 
#                 'Radiation_inside': 43.02354232728346, 
#                 'Radiation_outside': 125.09364397092604, 
#                 'Wind_speed_outside': 9.326526949516806}

# std_values = {'Temperature_inside': 4.135878232629625, 
#               'Humidity_inside': 11.783589741745608, 
#               'Vent_S1_Roof_1': 39.65604298195352, 
#               'Temperature_outside': 2.794039901247935, 
#               'Humidity_outside': 19.91871414941846, 
#               'Radiation_inside': 68.51759472364624, 
#               'Radiation_outside': 194.87237692608437, 
#               'Wind_speed_outside': 6.047619451448265, 
#               'Temperature_outside_forecast': 2.697844910020797}

# # from dataset A25
# mean_values = {'Temperature_inside': 20.77890175915441, 'Humidity_inside': 76.02947464263691, 'Vent_S1_Roof_1': 39.635112266259895, 'Temperature_outside': 17.463111099595366, 'Humidity_outside': 61.58656791851881, 'Radiation_inside': 106.34903179743938, 'Radiation_outside': 246.45274354364588, 'Wind_speed_outside': 12.784226441158179}
# std_values = {'Temperature_inside': 5.360120242946607, 'Humidity_inside': 23.07174134809954, 'Vent_S1_Roof_1': 35.314052207280454, 'Temperature_outside': 3.1888020925133684, 'Humidity_outside': 28.64580495960317, 'Radiation_inside': 168.7620969874042, 'Radiation_outside': 325.87278281983055, 'Wind_speed_outside': 10.359057137208897}

# from dataset J25
mean_values = {'Temperature_inside': 32.342568164197104, 'Humidity_inside': 63.79411825847823, 'Vent_S1_Roof_1': 80.02092027778434, 'Temperature_outside': 25.25148889461441, 'Humidity_outside': 81.95966951572673, 'Radiation_inside': 86.32341380944176, 'Radiation_outside': 281.81056163657706, 'Wind_speed_outside': 8.022132665739985}
std_values = {'Temperature_inside': 7.4359448074264725, 'Humidity_inside': 23.059840231170497, 'Vent_S1_Roof_1': 27.91434034616817, 'Temperature_outside': 2.1216390419033173, 'Humidity_outside': 16.610135379089392, 'Radiation_inside': 111.66727828792213, 'Radiation_outside': 336.167962355297, 'Wind_speed_outside': 5.617700492330189}

STAND_VALUES = Data(mean_values, std_values) 




URL = "opc.tcp://192.168.15.177:49320"

# OPC_PARAMETERS = ['OPC_INVER_TEMP_INTERIOR_S1', 'OPC_INVER_HR_INTERIOR_S1',
                  
#                   'OPC_UVCEN1_1_POS_VALOR',
#                   'OPC_UVCEN1_2_POS_VALOR',
#                   'OPC_UVCEN1_3_POS_VALOR',
#                   'OPC_UVLAT1N_POS_VALOR',
#                   'OPC_UVLAT1ON_POS_VALOR',
#                   'OPC_UVLAT1S_POS_VALOR',
#                   'OPC_UVLAT1OS_POS_VALOR',
                  
#                   'OPC_UVCEN2_1_POS_VALOR',
#                   'OPC_UVCEN2_2_POS_VALOR',
#                   'OPC_UVCEN2_3_POS_VALOR',
#                   'OPC_UVLAT2E_POS_VALOR',
#                   'OPC_UVLAT2N_POS_VALOR',
#                   'OPC_UVLAT2S_POS_VALOR',
                  
#                   'OPC_TEMP_EXTERIOR_10M', 'OPC_HR_EXTERIOR_10M', 
#                   'OPC_INVER_RADGLOBAL_INTERIOR_S1', 'OPC_RADGLOBAL_EXTERIOR_10M', 
#                   'OPC_VELVIENTO_EXTERIOR_10M'
#                   ]  

OPC_PARAMETERS = ['OPC_INVER_TEMP_INTERIOR_S1', 'OPC_INVER_HR_INTERIOR_S1',
                  'OPC_TEMP_EXTERIOR_10M', 'OPC_HR_EXTERIOR_10M', 
                  'OPC_INVER_RADGLOBAL_INTERIOR_S1', 'OPC_RADGLOBAL_EXTERIOR_10M', 
                  'OPC_VELVIENTO_EXTERIOR_10M'
                  ]  

OPC_CONTROL = ['OPC_UVCEN1_1_POS',
               'OPC_UVCEN1_2_POS',
               'OPC_UVCEN1_3_POS',
               'OPC_UVLAT1N_POS',
               'OPC_UVLAT1ON_POS',
               'OPC_UVLAT1S_POS',
               'OPC_UVLAT1OS_POS',
               
               'OPC_UVCEN2_1_POS',
               'OPC_UVCEN2_2_POS',
               'OPC_UVCEN2_3_POS',
               'OPC_UVLAT2E_POS',
               'OPC_UVLAT2N_POS',
               'OPC_UVLAT2S_POS'
               ]

OPC_CONTROL_READ = ['OPC_UVCEN1_1_POS_VALOR',
                   'OPC_UVCEN1_2_POS_VALOR',
                   'OPC_UVCEN1_3_POS_VALOR',
                   'OPC_UVLAT1N_POS_VALOR',
                   'OPC_UVLAT1ON_POS_VALOR',
                   'OPC_UVLAT1S_POS_VALOR',
                   'OPC_UVLAT1OS_POS_VALOR',
                  
                   'OPC_UVCEN2_1_POS_VALOR',
                   'OPC_UVCEN2_2_POS_VALOR',
                   'OPC_UVCEN2_3_POS_VALOR',
                   'OPC_UVLAT2E_POS_VALOR',
                   'OPC_UVLAT2N_POS_VALOR',
                   'OPC_UVLAT2S_POS_VALOR'
                   ]

OPC_CONTROL_PARAM_DICT = {
    'names': [
        'Vent_S1_Roof_1',
        'Vent_S1_Roof_2',
        'Vent_S1_Roof_3',
        'Vent_S1_Side_NW',
        'Vent_S1_Side_S',
        'Vent_S1_Side_N',
        'Vent_S1_Side_SW',

        'Vent_S2_Roof_1',
        'Vent_S2_Roof_2',
        'Vent_S2_Roof_3',
        'Vent_S2_Side_E',
        'Vent_S2_Side_S',
        'Vent_S2_Side_N',
    ],
    'write_vars': [
        'OPC_UVCEN1_1_POS',
        'OPC_UVCEN1_2_POS',
        'OPC_UVCEN1_3_POS',
        'OPC_UVLAT1ON_POS',
        'OPC_UVLAT1S_POS',
        'OPC_UVLAT1N_POS',
        'OPC_UVLAT1OS_POS',

        'OPC_UVCEN2_1_POS',
        'OPC_UVCEN2_2_POS',
        'OPC_UVCEN2_3_POS',
        'OPC_UVLAT2E_POS',
        'OPC_UVLAT2S_POS',
        'OPC_UVLAT2N_POS',
    ],
    'read_vars': [
        'OPC_UVCEN1_1_POS_VALOR',
        'OPC_UVCEN1_2_POS_VALOR',
        'OPC_UVCEN1_3_POS_VALOR',
        'OPC_UVLAT1ON_POS_VALOR',
        'OPC_UVLAT1S_POS_VALOR',
        'OPC_UVLAT1N_POS_VALOR',
        'OPC_UVLAT1OS_POS_VALOR',

        'OPC_UVCEN2_1_POS_VALOR',
        'OPC_UVCEN2_2_POS_VALOR',
        'OPC_UVCEN2_3_POS_VALOR',
        'OPC_UVLAT2E_POS_VALOR',
        'OPC_UVLAT2S_POS_VALOR',
        'OPC_UVLAT2N_POS_VALOR',
    ],
    'open_vars': [
        'OPC_UVCEN1_1a',
        'OPC_UVCEN1_2a',
        'OPC_UVCEN1_3a',
        'OPC_UVLAT1ONa',
        'OPC_UVLAT1Sa',
        'OPC_UVLAT1Na',
        'OPC_UVLAT1OSa',

        'OPC_UVCEN2_1a',
        'OPC_UVCEN2_2a',
        'OPC_UVCEN2_3a',
        'OPC_UVLAT2Ea',
        'OPC_UVLAT2Sa',
        'OPC_UVLAT2Na',
    ],
    'close_vars': [
        'OPC_UVCEN1_1c',
        'OPC_UVCEN1_2c',
        'OPC_UVCEN1_3c',
        'OPC_UVLAT1ONc',
        'OPC_UVLAT1Sc',
        'OPC_UVLAT1Nc',
        'OPC_UVLAT1OSc',

        'OPC_UVCEN2_1c',
        'OPC_UVCEN2_2c',
        'OPC_UVCEN2_3c',
        'OPC_UVLAT2Ec',
        'OPC_UVLAT2Sc',
        'OPC_UVLAT2Nc',
    ],
    'encoder_vars': [
        'OPC_INVER_UVCEN11_Encoder',
        'OPC_INVER_UVCEN12_Encoder',
        'OPC_INVER_UVCEN13_Encoder',
        'OPC_INVER_UVLAT1NO_Encoder',
        'OPC_INVER_UVLAT1S_Encoder',
        'OPC_INVER_UVLAT1N_Encoder',
        'OPC_INVER_UVLAT1SO_Encoder',

        'OPC_INVER_UVCEN21_Encoder',
        'OPC_INVER_UVCEN22_Encoder',
        'OPC_INVER_UVCEN23_Encoder',
        'OPC_INVER_UVLAT2E_Encoder',
        'OPC_INVER_UVLAT2S_Encoder',
        'OPC_INVER_UVLAT2N_Encoder',
    ],
    'open_time': [
        65, 63, 62, 237, 313, 245, 300,
        66, 67, 67, 308, 319, 259
    ],
    'close_time': [
        63, 62, 61, 237, 313, 244, 299,
        65, 66, 65, 307, 318, 259
    ]
}

    # Sensor IDs from your Postman collection
SENSOR_IDS = [
        "PLC1-PLC1-INVER_TEMP_INTERIOR_S1_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_HR_INTERIOR_S1_Agroconnect_Greenhouse_Modbus",
        "PLC6-PLC6-TEMP_EXTERIOR_10M_Agroconnect_Greenhouse_Modbus",
        "PLC6-PLC6-HR_EXTERIOR_10M_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_RADGLOBAL_INTERIOR_S1_Agroconnect_Greenhouse_Modbus",
        "PLC6-PLC6-RADGLOBAL_EXTERIOR_10M_Agroconnect_Greenhouse_Modbus",
        "PLC6-PLC6-VELVIENTO_EXTERIOR_10M_Agroconnect_Greenhouse_Modbus",

        "PLC1-PLC1-INVER_UVCEN11_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVCEN12_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVCEN13_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVLAT1NO_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVLAT1S_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVLAT1N_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC1-PLC1-INVER_UVLAT1SO_Encoder_Agroconnect_Greenhouse_Modbus",

        "PLC2-PLC2-INVER_UVCEN21_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC2-PLC2-INVER_UVCEN22_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC2-PLC2-INVER_UVCEN23_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC2-PLC2-INVER_UVLAT2E_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC2-PLC2-INVER_UVLAT2S_Encoder_Agroconnect_Greenhouse_Modbus",
        "PLC2-PLC2-INVER_UVLAT2N_Encoder_Agroconnect_Greenhouse_Modbus"
    ]



# COLUMNS = ["Date", 
#            "Temperature_inside", "Humidity_inside", 
           
#            "Vent_S1_Roof_1", 
#            "Vent_S1_Roof_2", 
#            "Vent_S1_Roof_3",
#            "Vent_S1_Side_N", 
#            "Vent_S1_Side_NW", 
#            "Vent_S1_Side_S", 
#            "Vent_S1_Side_SW",
           
#            "Vent_S2_Roof_1", 
#            "Vent_S2_Roof_2", 
#            "Vent_S2_Roof_3",
#            "Vent_S2_Side_E", 
#            "Vent_S2_Side_N", 
#            "Vent_S2_Side_S",
           
#            "Temperature_outside", "Humidity_outside", 
#            "Radiation_inside", "Radiation_outside",
#            "Wind_speed_outside"
#           ]

COLUMNS = ["Date", 
           "Temperature_inside", "Humidity_inside", 
           "Temperature_outside", "Humidity_outside", 
           "Radiation_inside", "Radiation_outside",
           "Wind_speed_outside",

            'Vent_S1_Roof_1',
            'Vent_S1_Roof_2',
            'Vent_S1_Roof_3',
            'Vent_S1_Side_NW',
            'Vent_S1_Side_S',
            'Vent_S1_Side_N',
            'Vent_S1_Side_SW',

            'Vent_S2_Roof_1',
            'Vent_S2_Roof_2',
            'Vent_S2_Roof_3',
            'Vent_S2_Side_E',
            'Vent_S2_Side_S',
            'Vent_S2_Side_N',
          ]

# COLUMNS = ['Date'] + OPC_PARAMETERS
