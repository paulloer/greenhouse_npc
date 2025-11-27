import time
from datetime import datetime, timedelta
# from opc_client import gh_client
from constants import TIME_CONTROL, URL, OPC_CONTROL_PARAM_DICT
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock

from postman import remote_read
from fiware_send import fiware_send

# control loop with encoder and PLC from remote
def controller_loop(shared_measurements: ListProxy, shared_vent_postion_targets: ListProxy, lock_m: Lock, lock_v:Lock) -> None:

    vent_names = OPC_CONTROL_PARAM_DICT['names']

    scada_vents = ['Vent_S1_Side_N']
    broken_vents = ['Vent_S2_Side_S']

    last_log_time = ""

    last_target = -1.0

    while True:

        current_measurement_time, current_measurement = remote_read()
        

        # Check if it is time to update vent positions and take measurement
        if current_measurement_time != last_log_time:
            
            print('\n' + current_measurement_time)
            print(current_measurement[7:20])
            with lock_m:
                shared_measurements.append([current_measurement_time] + current_measurement)

            last_log_time = current_measurement_time
        
        with lock_v:
            target = float(shared_vent_postion_targets[-1])
        if target is None:
            continue  # No target → nothing to do


        # technically, this could stay in the if statement, but to make sure the newly calculated control is send directly, it is sent every second
        if target != last_target:
            fiware_send(float(target))
            print(f"Send next target to remote. Vents will move to {target:.2f} %.")
            last_target = target

        time.sleep(1)  # Fast enough to respond quickly without overloading CPU


# # control loop with encoders
# def controller_loop(shared_measurements: ListProxy, shared_vent_postion_targets: ListProxy, lock_m: Lock, lock_v:Lock) -> None:

#     gh13_client = gh_client(URL)

#     vent_names = OPC_CONTROL_PARAM_DICT['names']
#     open_vars = OPC_CONTROL_PARAM_DICT['open_vars']
#     close_vars = OPC_CONTROL_PARAM_DICT['close_vars']
#     write_vars = OPC_CONTROL_PARAM_DICT['write_vars']
#     encoder_vars = OPC_CONTROL_PARAM_DICT['encoder_vars']

#     scada_vents = ['Vent_S1_Side_N']
#     broken_vents = ['Vent_S2_Side_S']

#     # reset system
#     gh13_client.write(open_vars, 0)
#     gh13_client.write(close_vars, 0)

#     current_positions = gh13_client.read(encoder_vars, names=vent_names)
#     motion_status = {name: 'at target' for name in vent_names}  # 'open', 'close', or 'at target'
#     start_times = {name: 0 for name in vent_names}

#     last_log_time = datetime.min
#     run_time = 0

#     try:
#         while True:
#             current_time = datetime.now()

#             current_positions = gh13_client.read(encoder_vars, names=vent_names)

#             # Check if forecast update is needed
#             if (current_time - last_log_time).total_seconds() >= TIME_CONTROL - run_time:
#                 last_log_time = current_time
#                 measurements = gh13_client.read(OPC_PARAMETERS)

#                 print('\n' + current_time.strftime('%Y-%m-%d %H:%M:%S'))
#                 for name, pos in current_positions.items():
#                     print(f"{name}: {pos}")
#                 with lock_m:
#                     shared_measurements.append([current_time.strftime('%Y-%m-%d %H:%M:%S')] + list(measurements.values()) + list(current_positions.values()))

#             for idx, name in enumerate(vent_names):
#                 with lock_v:
#                     target = shared_vent_postion_targets[-1]
#                 if target is None:
#                     continue  # No target → nothing to do

#                 current_pos = current_positions.get(name)

#                 diff = target - current_pos
#                 moving = motion_status[name]

#                 if name in scada_vents:
#                     if abs(diff) < 1:
#                         motion_status[name] = 'at target'
#                     else:
#                         # for avoiding frequent writes on the OPC
#                         if diff > 0 and moving != 'open':
#                             gh13_client.write([write_vars[idx]], target)
#                             motion_status[name] = 'open'
#                         elif diff < 0 and moving != 'close':
#                             gh13_client.write([write_vars[idx]], target)
#                             motion_status[name] = 'close'
#                 if name in broken_vents:
#                     continue
#                 else:
#                     if abs(diff) < 2 and moving != 'at target':  # Already close enough to target # TODO make hysterisis changeable
#                         gh13_client.write([open_vars[idx]], 0)
#                         gh13_client.write([close_vars[idx]], 0)
#                         motion_status[name] = 'at target'
#                     elif abs(diff) < 2 and moving == 'at target':
#                         continue
#                     else:
#                         if diff > 0 and moving != 'open':
#                             # Start opening
#                             gh13_client.write([close_vars[idx]], 0)
#                             gh13_client.write([open_vars[idx]], 1)
#                             start_times[name] = time.time()
#                             motion_status[name] = 'open'
#                         elif diff < 0 and moving != 'close':
#                             # Start closing
#                             gh13_client.write([open_vars[idx]], 0)
#                             gh13_client.write([close_vars[idx]], 1)
#                             start_times[name] = time.time()
#                             motion_status[name] = 'close'
#             time.sleep(0.1)  # Fast enough to respond quickly without overloading CPU
#             run_time = (datetime.now() - current_time).total_seconds()

#     except KeyboardInterrupt:
#         # reset system
#         gh13_client.write(open_vars, 0)
#         gh13_client.write(close_vars, 0)



# control loop without encoders
# def controller_loop_without_encoders(shared_measurements, shared_vent_postion_targets, lock_m, lock_v):

#     gh13_client = gh_client(URL)

#     vent_names = OPC_CONTROL_PARAM_DICT['names']
#     open_vars = OPC_CONTROL_PARAM_DICT['open_vars']
#     close_vars = OPC_CONTROL_PARAM_DICT['close_vars']
#     write_vars = OPC_CONTROL_PARAM_DICT['write_vars']
#     encoder_vars = OPC_CONTROL_PARAM_DICT['encoder_vars']
#     open_times = OPC_CONTROL_PARAM_DICT['open_time']
#     close_times = OPC_CONTROL_PARAM_DICT['close_time']
    
#     scada_vents = ['Vent_S1_Side_N']
#     broken_vents = ['Vent_S2_Side_S']

#     print('Opening all vents to 100%')
#     gh13_client.write(close_vars, 0)
#     gh13_client.write(open_vars, 1)
#     time.sleep(max(open_times))
#     gh13_client.write(open_vars, 0)
#     print('All vents at 100%')

#     current_positions = {name: 0 if name in broken_vents else 100 for name in vent_names}
#     motion_status = {name: 'at target' for name in vent_names}  # 'open', 'close', or 'at target'
#     start_times = {name: 0 for name in vent_names}

#     last_log_time = datetime.min
#     run_time = 0

#     while True:
#         current_time = datetime.now()

#         # Check if forecast update is needed
#         if (current_time - last_log_time).total_seconds() >= TIME_CONTROL - run_time:
#             last_log_time = current_time
#             measurements = gh13_client.read(OPC_PARAMETERS)

#             calc_measurements = []
#             for name in vent_names:
#                 pos = current_positions.get(name)
#                 if pos is not None:
#                     print(f"[{current_time}] {name} position: {pos}")
#                     calc_measurements.append(pos)

#             with lock_m:
#                 shared_measurements.append([current_time.strftime('%Y-%m-%d %H:%M:%S')] + list(measurements.values()) + calc_measurements)

#         # Calculate all current positions
#         for idx, name in enumerate(vent_names):
#             if name in scada_vents:
#                 current_positions[name] = gh13_client.read([encoder_vars[idx]])
#             elif name in broken_vents:
#                 continue
#             else:
#                 if motion_status[name] == 'close':
#                     current_positions[name] = current_positions[name] - 100 * (time.time()-start_times[name]) / close_times[idx]
#                     start_times[name] = time.time()
#                 elif motion_status[name] == 'open':
#                     current_positions[name] = current_positions[name] + 100 * (time.time()-start_times[name]) / open_times[idx]
#                     start_times[name] = time.time()

        
#         for idx, name in enumerate(vent_names):
#             with lock_v:
#                 target = shared_vent_postion_targets[-1]
#             if target is None:
#                 continue  # No target → nothing to do

#             current_pos = current_positions.get(name)

#             diff = target - current_pos
#             moving = motion_status[name]

#             if name in scada_vents:
#                 if abs(diff) < 1:
#                     gh13_client.write([close_vars[idx]], 0)
#                     motion_status[name] = 'at target'
#                 else:
#                     # for avoiding frequent writes on the OPC
#                     if diff > 0 and moving != 'open':
#                         gh13_client.write([write_vars[idx]], target)
#                         motion_status[name] = 'open'
#                     elif diff < 0 and moving != 'close':
#                         gh13_client.write([write_vars[idx]], target)
#                         motion_status[name] = 'close'
#             if name in broken_vents:
#                 continue
#             else:
#                 if abs(diff) < 1:  # Already close enough to target
#                     gh13_client.write([open_vars[idx]], 0)
#                     gh13_client.write([close_vars[idx]], 0)
#                     motion_status[name] = 'at target'
#                 else:
#                     if diff > 0 and moving != 'open':
#                         # Start opening
#                         gh13_client.write([close_vars[idx]], 0)
#                         gh13_client.write([open_vars[idx]], 1)
#                         start_times[name] = time.time()
#                         motion_status[name] = 'open'
#                     elif diff < 0 and moving != 'close':
#                         # Start closing
#                         gh13_client.write([open_vars[idx]], 0)
#                         gh13_client.write([close_vars[idx]], 1)
#                         start_times[name] = time.time()
#                         motion_status[name] = 'close'

#         time.sleep(0.5)  # Fast enough to respond quickly without overloading CPU
#         run_time = (datetime.now() - current_time).total_seconds()
