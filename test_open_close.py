import time
from datetime import datetime, timedelta
from opc_client import gh_client
from constants import TIME_CONTROL, URL, OPC_CONTROL_PARAM_DICT, OPC_PARAMETERS

shared_measurements = []
shared_vent_postion_targets = [100]

gh13_client = gh_client(URL)



vent_names = OPC_CONTROL_PARAM_DICT['names']
open_vars = OPC_CONTROL_PARAM_DICT['open_vars']
close_vars = OPC_CONTROL_PARAM_DICT['close_vars']
write_vars = OPC_CONTROL_PARAM_DICT['write_vars']
encoder_vars = OPC_CONTROL_PARAM_DICT['encoder_vars']

scada_vents = ['Vent_S1_Side_N']
broken_vents = ['Vent_S2_Side_S']

# reset system
gh13_client.write(open_vars, 0)
gh13_client.write(close_vars, 0)

current_positions = gh13_client.read(encoder_vars, names=vent_names)
motion_status = {name: 'at target' for name in vent_names}  # 'open', 'close', or 'at target'
start_times = {name: 0 for name in vent_names}

last_log_time = datetime.min

try:
    while True:
        tic = time.time()
        current_time = datetime.now()

        current_positions = gh13_client.read(encoder_vars, names=vent_names)

        # Check if forecast update is needed
        if (current_time - last_log_time).total_seconds() >= TIME_CONTROL:
            last_log_time = current_time
            measurements = gh13_client.read(OPC_PARAMETERS)

            print('\n' + current_time.strftime('%Y-%m-%d %H:%M:%S'))
            for name, pos in current_positions.items():
                print(f"{name}: {pos}")

            shared_measurements.append([current_time.strftime('%Y-%m-%d %H:%M:%S')] + list(measurements.values()) + list(current_positions.values()))

        for idx, name in enumerate(vent_names):
            target = shared_vent_postion_targets[-1]
            if target is None:
                continue  # No target → nothing to do

            current_pos = current_positions.get(name)

            diff = target - current_pos
            moving = motion_status[name]

            if name in scada_vents:
                if abs(diff) < 1:
                    motion_status[name] = 'at target'
                else:
                    # for avoiding frequent writes on the OPC
                    if diff > 0 and moving != 'open':
                        gh13_client.write([write_vars[idx]], target)
                        motion_status[name] = 'open'
                    elif diff < 0 and moving != 'close':
                        gh13_client.write([write_vars[idx]], target)
                        motion_status[name] = 'close'
            if name in broken_vents:
                continue
            else:
                if abs(diff) < 2 and moving != 'at target':  # Already close enough to target
                    gh13_client.write([open_vars[idx]], 0)
                    gh13_client.write([close_vars[idx]], 0)
                    motion_status[name] = 'at target'
                elif abs(diff) < 2 and moving == 'at target':
                    continue
                else:
                    if diff > 0 and moving != 'open':
                        # Start opening
                        gh13_client.write([close_vars[idx]], 0)
                        gh13_client.write([open_vars[idx]], 1)
                        start_times[name] = time.time()
                        motion_status[name] = 'open'
                    elif diff < 0 and moving != 'close':
                        # Start closing
                        gh13_client.write([open_vars[idx]], 0)
                        gh13_client.write([close_vars[idx]], 1)
                        start_times[name] = time.time()
                        motion_status[name] = 'close'


        time.sleep(0.1)  # Fast enough to respond quickly without overloading CPU
        toc = time.time()
        print(f'time for while loop is {(toc-tic):.3f} s')

except KeyboardInterrupt:
    # reset system
    gh13_client.write(open_vars, 0)
    gh13_client.write(close_vars, 0)