import os
import sys
from random import shuffle, randint

first_participant_id = ''
second_participant_id = ''
dyad_id = ''
date = '2-2-24'
day_order_int = 1

pathname = f'Data/{date}'

if not os.path.isdir(pathname):
    sys.exit(f"Folder '{pathname}' not found, please ensure the data has been added to the appropriate path")

for i in range(12):
    first_participant_id = f'{randint(0, 999999):06d}'
    second_participant_id= f'{randint(0, 999999):06d}'
    dyad_id = f'{randint(0, 999999):06d}'
    session_num = 'P1'
    subject_orders = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    os.mkdir(f'{pathname}/Dyad {i+1}')
    print(f'Dyad {i+1}')
    
    os.mkdir(f'{pathname}/Dyad {i+1}/Subject 1')
    base_emotions, emotions = gen_emotions()
    print(f'First Particpant ID: {first_participant_id}, Emotions: {base_emotions}')
    for idx in range(10):
        day_order = f'{day_order_int:02d}'
        os.mkdir(f'{pathname}/Dyad {i+1}/Subject 1/{first_participant_id}_{dyad_id}_{date}_{session_num}_{subject_orders[idx]}_{day_order}_solo_{emotions[idx]}')
        day_order_int = day_order_int + 1
    
    os.mkdir(f'{pathname}/Dyad {i+1}/Joint')    
    base_emotions, emotions = gen_emotions()
    print(f'Dyad ID: {dyad_id}, Emotions: {base_emotions}')
    for idx in range(10):
        day_order = f'{day_order_int:02d}'
        os.mkdir(f'{pathname}/Dyad {i+1}/Joint/{dyad_id}_{date}_{session_num}_{subject_orders[idx]}_{day_order}_joint_{emotions[idx]}')
        day_order_int = day_order_int + 1
    
    os.mkdir(f'{pathname}/Dyad {i+1}/Subject 2')    
    base_emotions, emotions = gen_emotions()
    print(f'Second Particpant ID: {second_participant_id}, Emotions: {base_emotions}')
    for idx in range(10):
        day_order = f'{day_order_int:02d}'
        os.mkdir(f'{pathname}/Dyad {i+1}/Subject 2/{second_participant_id}_{dyad_id}_{date}_{session_num}_{subject_orders[idx]}_{day_order}_solo_{emotions[idx]}')
        day_order_int = day_order_int + 1
    
    print()