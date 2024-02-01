import os
from random import shuffle, randint

dyad_num = 3
date = '2-2-24'
pathname = f'jarv3d 1.7.0/Records/User/{date}'

if not os.path.isdir(pathname):
    os.mkdir(pathname)

def gen_emotions():
    base_emotions = ['h', 'p', 'd', 's']
    shuffle(base_emotions)
    
    emotions = ['b', 'b', base_emotions[0], base_emotions[0], 
                base_emotions[1], base_emotions[1], 
                base_emotions[2], base_emotions[2], 
                base_emotions[3], base_emotions[3]]
    
    return base_emotions, emotions

day_order_int = 1
for i in range(dyad_num):
    first_participant_id = f'{randint(0, 999999):06d}'
    second_participant_id= f'{randint(0, 999999):06d}'
    dyad_id = f'{randint(0, 999999):06d}'
    session_num = 'P1'
    subject_orders = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

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