#!/usr/bin/env python3

import sys
import os

'''
Possible options: (please input these exactly as shown)

tree
fire_hydrant
boat
'''

objects = ""

for i, obj in enumerate(sys.argv[1:]):
    if i == 0:
        objects = obj
    else:
        objects = objects + ',' + obj

os.system('rosrun dynamic_reconfigure dynparam set /fsm objects_to_rescue {}'.format(objects))
print('Successfully changed objects to {}'.format(objects))