#!/usr/bin/bash

echo "Switching from Exploration mode (STAGE1) to Rescue Mode (STAGE2)";
rosrun dynamic_reconfigure dynparam set /fsm current_mode "stage2"