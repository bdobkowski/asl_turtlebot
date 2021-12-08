#!/usr/bin/bash

echo "Killing all Gazebo and ROS processes...";
killall gzclient;killall gzserver;killall roscore;killall rosout