#!/usr/bin/env python3
#!/usr/bin/env python3

# FSM: Combination of supervisor and navigator functions
# Including A* navigation and the ability to stop at stop signs, objects, ...

# TO-DO
# Improve how we load the waypoints (random values are hard-coded rn)
# Figure out how to switch the detection method between the exploring/saving and rescuing phases
# Make sure all of the necessary controllers are in the controllers folder (pose, trajectory, heading)
# A lot of the logic is in the PARK mode in the state machine section, but this might not be the best place for it
# Check: After rescuing all of the objects, is CROSS the right one to transition to? (should navigate back to origin)
# Check: Do we also need to return to the starting point between Stage 1 and Stage 2?
# How should we define the order of the objects to rescue? Currently, in same order as inputted in param

# NOTES:
# - We will be passing in the list of objects to rescue as a rosparam
# - The current project stage (1 or 2) will be passed in as a rosparam too

# Should we consider a DetOccupancyGrid?

import rospy
from os import name, nice
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped, PoseArray
from std_msgs.msg import String, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum
import time

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig
from asl_turtlebot.cfg import FSMConfig

# MY IMPORTS
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
#from asl_turtlebot.msg import IsItFound
from gazebo_msgs.msg import ModelStates

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3

    # Stuff from the supervisor
    STOP = 4
    CROSS = 5
    RESCUING = 6
    SAVING = 7

# Another mode defining whether we are in the exploring or rescuing stage of the project
class ProjectMode(Enum):
    STAGE1 = 1
    STAGE2 = 2

class FSM:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("fsm", anonymous=True)
        self.mode = Mode.IDLE
        self.switch_mode(Mode.IDLE)
        self.projectMode = ProjectMode.STAGE1


        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2  # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.05 # Previously 0.15
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0.0, 0.0, 0.0, self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher("/planned_path", Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher("/cmd_smoothed_path", Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher("/cmd_smoothed_path_rejected", Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.exploration_pub = rospy.Publisher("/exploration_progress", String, queue_size=10)
        self.marker_pub = rospy.Publisher('/marker_topic_array', MarkerArray, queue_size=10) # Jack's Marker Pub
        
        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)        

        # MY VARIABLES
        #need to update objects
        self.found_objects = {'house':False, 'tree':False, 'skyscraper':False, 'tent':False, 'boat':False} # initialize
        self.object_locations = {'house':(None, None), 'tree':(None, None), 'skyscraper':(None, None), 'tent':(None, None), 'boat':(None, None)} # initialize
        self.stored = False # to check if we have stored the objects to rescue or not
        self.num_obj_to_rescue = None
        self.x_init = None
        self.y_init = None
        self.theta_init = None
        self.window_size = 8
        # NOTE: Need to update these!!
        # self.waypoints = [(4.25, 0.9, 0.0),(4.35, 1.9, 0.0), (4.05,1.3, 0.0)] # NOTE initial list of waypoints for testing only!!!
        self.waypoints = [
            (0.287, 1.743, 0.5*np.pi), #first one is the starting point
            (0.431, 2.768, 0.5*np.pi), #just realized all the angles are 180 deg off
            (2.012, 2.997, 0.25*np.pi), #that's why the robot spins round so much
            (3.588, 3.574, 0.5*np.pi),
            (4.431, 3.482, -np.pi),
            (4.553, 1.784, np.pi),
            (4.087, 0.413, np.pi),
            (2.469, 0.482, 0.5*np.pi),
            (2.572, 2.995, np.pi),
            (0.319, 2.770, -0.5*np.pi),
            (0.365, 1.349, -np.pi),
            (1.469, 0.470, np.pi),
            (0.287, 1.743, 0.5*np.pi) #last one is the initial starting point
        ]
        # self.markerarray = MarkerArray()
        self.marker_array_msg = MarkerArray()
        self.currentWPind = 0
        self.currentRescueID = -1
        self.num_obj_rescued = 0
        self.objects_to_rescue = []

        self.pos_eps = rospy.get_param("~pos_eps", 0.1) # Threshold at which we consider the robot at a location
        self.theta_eps = rospy.get_param("~theta_eps", 0.3) # Threshold at which we consider the robot at a location
        self.stop_time = rospy.get_param("~stop_time", 3.) # Pause duration when at a stop sign
        self.save_time = rospy.get_param("~save_time", 1.) # Pause duration when saving the location of an object
        self.rescue_time = rospy.get_param("~rescue_time", 3.) # Pause duration when rescuing an object
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5) # Minimum distance from a stop sign to obey it
        self.crossing_time = rospy.get_param("~crossing_time", 3.) # Time taken to cross an intersection

        # SUBSCRIBERS
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.object_detected_callback)

        print("finished init")
    
    def make_markerarray(self, points):
        
        for i in range(len(points)):
            x, y, th = points[i]
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time()
            marker.id = i #0
            marker.text = str(i)
            marker.type = 2 # sphere
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0 
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.marker_array_msg.markers.append(marker) 

    # CALLBACKS
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}, current_mode:{current_mode}, objects_to_rescue:{objects_to_rescue}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]

        mode = config["current_mode"]

        # Resetting project mode based on dyamic reconfiguration
        if mode == "Stage1" or mode == "stage1" or mode == "STAGE1":
            self.projectMode = ProjectMode.STAGE1
            rospy.loginfo("Switched project mode to {}".format(self.projectMode))
        elif mode == "Stage2" or mode == "stage2" or mode == "STAGE2":
            self.projectMode = ProjectMode.STAGE2
            rospy.loginfo("Switched project mode to {}".format(self.projectMode))
        else:
            rospy.loginfo("Invalid Mode to Switch to - Keeping Current Mode of Operation: {}".format(self.projectMode))

        # Setting list of objects to rescue based on dynamic reconfiguration
        object_string = config["objects_to_rescue"]
        if object_string == "Nothing":
            self.objects_to_rescue = None
        else:
            rospy.loginfo('Objects to be rescued: {}'.format(object_string))
            self.objects_to_rescue = object_string.split(",")
            if set(self.objects_to_rescue).issubset(set(self.object_locations.keys())):
            # if self.objects_to_rescue[i] in self.object_locations for i in len(self.objects_to_rescue).keys:
                # This can be improved, might want to go to the closest one first
                self.switch_mode(Mode.ALIGN)
                self.iterate_rescueTarget()
        
        return config

    # Callbacks from original code - Good to go
    def cmd_nav_callback(self, data): # Loads in goal if different from current goal; replans
        """ loads in goal if different from current goal, and replans  """
        if ( data.x != self.x_g  or  data.y != self.y_g  or  data.theta != self.theta_g ):
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()
    def map_md_callback(self, msg): # Receives map metadata; stores it
        """ receives maps meta data and stores it """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)
    def map_callback(self, msg): # Receives new map info and updates the map
        """ receives new map info and updates the map  """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if ( self.map_width > 0  and  self.map_height > 0  and  len(self.map_probs) > 0 ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                self.window_size,
                self.map_probs,
            )
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan
    def shutdown_callback(self): # Publishes 0 velocity when ros is shut down
        """  publishes zero velocities upon rospy shutdown  """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    # MY CALLBACKS
    def stop_sign_detected_callback(self, msg):
        dist = msg.distance # distance of the stop sign
        # if close enough and in a navigation mode (e.g. not CROSS), start the stop sign process
        if dist > 0 and dist < self.params.stop_min_dist and (self.mode == Mode.ALIGN or self.mode == Mode.TRACK or self.mode == Mode.PARK):
            self.init_stop_sign()
    
    def object_detected_callback(self, msg):
        # Only need to perform this callback if we are in stage 1 (recording locations)
        # (Detector does not need to be run during Stage 2 (rescue) because we already know where these things are)
        if self.projectMode == ProjectMode.STAGE1:
            # Get info from message
            objectsList = msg.objects # Object names, list of strings
            objectMessages = msg.ob_msgs # Object messages list, of form DetectedObject, includes id, name, confidence, distance, thetaleft, thetaright, corners

            num_obj = len(objectsList)

            # check if we have a non-empty message and that at least one object has not been found yet
            if num_obj>0 and any(x==False for x in self.found_objects.values()): 
                # Begin the saving process if so
                self.init_saving(objectsList, objectMessages)
    
    # Booleans from original code - good to go
    def near_goal(self): # Returns T/F: Robot is close enough to goal to start using pose control? 
        """ returns whether the robot is close enough in position to the goal to
        start using the pose controller  """
        return ( linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g])) < self.near_thresh )
    def at_goal(self): # Returns T/F: Robot has reached goal position accurately?
        """ returns whether the robot has reached the goal position with enough
            accuracy to return to idle state """
        return (linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g])) < self.at_thresh
                and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta )
    def aligned(self): # Returns T/F: Robot is aligned with start of path enough to start using tracking control?
        """  returns whether robot is aligned with starting direction of path
             (enough to switch to tracking controller) """
        return ( abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh )
    def close_to_plan_start(self): # Returns T/F: Current robot x,y position is close enough to the start of the plan?
        return ( abs(self.x - self.plan_start[0]) < self.start_pos_thresh
                 and abs(self.y - self.plan_start[1]) < self.start_pos_thresh )
    
    # Other functions from original code - good to go (Unmodified)
    def snap_to_grid(self, x): # Returns (x,y) position on occupancy grid
        return (self.plan_resolution * round(x[0] / self.plan_resolution),
                self.plan_resolution * round(x[1] / self.plan_resolution)  )
    def switch_mode(self, new_mode): # Switches self.mode to input mode and logs
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode
    def publish_planned_path(self, path, publisher): # Publishes planned path for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)
    def publish_smoothed_path(self, traj, publisher): # Publishes smoothed planned path for visualization
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """ Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded  """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            try:
                V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
            except:
                return
        elif self.mode == Mode.TRACK:
            try:
                V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
            except:
                return
        elif self.mode == Mode.ALIGN:
            try:
                V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
            except:
                return
        elif self.mode == Mode.CROSS:
            try:
                V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
            except:
                return
        else:
            V = 0.0
            om = 0.0

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    # MY FUNCTIONS
    def stay_idle(self):
        """ sends zero velocity to stay put """
        vel_g_msg = Twist()

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """
        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps and \
               abs(theta - self.theta) < self.params.theta_eps

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """
        self.stop_sign_start = rospy.get_rostime()
        self.switch_mode(Mode.STOP)

    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)
    
    def has_saved(self):
        """ checks if saving is over """
        return self.mode == Mode.SAVING and \
               rospy.get_rostime() - self.saving_start > rospy.Duration.from_sec(self.params.save_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        self.cross_start = rospy.get_rostime()
        self.switch_mode(Mode.CROSS)

    def has_crossed(self):
        """ checks if crossing maneuver is over """
        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)

    def init_rescuing(self):
        self.rescue_start = rospy.get_rostime()
        self.switch_mode(Mode.RESCUING)
        # NOTE: need to add some stuff here about what objects have been rescuing, marking as rescued, ...
        # Some stuff we have from the other function:
        # self.objectsToRescue is a list of names of objects we want to rescue
        # self.num_obj_to_rescue = is the number of objects we rescue
        # self.isRescued is a dictionary containing name => boolean(is this object rescued? True or False)

    def mark_rescued(self, objectName):
        self.isRescued[objectName] = True
        self.num_obj_rescued = sum(self.isRescued.values()) # update: Count the number of True values in dict

    def has_rescued(self):
        """ checks if rescuing maneuver is over """
        return self.mode == Mode.RESCUING and \
               rospy.get_rostime() - self.rescue_start > rospy.Duration.from_sec(self.params.rescue_time)

    def init_saving(self, objectsList, objectMessages):
        self.saving_start = rospy.get_rostime()
        num_obj = len(objectsList)
        for i in range(num_obj):
            # Load the parameters from the objectMessages format
            obj = objectMessages[i]
            # id = obj.id
            name = obj.name
            # confidence = obj.confidence
            distance = obj.distance
            # thetaleft = obj.thetaleft
            # thetaright = obj.thetaright
            # corners = obj.corners
            if name in self.found_objects.keys(): # If this is an object we are expecting to see
                # If we are close enough to the previously-undiscovered object, save!!
                if distance > 0 and distance < self.params.stop_min_dist and self.found_objects[name] == False:
                    self.switch_mode(Mode.SAVING)
                    # Get the world frame coordinates of the detected object
                    x = self.x + distance*np.cos(self.theta)
                    y = self.y + distance*np.cos(self.theta)
                    # Assign the corresponding elements in the dictionaries if we haven't already done so
                    self.found_objects[name] = True
                    self.object_locations[name] = (x,y)

    def iterate_waypoint(self):
        # Sets the goal position to be the next waypoint
        self.currentWPind +=1
        x,y,th = self.waypoints[self.currentWPind - 1]
        self.x_g = x
        self.y_g = y
        self.theta_g = th
        self.make_markerarray(self.waypoints)
        self.marker_pub.publish(self.marker_array_msg)
        # #put marker stuff here
        # marker = Marker()
        # marker.header.frame_id = "map"
        # marker.header.stamp = rospy.Time()
        # marker.id = 0
        # marker.type = 2 # sphere
        # marker.pose.position.x = x
        # marker.pose.position.y = y
        # marker.pose.position.z = 0
        # marker.pose.orientation.x = 0.0
        # marker.pose.orientation.y = 0.0
        # marker.pose.orientation.z = 0.0
        # marker.pose.orientation.w = 1.0
        # marker.scale.x = 0.1
        # marker.scale.y = 0.1
        # marker.scale.z = 0.1
        # marker.color.a = 1.0 
        # marker.color.r = 1.0
        # marker.color.g = 0.0
        # marker.color.b = 0.0
        # self.marker_pub.publish(marker)
        # #end marker stuff

        percent_explored = 100*self.currentWPind / len(self.waypoints)

        self.exploration_pub.publish("Percent Explored: {}".format(percent_explored))

        self.replan()
    
    def iterate_rescueTarget(self):
        self.currentRescueID += 1
        x,y,th = self.object_locations[self.objects_to_rescue[self.currentRescueID]]
        self.x_g = x
        self.y_g = y
        self.theta_g = th
        self.replan()

    def returnHome(self):
        # Sets the goal position to be the initial position
        self.x_g = self.x_init
        self.y_g = self.y_init
        self.theta_g = self.theta_init
        self.replan()


    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (self.current_plan_duration - self.get_current_plan_time())

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)" )
                self.publish_smoothed_path( traj_new, self.nav_smoothed_path_rej_pub )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        savedInitPosition = False
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform("/map", "/base_footprint", rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]

                # NEW: NOTE: check to see if this is the best way to save the initial map position
                if not savedInitPosition:
                    self.x_init = self.x
                    self.y_init = self.y
                    self.theta_init = self.theta
                    savedInitPosition = True

            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass
            
            # In this main loop, check on our parameters, save these to variables
            # stageVal = rospy.get_param("~current_stage", 1) # default of 1, we will manually switch to 2
            # if stageVal == 1:
            #     self.projectMode = ProjectMode.STAGE1
            # elif stageVal == 2:
            #     self.projectMode = ProjectMode.STAGE2
            # else:
            #     raise Exception("Wrong project mode input!")

            # This runs once we input the rosparam of the things we want to rescue
            # will initialize some stuff about what we are gonna rescue
            if rospy.has_param("objectsToRescue") and not self.stored:
                self.objectsToRescue = rospy.get_param("objectsToRescue") # should be a list
                self.num_obj_to_rescue = len(self.objectsToRescue)
                # initialize the isRescued dictionary
                self.isRescued = dict(zip(self.objectsToRescue, [False for i in range(self.num_obj_to_rescue)]))
                self.stored = True
            
            #if self.projectMode == ProjectMode.STAGE2 and  #FINISH

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                if self.projectMode == ProjectMode.STAGE1:
                    if self.currentWPind == len(self.waypoints): # if at the last WP
                        self.x_g = None
                        self.y_g = None
                        self.theta_g = None
                        self.stay_idle()
                    elif not self.currentWPind > len(self.waypoints):
                        if self.occupancy:
                            self.iterate_waypoint()
                            self.switch_mode(Mode.ALIGN)
                        else:
                            self.stay_idle()
                    else:
                        self.stay_idle()
                else:
                    self.stay_idle()
            elif self.mode == Mode.ALIGN:
                if not hasattr(self.heading_controller, 'th_g'):
                    time.sleep(2)
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    # If we are at the last waypoint, idle
                    if self.projectMode == ProjectMode.STAGE1:
                        print('in mode PARK and currentWPind {} and waypoints {}'.format(self.currentWPind, self.waypoints))
                        if self.currentWPind == len(self.waypoints): # if at the last WP
                            self.x_g = None
                            self.y_g = None
                            self.theta_g = None
                            self.switch_mode(Mode.IDLE) # NOTE maybe switch to stage 2 automatically??
                        elif not self.currentWPind > len(self.waypoints):
                            self.iterate_waypoint()
                            self.switch_mode(Mode.ALIGN)
                        else:
                            self.switch_mode(Mode.IDLE)
                    else: # in Stage 2
                        if not self.num_obj_rescued == self.num_obj_to_rescue:# We are still trying to rescue objects (we have not rescued all objects) NOTE finish
                            self.init_rescuing()
                        elif not self.close_to(self.x_init, self.y_init, self.theta_init): # We have rescued all, but still need to return back to initial position
                            self.returnHome()
                        else: # We have returned to the initial position
                            self.switch_mode(Mode.IDLE)

            # MY FSM ADDITIONS
            elif self.mode == Mode.STOP:
                # At a stop sign
                if self.has_stopped():
                    self.init_crossing()
                    self.switch_mode(Mode.CROSS)
                else:
                    self.stay_idle()

            elif self.mode == Mode.CROSS:
                # Crossing an intersection
                if self.has_crossed():
                    self.switch_mode(Mode.ALIGN)
                # self.nav_to_pose() 
                # self.publish_control()
            
            elif self.mode == Mode.SAVING:
                if self.has_saved():
                    self.init_crossing()
                    self.switch_mode(Mode.CROSS)
                else:
                    self.stay_idle()

            elif self.mode == Mode.RESCUING:
                if self.has_rescued():
                    self.mark_rescued(self, self.objectsToRescue[self.currentRescueID])
                    self.iterate_rescueTarget()
                    self.init_crossing()
                    self.switch_mode(Mode.CROSS)
                else:
                    self.stay_idle()

            self.publish_control()
            rate.sleep()


if __name__ == "__main__":
    nav = FSM()
    rospy.on_shutdown(nav.shutdown_callback)
    print("========= FSM Successfully Launched ==========")
    nav.run()
