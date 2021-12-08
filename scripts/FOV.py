#!/usr/bin/env python3

# Publishing the camera FOV lines
import math
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def publisher():
    FOV_pub = rospy.Publisher('FOV_topic', Marker, queue_size = 10)
    rospy.init_node('FOV_node', anonymous=True)
    rospy.loginfo('Publishing the camera FOV lines')
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        marker = Marker()
        
        marker.header.frame_id = "base_footprint" # CHECK ON THIS FRAME!!!
        marker.header.stamp  = rospy.Time()
        marker.type = marker.LINE_LIST
        marker.id = 0
        
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0.1
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.05 # Line width
        marker.scale.y = 0.25 # not needed?
        marker.scale.z = 0.25 # not needed?
        
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Dimensions for the FOV viewer
        d = .5 # Arbitrary design choice
        theta = 1.3962634 # FOV angle from xacro file
        w = 2*d*math.tan(theta/2)
        h = w
       
        marker.points = []

        R = Point() # Robot base point
        R.x, R.y, R.z = (0,0,0)
        
        A = Point() # Top left
        A.x, A.y, A.z = (d, w/2, h/2)

        B = Point() # Top right
        B.x, B.y, B.z = (d, -w/2, h/2)

        C = Point() # Bottom left
        C.x, C.y, C.z = (d, w/2, -h/2)

        D = Point() # Bottom right
        D.x, D.y, D.z = (d, -w/2, -h/2)

        # Add the points in the correct order to generate the lines
        marker.points.extend([R,A])
        marker.points.extend([R,B])
        marker.points.extend([R,C])
        marker.points.extend([R,D])
        marker.points.extend([A,B])
        marker.points.extend([C,D])
        marker.points.extend([A,C])
        marker.points.extend([B,D])

        FOV_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
