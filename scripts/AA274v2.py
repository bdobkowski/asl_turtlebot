#!/usr/bin/env python3

# Publishing the camera FOV lines
import numpy as np
from numpy.core.numerictypes import ScalarType
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def publisher():
    AA274_pub = rospy.Publisher('AA274_topic', Marker, queue_size = 10)
    rospy.init_node('AA274_node', anonymous=True)
    rospy.loginfo('Publishing the AA274 marker')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        marker = Marker()
        
        marker.header.frame_id = "map" 
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
        
        marker.scale.x = 0.25 # Thickness of the lines
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        scale = 0.75 # DEFINES THE OVERALL SCALE OF THE LETTERS
        w = 1*scale
        h = 2*scale
        spacer = w/2
        
        # Defining origin to be the bottom left most point in a rectangular bounding box of the letter/number
        originA = np.array([0,4.5,0]) # Starting point for the bottom left most point of the first A
        originAA =    originA + np.array([   w + spacer , 0, 0])
        originTwo =   originA + np.array([2*(w + spacer), 0, 0])
        originSeven = originA + np.array([3*(w + spacer), 0, 0])
        originFour =  originA + np.array([4*(w + spacer), 0, 0])

        marker.points = []

        A1 = Point()
        A1.x, A1.y, A1.z = originA
        A2 = Point()
        A2.x, A2.y, A2.z = originA + np.array([0.5*w, 0, h])
        A3 = Point()
        A3.x, A3.y, A3.z = originA + np.array([w, 0, 0])
        A4 = Point()
        A4.x, A4.y, A4.z = originA + np.array([.25*w, 0, h/2])
        A5 = Point()
        A5.x, A5.y, A5.z = originA + np.array([.75*w, 0, h/2])

        AA1 = Point()
        AA1.x, AA1.y, AA1.z = originAA
        AA2 = Point()
        AA2.x, AA2.y, AA2.z = originAA + np.array([0.5*w, 0, h])
        AA3 = Point()
        AA3.x, AA3.y, AA3.z = originAA + np.array([w, 0, 0])
        AA4 = Point()
        AA4.x, AA4.y, AA4.z = originAA + np.array([.25*w, 0, h/2])
        AA5 = Point()
        AA5.x, AA5.y, AA5.z = originAA + np.array([.75*w, 0, h/2])

        Two1 = Point()
        Two1.x, Two1.y, Two1.z = originTwo + np.array([0, 0, h])
        Two2 = Point()
        Two2.x, Two2.y, Two2.z = originTwo + np.array([w, 0, h])
        Two3 = Point()
        Two3.x, Two3.y, Two3.z = originTwo + np.array([w, 0, h/2])
        Two4 = Point()
        Two4.x, Two4.y, Two4.z = originTwo + np.array([0, 0, h/2])
        Two5 = Point()
        Two5.x, Two5.y, Two5.z = originTwo
        Two6 = Point()
        Two6.x, Two6.y, Two6.z = originTwo + np.array([w, 0, 0])

        Seven1 = Point()
        Seven1.x, Seven1.y, Seven1.z = originSeven + np.array([0, 0, h])
        Seven2 = Point()
        Seven2.x, Seven2.y, Seven2.z = originSeven + np.array([w, 0, h])
        Seven3 = Point()
        Seven3.x, Seven3.y, Seven3.z = originSeven

        Four1 = Point()
        Four1.x, Four1.y, Four1.z = originFour + np.array([0, 0, h])
        Four2 = Point()
        Four2.x, Four2.y, Four2.z = originFour + np.array([0, 0, h/2])
        Four3 = Point()
        Four3.x, Four3.y, Four3.z = originFour + np.array([w, 0, h/2])
        Four4 = Point()
        Four4.x, Four4.y, Four4.z = originFour + np.array([w, 0, h])
        Four5 = Point()
        Four5.x, Four5.y, Four5.z = originFour + np.array([w, 0, 0])


        # Add the points in the correct order to generate the lines
        marker.points.extend([A1,A2])
        marker.points.extend([A2,A3])
        marker.points.extend([A4,A5])

        marker.points.extend([AA1,AA2])
        marker.points.extend([AA2,AA3])
        marker.points.extend([AA4,AA5])

        marker.points.extend([Two1,Two2])
        marker.points.extend([Two2,Two3])
        marker.points.extend([Two3,Two4])
        marker.points.extend([Two4,Two5])
        marker.points.extend([Two5,Two6])

        marker.points.extend([Seven1,Seven2])
        marker.points.extend([Seven2,Seven3])

        marker.points.extend([Four1,Four2])
        marker.points.extend([Four2,Four3])
        marker.points.extend([Four4,Four5])

        AA274_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
