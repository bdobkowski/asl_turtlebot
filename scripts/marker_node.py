#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int8
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
class MarkerRelay:
    def __init__(self):
        rospy.init_node('marker_node')
        self.pub = rospy.Publisher('/marker_topic', Marker, queue_size=10)
        self.sub = rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)
    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        # IMPORTANT: If youre creating multiple markers,
        # each need to have a separate marker ID.
        marker.id = 0
        marker.type = 2 # sphere
        marker.pose.position.x = data.x
        marker.pose.position.y = data.y
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0 # Dont forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.pub.publish(marker)
    def run(self):
        rospy.spin()
if __name__ == '__main__':
    mr = MarkerRelay()
    mr.run()
