import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist, Pose2D, PoseStamped


def publisher(data):
    vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
    rospy.init_node('marker_node', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 0 # arrow

        marker.pose.position.x = data.x
        marker.pose.position.y = data.y
        marker.pose.position.z = 0

        marker.pose.orientation.x = np.cos(data.theta)
        marker.pose.orientation.y = np.sin(data.theta)
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        vis_pub.publish(marker)
        print('Published marker!')
        
        rate.sleep()


if __name__ == '__main__':
    try:
    	rospy.Subscriber('/cmd_nav', Pose2D, publisher)
    except rospy.ROSInterruptException:
        pass
