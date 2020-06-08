#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range, Image
from geometry_msgs.msg import Pose, Twist, Vector3
from tf.transformations import euler_from_quaternion
import numpy as np
import pdb
from math import pow, atan2, sqrt
import random
import time
from rospy.numpy_msg import numpy_msg
import time
import uuid
import os.path
from os import path
import thread
import os
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import cv2

class ThymioController:

    def __init__(self):
        """Initialization."""

        # Variable to store the current range of sensors 
        # Initializing to 10 to avoid interference with collision detection
        self.left_sensor = 10
        self.center_left_sensor = 10
        self.center_sensor = 10
        self.center_right_sensor = 10
        self.right_sensor = 10

        self.collision_tol = .02
        ## changes
        self.vX = np.array([0,1,2,5,6,7])
        self.vY = np.array([0,-1,-3,-5,-4])

        self.collided = False # Flag to stop the robot to avoid obstacles

        # initialize the node
        rospy.init_node(
        'thymio_controller', # name of the node
        anonymous = True  
        )

        self.name = rospy.get_param('~robot_name')

        # log robot name to console
        rospy.loginfo('Controlling %s' % self.name)

        # create velocity publisher
        self.velocity_publisher = rospy.Publisher(
        self.name + '/cmd_vel',  # name of the topic
        Twist,  # message type
        queue_size=10  # queue size
        )

        ############################################################
        # Sensor Subscribers
        self.proximity_subscriber_right = rospy.Subscriber(
        self.name + '/proximity/right',
        Range,
        self.log_sensor_right
        )
        self.proximity_subscriber_center_right = rospy.Subscriber(
        self.name + '/proximity/center_right',
        Range,
        self.log_sensor_center_right
        )
        self.proximity_subscriber_center = rospy.Subscriber(
        self.name + '/proximity/center',
        Range,
        self.log_sensor_center
        )
        self.proximity_subscriber_center_left = rospy.Subscriber(
        self.name + '/proximity/center_left',
        Range,
        self.log_sensor_center_left
        )
        self.proximity_subscriber_left = rospy.Subscriber(
        self.name + '/proximity/left',
        Range,
        self.log_sensor_left
        )

        self.image_subscriber = rospy.Subscriber(
            self.name + '/camera/image_raw',
            numpy_msg(Image),
            self.process_image
        )

        self.image_save_frequency = 1.5 # seconds
        self.collision_save_frequency = 0.2
        self.image_count = 0
        self.image = []
        self.dataset_images = []
        self.dataset_labels = []


        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # initialiself.first_runze pose to (X=0, Y=0, theta=0)
        self.pose = Pose()

        # initialize linear and angular velocities to 0
        self.velocity = Twist()

        # set node update frequency in Hz
        self.rate = rospy.Rate(10)


    def log_sensor_right(self, data):
        """Subscriber callback for robot right sensor data"""
        self.right_sensor = data.range

    def log_sensor_center_right(self, data):
        """Subscriber callback for robot right sensor data"""
        self.center_right_sensor = data.range

    def log_sensor_center(self, data):
        """Subscriber callback for robot right sensor data"""
        self.center_sensor = data.range


    def log_sensor_center_left(self, data):
        """Subscriber callback for robot right sensor data"""
        self.center_left_sensor = data.range

    def log_sensor_left(self, data):
        """Subscriber callback for robot right sensor data"""
        self.left_sensor = data.range


    def create_dataset(self):
        np.save("dataset/" + "dataset_images" + ".npy", self.dataset_images)
        np.save("dataset/" + "dataset_labels" + ".npy", self.dataset_labels)
        print("*"*25)
        print("Data collection complete")
        print("*"*25)

    def save_image(self, total_images=4000):
        # Waiting for image data
        while True:
            if(len(self.image)>0):
                break
            time.sleep(self.image_save_frequency)

        while self.image_count<=total_images:
            if self.image_count%100 == 0:
                print("Saved {} images".format(self.image_count))

            sleep_duration = self.image_save_frequency
            desired_size = (100,100)
            current_image = cv2.resize(self.image, dsize=desired_size, interpolation=cv2.INTER_CUBIC)
                        
            sensor_values = [self.left_sensor, 
                            self.center_left_sensor, 
                            self.center_sensor, 
                            self.center_right_sensor, 
                            self.right_sensor]
            angular_velocity = np.dot(sensor_values, [1,2,0,-2,-1])

            if min(sensor_values) < 0.7:
                sleep_duration = self.collision_save_frequency

            self.dataset_images.append(current_image)
            self.dataset_labels.append(angular_velocity)
            self.image_count += 1
            time.sleep(sleep_duration)
        self.create_dataset()


    def process_image(self, data):
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.image = np.array(im, dtype=np.float32)

        ## to respawn/teleport the robot if it crashes.
    def respawn(self,theta): # theta = [-pi/2, pi/2]
        # http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer
        w = np.cos(theta/2) 
        z = np.sin(theta/2)
        vMsg = ModelState()
        vMsg.model_name = self.name
        vMsg.model_name = vMsg.model_name[1:]
        x = np.random.choice(self.vX,1)
        y = np.random.choice(self.vY,1)
        vMsg.pose.position.x = float(x)
        vMsg.pose.position.y = float(y)
        vMsg.pose.position.z = 0

        vMsg.pose.orientation.x = 0
        vMsg.pose.orientation.y = 0
        vMsg.pose.orientation.z = z
        vMsg.pose.orientation.w = w
        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            vState = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            return vState(vMsg)
        except rospy.ServiceException, e:
            print(e)


    def get_control(self):
        angular_z = 0
        return Twist(
        linear=Vector3(
        .25,  # moves forward .25 m/s
        0,
        .0,
        ),
        angular=Vector3(
        .0,
        .0,
        angular_z
        )
        )

    def run(self):
        thread.start_new_thread(self.save_image, ())
        while not rospy.is_shutdown():
            if self.collided==True:
                q_theta = np.random.uniform(-1,1)*np.pi
                self.respawn(q_theta)
                self.collided=False


            while not self.collided:
                # decide control action
                velocity = self.get_control()
                # publish velocity message
                self.velocity_publisher.publish(velocity)
                
                vsensor = np.array([self.left_sensor, 
                            self.center_left_sensor, 
                            self.center_sensor, 
                            self.center_right_sensor, 
                            self.right_sensor])

                if min(vsensor)<self.collision_tol:
                    self.collided=True

                # sleep until next step
            self.rate.sleep()

 #           self.collided = False # Set the flag back
        return


    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
        Twist()  # set velocities to 0
        )

        self.rate.sleep()


if __name__ == '__main__':
    controller = ThymioController()

    try:
        controller.run()
    except rospy.ROSInterruptException as e:
        pass
