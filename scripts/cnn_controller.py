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
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
#        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.Linear(1296, 32)
        self.fc2 = nn.Linear(32, 1)
        self.drop = nn.Dropout(.4)
        self.drop2 = nn.Dropout(.4)
        
#        self.fc3 = nn.Linear(5, 1)
    def forward(self, x):
        # pdb.set_trace()
        x = self.drop(self.pool(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(self.pool(F.relu(self.conv3(x))))
        v_to = 1
        for i in range(1,len(x.shape)):
          v_to = v_to*x.shape[i]
        x = x.view(-1, v_to)
        #pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#        x = self.fc3(x)
        x = x
        # print("Output data of model is {}".format(x[0]))

        return x

class Cnn_model:
    def __init__(self):
        self.vFile = "/home/usi/Desktop/799_model.pt"


    def load_model(self):
        use_cuda = torch.cuda.is_available()
        DEVICE = torch.device('cuda' if use_cuda else 'cpu')   # 'cpu' in this case
        cpu_model = Net()
        cpu_model.load_state_dict(torch.load(self.vFile, map_location=DEVICE))
        return cpu_model


class ThymioController:

    def __init__(self):
        """Initialization."""
        self.file = open("prediction_log.txt", "a")
        self.left_sensor = 10
        self.center_left_sensor = 10
        self.center_sensor = 10
        self.center_right_sensor = 10
        self.right_sensor = 10

        self.collision_tol = .02

        self.vX = np.array([0,1,2,5,6,7])
        self.vY = np.array([0,-1,-3,-5,-4])

        self.collided = False # Flag to stop the robot to avoid obstacles

        # initialize the node
        rospy.init_node(
        'thymio_controller', # name of the node
        anonymous = True  
        )

        self.cnn_model = Cnn_model()
        self.model = self.cnn_model.load_model()
        self.model.eval()

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
        self.image = np.zeros((500,500,3), dtype = np.float32)


        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # set node update frequency in Hz
        self.rate = rospy.Rate(50)

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

    def cnn_controller(self):
        desired_size = (100,100)
        current_image = cv2.resize(self.image, dsize=desired_size, interpolation=cv2.INTER_CUBIC)
        current_image = current_image/255 # Normalization
        v_image = torch.from_numpy(current_image)
        vShape = v_image.size()
        v_image = v_image.reshape((1,vShape[2],vShape[0],vShape[1]))

        ## in pytorch unlike TF image has to be passed as (batch, channels, height, width) therefore reshaping.
        self.model.eval()
        angular_z = self.model(v_image)
        return angular_z

    def process_image(self, data):
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.image = np.array(im, dtype=np.float32)

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


    def get_control(self, linear_x=0.25):
        # to put a higher limit on total velocity
        sensor_values = [self.left_sensor, 
                            self.center_left_sensor, 
                            self.center_sensor, 
                            self.center_right_sensor, 
                            self.right_sensor]

        predicted_angular_z = self.cnn_controller().detach().numpy()
        self.file.write(predicted_angular_z)
        #predicted_angular_z = np.dot(predicted_sensors, np.array([1,2,0,-2,-1]))/3
        #computed_angular_z = np.dot(sensor_values, np.array([1,2,0,-2,-1]))/3
#        print(predicted_angular_z)
#        if abs(predicted_angular_z) < 0.1:
#            predicted_angular_z = 0 
        return Twist(
        linear=Vector3(
        linear_x,  # moves forward .2 m/s
        0,
        .0,
        ),
        angular=Vector3(
        .0,
        .0,
        predicted_angular_z*4
        )
        )

    def run(self):
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
