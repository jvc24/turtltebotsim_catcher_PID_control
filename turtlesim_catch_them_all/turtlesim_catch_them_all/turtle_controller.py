#!/usr/bin/env python3
import math
import rclpy
from functools import partial
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from my_robot_interfaces.msg import Turtle
from my_robot_interfaces.msg import TurtleArray
from my_robot_interfaces.srv import CatchTurtle

class TurtleControllerNode(Node):
    def __init__(self):
        super().__init__("turtle_controller")
        
        self.declare_parameter("closet_turtle_first",True)
        self.closet_turtle_first_ = self.get_parameter("closet_turtle_first").value
        self.turtle_to_catch_: Turtle = None
        self.pose_: Pose = None


        #PID Control Parameters 
        # Linear PID gains
        self.kp_linear = 2.0
        self.ki_linear = 0.0
        self.kd_linear = 0.2

        # Angular PID gains
        self.kp_angular = 6.0
        self.ki_angular = 0.0
        self.kd_angular = 0.3

        # Error memory
        self.prev_distance_error = 0.0
        self.prev_angle_error = 0.0

        # Integral terms
        self.distance_integral = 0.0
        self.angle_integral = 0.0

        # Control timestep
        self.dt = 0.01

        self.cmd_vel_publisher_ = self.create_publisher(
            Twist, "/turtle1/cmd_vel", 10)
        self.pose_subscriber_ = self.create_subscription(
            Pose, "/turtle1/pose", self.callback_pose, 10)
        self.alive_turtles_subscriber_ = self.create_subscription(
            TurtleArray, "alive_turtles", self.callback_alive_turtles, 10)
        self.catch_turtle_client_ = self.create_client(CatchTurtle, "catch_turtle")
        self.control_loop_timer_ = self.create_timer(
            0.01, self.control_loop)

    def callback_pose(self, pose: Pose):
        self.pose_ = pose

    def callback_alive_turtles(self, msg: TurtleArray):
        if len(msg.turtles) > 0:
            if self.closet_turtle_first_:
                closet_turtle = None
                closet_turtle_distance = None

                for turtle in msg.turtles:
                    dist_x = turtle.x - self.pose_.x
                    dist_y = turtle.y - self.pose_.y
                    
                    distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)

                    if closet_turtle == None or distance < closet_turtle_distance: 
                        closet_turtle = turtle
                        closet_turtle_distance = distance
                
                self.turtle_to_catch_ = closet_turtle

            else:
                self.turtle_to_catch_ = msg.turtles[0]

    def control_loop(self):

        if self.pose_ is None or self.turtle_to_catch_ is None:
            return

        dist_x = self.turtle_to_catch_.x - self.pose_.x
        dist_y = self.turtle_to_catch_.y - self.pose_.y

        distance_error = math.sqrt(
            dist_x * dist_x +
            dist_y * dist_y
        )

        goal_theta = math.atan2(dist_y, dist_x)

        angle_error = goal_theta - self.pose_.theta

        # Normalize angle
        if angle_error > math.pi:
            angle_error -= 2 * math.pi
        elif angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd = Twist()

        if distance_error > 0.5:

            # ----------------------------
            # LINEAR PID
            # ----------------------------

            self.distance_integral += distance_error * self.dt

            distance_derivative = (
                distance_error - self.prev_distance_error
            ) / self.dt

            cmd.linear.x = (
                self.kp_linear * distance_error +
                self.ki_linear * self.distance_integral +
                self.kd_linear * distance_derivative
            )

            # ----------------------------
            # ANGULAR PID
            # ----------------------------

            self.angle_integral += angle_error * self.dt

            angle_derivative = (
                angle_error - self.prev_angle_error
            ) / self.dt

            cmd.angular.z = (
                self.kp_angular * angle_error +
                self.ki_angular * self.angle_integral +
                self.kd_angular * angle_derivative
            )

            # Save previous errors
            self.prev_distance_error = distance_error
            self.prev_angle_error = angle_error

            # Optional velocity limits
            cmd.linear.x = min(cmd.linear.x, 3.0)

            cmd.angular.z = max(
                min(cmd.angular.z, 6.0),
                -6.0
            )

        else:

            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            self.call_catch_turtle_service(
                self.turtle_to_catch_.name
            )

            self.turtle_to_catch_ = None

            # Reset PID memory
            self.distance_integral = 0.0
            self.angle_integral = 0.0

        self.cmd_vel_publisher_.publish(cmd)

    def call_catch_turtle_service(self, turtle_name):
        while not self.catch_turtle_client_.wait_for_service(1.0):
            self.get_logger().warn("Waiting for catch turtle service...")
        
        request = CatchTurtle.Request()
        request.name = turtle_name

        future = self.catch_turtle_client_.call_async(request)
        future.add_done_callback(
            partial(self.callback_call_catch_turtle_service, turtle_name=turtle_name))

    def callback_call_catch_turtle_service(self, future, turtle_name):
        response: CatchTurtle.Response = future.result()
        if not response.success:
            self.get_logger().error("Turtle " + turtle_name + " could not be removed")

def main(args=None):
    rclpy.init(args=args)
    node = TurtleControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
