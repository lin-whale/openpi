import copy
from threading import Lock

import arm_interfaces.msg
from custom_image_msg.msg import Image4m
from dual_arm_interfaces.msg import DualArmStatus
from dual_arm_interfaces.msg import TeleoperationJointCommand
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from PIL import Image
import time


class RosNode(Node):
    def __init__(self, action_lock):
        super().__init__("lerobot_node")
        # Gripper position when fully closed (float or int)
        self.gripper_position_close = 100  # master gripper
        # Gripper position when fully open (float or int)
        self.gripper_position_open = 0.0
        # Number of joints per single arm (int)
        self.joint_num_single_arm = 8
        # Temporal chunk size for inference (int)
        # self.chunk_size = chunk_size
        # Flag to indicate if this is the first frame (bool)
        self.first_frame = True
        # Lock for thread-safe observation access
        self.action_lock = action_lock
        self.obs_updated = False

        # Current observation: [images, joint_positions + joint_velocities]
        self.observation = None

        # QoS profile for ROS2 topic subscriptions
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ROS2 mode: subscribe to image and joint status topics
        self.get_logger().info("aloha infer node using ROS2 mode.")
        # Head camera image subscriber
        head_image_sub = Subscriber(self, Image4m, "/camera_dcw2/custom_cam_color", qos_profile=qos)
        # Right wrist camera image subscriber
        right_wrist_image_sub = Subscriber(self, Image4m, "/camera_dcl_right/custom_cam_color", qos_profile=qos)
        # Left wrist camera image subscriber
        left_wrist_image_sub = Subscriber(self, Image4m, "/camera_dcl_left/custom_cam_color", qos_profile=qos)
        # Joint status subscriber
        # slave_joint_sub = Subscriber(self, DualArmStatus, "/dual_arm_status")
        joint_sub = Subscriber(self, arm_interfaces.msg.ArmStatus, "/arm_status", qos_profile=qos)
        # Synchronize image and joint status topics
        self.subscriber = ApproximateTimeSynchronizer(
            [head_image_sub, right_wrist_image_sub, left_wrist_image_sub, joint_sub], 20, 0.05
        )
        self.subscriber.registerCallback(self._callback)

        # Publisher for teleoperation joint commands
        # self.joint_publisher = self.create_publisher(TeleoperationJointCommand, "/tele_joint_cmd_office", 1)
        self.joint_publisher = self.create_publisher(arm_interfaces.msg.MasterArmCommand, "/tele_arm_cmd", 1)

        self.action_chunk = []

        self.fps = 18
        self.publish_timer = self.create_timer(1 / self.fps, self.publish_action)       # 从臂动作发送定时器

        self.start_time = time.time()

    def publish_action(self):
        """
        Publish actions at a fixed rate defined by self.fps.
        If enough actions are accumulated, send the next action to the robot arms.
        """
        if self.action_chunk:
            with self.action_lock:
                action = self.action_chunk.pop(0)
            self.send_goal(action)
            self._logger.info(f"Published action to robot arms: {action}")
        else:
            print("No action to publish. exit.")

    def _callback(
        self, head_msg: Image4m, right_wrist_msg: Image4m, left_wrist_msg: Image4m, slave_joint_msg: DualArmStatus
    ):
        """
        Callback for synchronized ROS2 topics (images and joint status).
        Receives image and joint messages, runs inference, and publishes control commands.

        Args:
            head_msg (Image4m): Head camera image message.
            right_wrist_msg (Image4m): Right wrist camera image message.
            left_wrist_msg (Image4m): Left wrist camera image message.
            slave_joint_msg (DualArmStatus): Dual arm joint status message.
        """
        self.get_logger().info("enter to callback")
        # Convert ROS image messages to numpy arrays
        size = head_msg.height * head_msg.width * 3
        head_image = np.frombuffer(head_msg.data[:size], dtype=np.uint8).reshape((head_msg.height, head_msg.width, 3))
        size = right_wrist_msg.height * right_wrist_msg.width * 3
        right_image = np.frombuffer(right_wrist_msg.data[:size], dtype=np.uint8).reshape(
            (right_wrist_msg.height, right_wrist_msg.width, 3)
        )
        size = left_wrist_msg.height * left_wrist_msg.width * 3
        left_image = np.frombuffer(left_wrist_msg.data[:size], dtype=np.uint8).reshape(
            (left_wrist_msg.height, left_wrist_msg.width, 3)
        )
        # Convert joint status message to list
        left_joint, right_joint = self._joints_msg_to_list(slave_joint_msg)
        if left_joint is None or right_joint is None:
            self.get_logger().error("ERROR: get joint error")
            return

        transposed_head_img = np.transpose(head_image, (2, 0, 1))
        transposed_left_img = np.transpose(left_image, (2, 0, 1))
        transposed_right_img = np.transpose(right_image, (2, 0, 1))

        # 保存图片验证。
        # saveable_img = np.transpose(transposed_head_img, (1, 2, 0))
        # Image.fromarray(saveable_img).save("transposed_image.jpg")

        self.observation = {
            "state": np.array(left_joint + right_joint, dtype=np.float64),
            "images": {
                "cam_high": transposed_head_img,
                "cam_left_wrist": transposed_left_img,
                "cam_right_wrist": transposed_right_img,
            },
            "prompt": "Fold the shorts on the bed.",
        }
        self.obs_updated = True
        self.get_logger().info(f"Observation update time: {time.time() - self.start_time}")
        self.start_time = time.time()
        # Run inference to get action
        # action = self.alohaInfer.predict([head_image, right_image, left_image], slave_joint)

    def send_goal(self, goal_pos):
        """
        Send goal positions to the robot arms.

        Args:
            goal_pos (dict): Dictionary with joint names as keys and target positions as values.
        """
        goal_msg = arm_interfaces.msg.MasterArmCommand()
        # Fill in joint commands for both arms
        for i, pos in enumerate(goal_pos[:7]):
            goal_msg.left_command[i] = pos
        for i, pos in enumerate(goal_pos[8:15]):
            goal_msg.right_command[i] = pos
        # Denormalize the gripper position for both arms
        goal_msg.int_command[0] = goal_pos[7]
        goal_msg.int_command[1] = goal_pos[15]
        # 左右手使能
        goal_msg.left_button[9] = True
        goal_msg.right_button[9] = True

        # Publish the command message
        self.joint_publisher.publish(goal_msg)


    def _joints_msg_to_list(self, joint_msg: arm_interfaces.msg.ArmStatus):
        """
        Convert a ArmStatus message to lists of joint positions and velocities.
        ArmStatus:
            std_msgs/Header header
            arm_interfaces/JointStatus[7] left_arm
            arm_interfaces/JointStatus[7] right_arm
            arm_interfaces/JointStatus[7] gripper
            geometry_msgs/Pose left_arm_pose
            geometry_msgs/Pose right_arm_pose
            uint8[20] other_status

        Args:
            joint_msg (DualArmStatus): Dual arm joint status message.

        Returns:
            tuple: (joint_positions, joint_velocities)
                joint_positions (list of float): Concatenated right and left arm joint positions (including grippers).
                joint_velocities (list of float): Concatenated right and left arm joint velocities (including grippers).
                Returns (None, None) if error_code == 1.
        """
        # if joint_msg.error_code == 1:
        #     return None, None
        # else:
        left_joint_pos = [joint_status.joint_status[0] for joint_status in joint_msg.left_arm[:7]]
        left_gripper_pos = joint_msg.gripper[0].joint_status[0]
        left_joint_pos.append(left_gripper_pos)

        right_joint_pos = [joint_status.joint_status[0] for joint_status in joint_msg.right_arm[:7]]
        right_gripper_pos = joint_msg.gripper[1].joint_status[0]
        right_joint_pos.append(right_gripper_pos)

        return left_joint_pos, right_joint_pos  # joint velocities not available

        # # Right arm joint positions and velocities
        # right_joint_pos = [joint.joint_position for joint in joint_msg.right_arm_joints]
        # right_joint_vel = [joint.joint_speed for joint in joint_msg.right_arm_joints]
        # right_gripper_pos = joint_msg.right_gripper.gripper_position
        # # Normalize gripper position to [0, 1]
        # right_gripper_pos = (right_gripper_pos - self.gripper_position_close) / (
        #         self.gripper_position_open - self.gripper_position_close)
        # right_joint_pos.append(right_gripper_pos)
        # right_joint_vel.append(joint_msg.right_gripper.gripper_speed)
        # right_joint_vel = self.rpm2speed(right_joint_vel)

        # # Left arm joint positions and velocities
        # left_joint_pos = [joint.joint_position for joint in joint_msg.left_arm_joints]
        # left_joint_vel = [joint.joint_speed for joint in joint_msg.left_arm_joints]
        # left_gripper_pos = joint_msg.left_gripper.gripper_position
        # # Normalize gripper position to [0, 1]
        # left_gripper_pos = (left_gripper_pos - self.gripper_position_close) / (
        #         self.gripper_position_open - self.gripper_position_close)
        # left_joint_pos.append(left_gripper_pos)
        # left_joint_vel.append(joint_msg.left_gripper.gripper_speed)
        # left_joint_vel = self.rpm2speed(left_joint_vel)

        # return right_joint_pos + left_joint_pos, right_joint_vel + left_joint_vel
