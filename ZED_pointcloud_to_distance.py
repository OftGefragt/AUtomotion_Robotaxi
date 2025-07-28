'''
Zed ROS2 datası referanstır 

/zed/zed_node/point_cloud/cloud_registered 




"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import math

class ZEDPointCloudReader(Node):
    def __init__(self):
        super().__init__('zed_pointcloud_reader')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10
        )

        # You can change these to the center of any detected object (from YOLO)
        self.u = 640  # center pixel X (for 1280 width)
        self.v = 360  # center pixel Y (for 720 height)

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            # Ensure u and v are within the bounds of the image
            if self.u >= msg.width or self.v >= msg.height:
                self.get_logger().warn("Pixel out of bounds")
                return

            index = self.v * msg.row_step + self.u * msg.point_step

            # Read float32 x, y, z (each 4 bytes)
            x = struct.unpack_from('f', msg.data, index + 0)[0]
            y = struct.unpack_from('f', msg.data, index + 4)[0]
            z = struct.unpack_from('f', msg.data, index + 8)[0]

            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                self.get_logger().warn("Invalid point (NaN)")
                return

            distance = math.sqrt(x**2 + y**2 + z**2)
            self.get_logger().info(f"3D Point at (u={self.u}, v={self.v}) = ({x:.2f}, {y:.2f}, {z:.2f}) m | Distance = {distance:.2f} m")

        except Exception as e:
            self.get_logger().error(f"Error reading point cloud: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ZEDPointCloudReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



