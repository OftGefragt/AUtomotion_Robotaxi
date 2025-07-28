'''

'''



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import math

# Import your YOLO detections message here
# For example, if your custom message is `MyYoloDetections`
# from my_yolo_package.msg import MyYoloDetections

class YoloZedDepthMatcher(Node):
    def __init__(self):
        super().__init__('yolo_zed_depth_matcher')

        self.pointcloud_msg = None
        self.detections_msg = None

        # Subscribe to ZED point cloud
        self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10
        )

        # Subscribe to your YOLO detections topic and message type
        self.create_subscription(
            YourYoloDetectionsMsgType,
            '/your_yolo_topic',
            self.detections_callback,
            10
        )

        self.create_timer(0.1, self.process_detections)  # 10 Hz processing

    def pointcloud_callback(self, msg):
        self.pointcloud_msg = msg

    def detections_callback(self, msg):
        self.detections_msg = msg

    def get_xyz_from_pointcloud(self, msg, u, v):
        u = int(u)
        v = int(v)

        if not msg or u >= msg.width or v >= msg.height or u < 0 or v < 0:
            return None

        index = v * msg.row_step + u * msg.point_step
        try:
            x = struct.unpack_from('f', msg.data, index)[0]
            y = struct.unpack_from('f', msg.data, index + 4)[0]
            z = struct.unpack_from('f', msg.data, index + 8)[0]
        except Exception as e:
            self.get_logger().warn(f"Error unpacking point cloud data: {e}")
            return None

        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return None

        return (x, y, z)

    def process_detections(self):
        if self.pointcloud_msg is None or self.detections_msg is None:
            return

        # Loop over detections, adapt fields to your message format!
        for detection in self.detections_msg.detections:  
            # Example fields — change according to your message:
            xmin = detection.bbox.xmin
            ymin = detection.bbox.ymin
            xmax = detection.bbox.xmax
            ymax = detection.bbox.ymax
            label = detection.label

            u = (xmin + xmax) / 2
            v = (ymin + ymax) / 2

            xyz = self.get_xyz_from_pointcloud(self.pointcloud_msg, u, v)
            if xyz is None:
                self.get_logger().warn(f"Invalid depth at pixel ({u},{v}) for object '{label}'")
                continue

            x, y, z = xyz
            distance = math.sqrt(x*x + y*y + z*z)
            self.get_logger().info(f"Object '{label}': Distance = {distance:.2f} meters")

def main(args=None):
    rclpy.init(args=args)
    node = YoloZedDepthMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

