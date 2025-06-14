import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64, String

class ModelPerformanceSubscriber(Node):
    def __init__(self):
        super().__init__('model_performance_subscriber')
        self.subscription = self.create_subscription(String, 'model_metrics', self.listener_callback, 10)

    def listener_callback(self, msg):
        self.get_logger().info(f"Model Performance Metrics: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = ModelPerformanceSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
