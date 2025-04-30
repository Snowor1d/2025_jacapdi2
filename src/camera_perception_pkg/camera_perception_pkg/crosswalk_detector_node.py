import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from interfaces_pkg.msg import DetectionArray, Detection
from std_msgs.msg import Float32MultiArray

class CrosswalkDetector(Node):
    def __init__(self):
        super().__init__('crosswalk_detector_node')

        # 토픽 이름 파라미터
        self.sub_detection_topic = self.declare_parameter(
            'sub_detection_topic', 'detections'
        ).value
        self.pub_topic = self.declare_parameter(
            'pub_topic', 'crosswalk_detections'
        ).value
        self.size_pub_topic = self.declare_parameter(
            'size_pub_topic', 'crosswalk_bbox_sizes'
        ).value

        # QoS 설정
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # ★ DetectionArray만 구독
        self.det_sub = self.create_subscription(
            DetectionArray,
            self.sub_detection_topic,
            self.detections_callback,
            qos
        )

        # 횡단보도 검출 결과만 다시 퍼블리시
        self.det_pub = self.create_publisher(
            DetectionArray,
            self.pub_topic,
            qos
        )
        # 바운딩박스 크기(폭, 높이) 정보 퍼블리시
        self.size_pub = self.create_publisher(
            Float32MultiArray,
            self.size_pub_topic,
            qos
        )

        self.get_logger().info(f"[CrosswalkDetector] Subscribing `{self.sub_detection_topic}`, "
                               f"publishing filtered on `{self.pub_topic}` "
                               f"and sizes on `{self.size_pub_topic}`")

    def detections_callback(self, msg: DetectionArray):
        # 1) 횡단보도 검출만 필터링
        crosswalks = [
            d for d in msg.detections
            if d.class_name.lower() == 'crosswalk'
        ]

        # 2) DetectionArray 형식으로 퍼블리시
        out_det = DetectionArray()
        out_det.header = msg.header
        out_det.detections = crosswalks
        self.det_pub.publish(out_det)

        # 3) 바운딩박스 크기 정보(Float32MultiArray) 생성
        size_msg = Float32MultiArray()
        # [w1, h1, w2, h2, ...]
        for d in crosswalks:
            w = float(d.bbox.size.x)
            h = float(d.bbox.size.y)
            size_msg.data.extend([w, h])
            # 또는 로그로도 출력 가능
            self.get_logger().info(f"Crosswalk bbox size → width: {w:.1f}, height: {h:.1f}")

        self.size_pub.publish(size_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CrosswalkDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
