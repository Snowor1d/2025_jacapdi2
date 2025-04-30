import cv2
import numpy as np
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, Detection


class CrosswalkDetector(Node):
    def __init__(self):
        super().__init__('crosswalk_detector_node')

        # 1) 파라미터 선언
        self.sub_detection_topic = self.declare_parameter(
            'sub_detection_topic', 'detections'
        ).value
        self.sub_image_topic = self.declare_parameter(
            'sub_image_topic', 'image_raw'
        ).value
        self.pub_topic = self.declare_parameter(
            'pub_topic', 'crosswalk_detections'
        ).value

        # 2) CvBridge 및 QoS 설정
        self.bridge = CvBridge()
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 3) message_filters 를 이용한 동기화 서브스크라이버
        self.det_sub = Subscriber(self, DetectionArray, self.sub_detection_topic, qos_profile=qos)
        self.img_sub = Subscriber(self, Image, self.sub_image_topic, qos_profile=qos)
        self.ts = ApproximateTimeSynchronizer(
            [self.det_sub, self.img_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # 4) crosswalk 정보만 담아 퍼블리시할 퍼블리셔
        self.pub = self.create_publisher(DetectionArray, self.pub_topic, qos)

        self.get_logger().info(f"Listening to `{self.sub_detection_topic}` + `{self.sub_image_topic}`, "
                               f"publishing crosswalks on `{self.pub_topic}`")

    def sync_callback(self, det_msg: DetectionArray, img_msg: Image):
        # OpenCV 이미지로 변환 (필요 시 ROI 처리 등에 사용 가능)
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # crosswalk Detection만 필터링
        crosswalk_dets: List[Detection] = [
            d for d in det_msg.detections
            if d.class_name.lower() == 'crosswalk'
        ]

        if not crosswalk_dets:
            # 횡단보도가 하나도 없으면 빈 DetectionArray 퍼블리시
            empty_msg = DetectionArray()
            empty_msg.header = det_msg.header
            self.pub.publish(empty_msg)
            return

        # 횡단보도만 담은 새로운 메시지 생성 및 퍼블리시
        out_msg = DetectionArray()
        out_msg.header = det_msg.header
        out_msg.detections = crosswalk_dets
        self.pub.publish(out_msg)

        # (선택) 디버그용 윈도우로 바운딩박스 그려보기
        for det in crosswalk_dets:
            bb = det.bbox  # BoundingBox2D 타입
            p1 = (int(bb.xmin), int(bb.ymin))
            p2 = (int(bb.xmax), int(bb.ymax))
            cv2.rectangle(cv_img, p1, p2, (0,255,0), 2)
        cv2.imshow('crosswalks', cv_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CrosswalkDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
