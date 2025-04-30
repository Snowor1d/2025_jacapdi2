import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.lane_name = self.declare_parameter('lane_name', 'lane1').value
        self.pub_topic = f'yolov8_{self.lane_name}_info'
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # 검출 결과 구독자
        self.subscriber = self.create_subscription(
            DetectionArray, self.sub_topic,
            self.yolov8_detections_callback,
            self.qos_profile
        )

        # 차선별 퍼블리셔 생성 (lane1, lane2)
        self.lane_names = ['lane1', 'lane2']
        self.lane_publishers = {}
        for ln in self.lane_names:
            topic = f'yolov8_{ln}_info'
            self.lane_publishers[ln] = self.create_publisher(LaneInfo, topic, self.qos_profile)

        # ROI 이미지 퍼블리셔
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return
        # lane2_edge_image = CPFL.draw_edges(detection_msg, cls_name=self.lane_name, color=255)

        # (h, w) = (lane2_edge_image.shape[0], lane2_edge_image.shape[1]) #(480, 640)
        # dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        #src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]
        
        # lane2_bird_image = CPFL.bird_convert(lane2_edge_image, srcmat=src_mat, dstmat=dst_mat)
        # roi_image = CPFL.roi_rectangle_below(lane2_bird_image, cutting_idx=300)

        # if self.show_image:
        #     cv2.imshow('lane2_edge_image', lane2_edge_image)
        #     cv2.imshow('lane2_bird_img', lane2_bird_image)
        #     cv2.imshow('roi_img', roi_image)
        #     cv2.waitKey(1)

        # # roi_image를 uint8 형식으로 변환
        # roi_image = cv2.convertScaleAbs(roi_image)  # 64FC1 -> uint8로 변환

        # # roi_image를 ROS Image 메시지로 변환
        # try:
        #     roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
        #     # ROI 이미지를 퍼블리시
        #     self.roi_image_publisher.publish(roi_image_msg)
        # except Exception as e:
        #     self.get_logger().error(f"Failed to convert and publish ROI image: {e}")
        
        # grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
                
        # target_points = []
        # for target_point_y in range(5, 155, 50):  # 예시로 5에서 155까지 50씩 증가
        #     target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
        #                                         detection_thickness=10, road_gradient=grad, lane_width=300)
            
        #     target_point = TargetPoint()
        #     target_point.target_x = round(target_point_x)
        #     target_point.target_y = round(target_point_y)
        #     target_points.append(target_point)

        # lane = LaneInfo()
        # lane.lane_id = self.lane_name
        # lane.slope = grad
        # lane.target_points = target_points

        # self.publisher.publish(lane)
        src_mat = [[238, 316],[402, 313], [501, 476], [155, 476]]
        for ln in self.lane_names:
            # 선택한 차선 엣지 영상 생성
            edge_img = CPFL.draw_edges(detection_msg, cls_name=ln, color=255)
            h, w = edge_img.shape[:2]
            # 목적 매트릭스 동적 생성
            dst_mat = [
                [round(w * 0.3), 0],
                [round(w * 0.7), 0],
                [round(w * 0.7), h],
                [round(w * 0.3), h]
            ]

            # 버드 아이 뷰 변환 및 ROI 추출
            bird_img = CPFL.bird_convert(edge_img, srcmat=src_mat, dstmat=dst_mat)
            roi_img = CPFL.roi_rectangle_below(bird_img, cutting_idx=300)

            # 디버그용 윈도우
            if SHOW_IMAGE:
                cv2.imshow(f'{ln}_edge_image', edge_img)
                cv2.imshow(f'{ln}_bird_img', bird_img)
                cv2.imshow(f'{ln}_roi_img', roi_img)
                cv2.waitKey(1)

            # ROI를 uint8로 변환 후 퍼블리시
            roi_uint8 = cv2.convertScaleAbs(roi_img)
            try:
                roi_msg = self.cv_bridge.cv2_to_imgmsg(roi_uint8, encoding="mono8")
                self.roi_image_publisher.publish(roi_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish ROI image for {ln}: {e}")

            # 차선 기울기 및 타겟 포인트 계산
            grad = CPFL.dominant_gradient(roi_uint8, theta_limit=70)
            target_points = []
            for y in range(5, 155, 50):
                x = CPFL.get_lane_center(
                    roi_uint8,
                    detection_height=y,
                    detection_thickness=10,
                    road_gradient=grad,
                    lane_width=300
                )
                tp = TargetPoint()
                tp.target_x = round(x)
                tp.target_y = round(y)
                target_points.append(tp)

            # LaneInfo 메시지 생성 및 퍼블리시
            lane_msg = LaneInfo()
            lane_msg.lane_id = ln
            lane_msg.slope = grad
            lane_msg.target_points = target_points
            self.lane_publishers[ln].publish(lane_msg)



def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()
