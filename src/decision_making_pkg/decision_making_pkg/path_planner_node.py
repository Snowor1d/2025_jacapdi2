import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult, DetectionArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

#---------------Variable Setting---------------
#SUB_LANE_TOPIC_NAME = "yolov8_lane_info"  # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점 토픽
PUB_TOPIC_NAME = "path_planning_result"   # 경로 계획 결과 퍼블리시 토픽
CAR_CENTER_POINT = (320, 179) # 이미지 상에서 차량 앞 범퍼의 중심이 위치한 픽셀 좌표

#----------------------------------------------
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # 파라미터 선언
        
       # self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point = self.declare_parameter('car_center_point', CAR_CENTER_POINT).value
        self.selected_lane = 'lane2'
        
        self.cooldown_period = 500
        self.size_threshold_1_to_2 = 135
        self.size_threshold_2_to_1 = 140
        self.cooldown_counter = 0
        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.target_points = []  # 차선의 타겟 지점들 (차선 중앙)

        # 서브스크라이버 설정 (타겟 지점 구독)
        self.sub1 = self.create_subscription(
            LaneInfo, 'yolov8_lane1_info', lambda msg: self.lane_callback(msg, 'lane1'), self.qos_profile)
        self.sub2 = self.create_subscription(
            LaneInfo, 'yolov8_lane2_info', lambda msg: self.lane_callback(msg, 'lane2'), self.qos_profile)
        # 퍼블리셔 설정 (경로 계획 결과 퍼블리시)
        
        self.crosswalk_sub = self.create_subscription(
            DetectionArray, 'detections', self.detections_callback, self.qos_profile)
        
        
        self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

    def lane_callback(self, msg: LaneInfo, lane_id: str):
        if (lane_id != self.selected_lane):
            return
        
        # 타겟 지점 받아오기
        self.target_points = msg.target_points
        
        # 타겟 지점이 3개 이상 모이면 경로 계획 시작
        if len(self.target_points) >= 3:
            self.plan_path()
    def detections_callback(self, msg: DetectionArray):

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return


        large_traffic_lights = []
        for d in msg.detections:
            # if d.class_name.lower() == 'crosswalk':
            #     w = d.bbox.size.x
            #     h = d.bbox.size.y
            #     if(self.selected_lane == 'lane1' and w >= self.size_threshold_1_to_2) or \
            #        (self.selected_lane == 'lane2' and w >= self.size_threshold_2_to_1):
            #         sizes.append([w, h])
            if d.class_name.lower() == 'traffic_light':
                w = d.bbox.size.x
                #self.get_logger().info(f"w: {w}")
                h = d.bbox.size.y
                if(self.selected_lane == 'lane1' and w >= self.size_threshold_1_to_2) or \
                   (self.selected_lane == 'lane2' and w >= self.size_threshold_2_to_1):
                    large_traffic_lights.append([w, h])
        
        if not large_traffic_lights:
            return
        self.get_logger().info(f"🚦 Large traffic light detected (w,h ≥ {self.size_threshold_1_to_2})")
        
        self.handle_trafficlight()
        self.cooldown_counter = self.cooldown_period
        
        
            
    # def crosswalk_callback(self, msg: DetectionArray):
    #     # 1) 이미 쿨다운 중이면 카운터만 줄이고 리턴
    #     if self.cooldown_counter > 0:
    #         self.cooldown_counter -= 1
    #         return

    #     # 2) 바운딩박스 크기가 threshold 이상인 crosswalk만 필터
    #     large_crosswalks = []
    #     for d in msg.detections:
    #         if d.class_name.lower() == 'crosswalk':
    #             w = d.bbox.size.x
    #             h = d.bbox.size.y
    #             if(self.selected_lane == 'lane1' and w >= self.size_threshold_1_to_2) or \
    #                (self.selected_lane == 'lane2' and w >= self.size_threshold_2_to_1):
    #                 if(self.selected_lane == 'lane1'):
    #                     time.sleep(0.4)
    #                 large_crosswalks.append(d)

    #     if not large_crosswalks:
    #         return

    #     # 3) 교차로 감지 & 쿨다운 없음 → 차선 전환
    #     #self.get_logger().info(f"🚸 Large crosswalk detected (w,h ≥ {self.size_threshold})")
    #     self.handle_crosswalk()
    #     self.cooldown_counter = self.cooldown_period

    # def handle_crosswalk(self):
    #     # lane toggle
    #     old = self.selected_lane
    #     self.selected_lane = 'lane1' if old == 'lane2' else 'lane2'
    #     self.get_logger().info(f"🚗 Switching from {old} → {self.selected_lane}")
        

    def handle_trafficlight(self):
        # lane toggle
        old = self.selected_lane
        self.selected_lane = 'lane1' if old == 'lane2' else 'lane2'
        self.get_logger().info(f"🚗 Switching from {old} → {self.selected_lane}")



    def plan_path(self):
        # self.target_points가 TargetPoint 객체들의 리스트라고 가정
        if not self.target_points:
            self.get_logger().warn("No target points available")
            return
        
        # TargetPoint 객체에서 x, y 값 추출
        x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points])

        #차량 앞 범퍼의 중심이 위치한 픽셀 좌표 추가
        y_points_list, x_points_list = list(y_points), list(x_points) 
        y_points_list.append(self.car_center_point[1])
        x_points_list.append(self.car_center_point[0])
        y_points, x_points = tuple(y_points_list), tuple(x_points_list)
        
        # y 값을 기준으로 정렬 (y가 증가하는 순서로 정렬)
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

        # 정렬된 y, x 값을 다시 분리
        y_points, x_points = zip(*sorted_points)
        
        # 몇개의 점으로 경로 계획을 하는지 확인
        #self.get_logger().info(f"Planning path with {len(y_points)} points")

        # 스플라인 보간법을 사용하여 경로 생성
        cs = CubicSpline(y_points, x_points, bc_type='natural')

        # 생성된 경로 점들 (추가적인 점들을 생성하여 부드러운 경로를 얻음)
        y_new = np.linspace(min(y_points), max(y_points), 100)
        x_new = cs(y_new)

        # 경로를 따라가는 정보 (PathPlanningResult 메시지로 발행)
        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)

        # 경로 퍼블리시
        self.publisher.publish(path_msg)

        # 타겟 지점 초기화 (다음 경로 계산을 위해)
        self.target_points.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
