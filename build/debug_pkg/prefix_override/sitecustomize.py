import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/amrl_sunny/ros2_autonomous_vehicle_simulation/install/debug_pkg'
