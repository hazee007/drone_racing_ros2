#### Build this package
    git clone https://github.com/crcz25/aerial_multi_robot_project.git
    cd aerial_multi_robot_project
    source /opt/ros/galactic/setup.bash
    colcon build



#### Run a teleop simulation
    cd PATH_TO_REPO
    source install/setup.bash
    export GAZEBO_MODEL_PATH=${PWD}/install/tello_gazebo/share/tello_gazebo/models
    source /usr/share/gazebo/setup.sh
    


#### Run case-1
    ros2 launch tello_gazebo case_1_track_launch.py


#### Open another terminal and run:
    source /opt/ros/galactic/setup.bash && cd path/to/drone_race && python3 main.py


#### Run case-2
    ros2 launch tello_gazebo case_2_track_launch.py


#### Open another terminal and run:
    source /opt/ros/galactic/setup.bash && cd path/to/drone_race && python3 main.py


#### Run case-3
    ros2 launch tello_gazebo case_3_track_height_launch.py


#### Open another terminal and run:
    source /opt/ros/galactic/setup.bash && cd path/to/drone_race && python3 main.py


#### Run random track simple
    ros2 launch tello_gazebo random_track_launch.py


#### Open another terminal and run:
    source /opt/ros/galactic/setup.bash && cd path/to/drone_race && python3 main.py


#### Run random track rotated
    ros2 launch tello_gazebo random_track_launch.py


#### Open another terminal and run:
    source /opt/ros/galactic/setup.bash && cd path/to/drone_race && python3 main.py
