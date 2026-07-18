FROM ros:humble

RUN apt-get update && \
    apt-get install -y \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-nav2-simple-commander \
        ros-humble-rviz2 \
        ros-humble-xacro \
        ros-humble-rosbag2-storage-mcap \
        python3-pip \
        tmux \
        tree \
        nano \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    opencv-python \
    scikit-learn \ 
    pyserial \
    "numpy<1.25" \
    "scipy<1.12"

WORKDIR /root/ws

COPY src ./src

ENV ROS_DISTRO=humble

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]