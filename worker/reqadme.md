Характеристики VM
Ubuntu 24.04.2 LTS, 4CPU 8Gi RAM

Настройка окружения:
```bash
  python3 -V
   sudo apt update && sudo apt upgrade
   # dependency 
   sudo apt install build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev v4l-utils \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    python3-numpy python3-pip unzip
    # install openCV library
    wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/heads/master.zip
    unzip opencv. zip
    cd opencv-master/
      mkdir build && cd build \
      cmake .. \
      make -j $(nproc) \
      sudo make install  
```

```python
import cv2
print(cv2.__version__)
```



