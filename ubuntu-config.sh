echo '> update apt-get'
apt-get update
echo '> install python3.6'
apt-get install python3.6
echo '> install dev'
pip3 install torch torchvision
pip3 install matplotlib opencv-python
echo '> fix cv2'
apt-get update
apt-get install libgtk2.0-dev