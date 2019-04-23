echo '> update apt-get'
sudo apt-get update
echo '> install dev'
pip3 install torch torchvision
pip3 install sklearn
pip3 install matplotlib opencv-python
echo '> fix cv2'
apt-get update
apt-get install libgtk2.0-dev