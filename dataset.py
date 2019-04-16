from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os

def load_all_images(root):
  files = os.listdir(root)
  images_list = []
  for f in files:
    if not os.path.isdir(f):
      images_list.append(root + f)
  return images_list

class ImageDataset(Dataset):
  def __init__(self, root, transform=[], mode='train'):
    self.transform = transforms.Compose(transform)
    self.set_A = load_all_images(root + mode + 'A/')
    self.set_B = load_all_images(root + mode + 'B/')
  
  def __len__(self):
    return max(len(self.set_A), len(self.set_B))
  
  def __getitem__(self, index):
    ima = self.set_A[index % len(self.set_A)]
    imb = self.set_B[index % len(self.set_B)]
    ima = cv2.imread(ima)
    imb = cv2.imread(imb)
    # ima = ima.reshape(ima.shape[0], ima.shape[2], ima.shape[1])
    # imb = imb.reshape(imb.shape[0], imb.shape[2], imb.shape[1])
    ima = self.transform(ima)
    imb = self.transform(imb)
    return { 'A': ima, 'B': imb }
