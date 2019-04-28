import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from dataset import ImageDataset
from model import Generator, Discriminator
from config import data_root, batch_size
import os

cuda = torch.cuda.is_available()

transforms_ = [
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

GBA = Generator(3, 3, 9)
GAB = Generator(3, 3, 9)
if cuda:
  print('cuda available')
  GBA = GBA.cuda()
  GAB = GAB.cuda()
GBA.load_state_dict(torch.load('./result/Gba.pkl'))
GBA.eval()
GAB.load_state_dict(torch.load('./result/Gab.pkl'))
GAB.eval()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# input_B = Tensor(batch_size, 3, 256, 256)
dataloader = DataLoader(ImageDataset(data_root, transforms_, 'test'), 
batch_size=batch_size, shuffle=False)
if not os.path.exists('outputA/'):
  os.makedirs('outputA/')
if not os.path.exists('outputB/'):
  os.makedirs('outputB/')
for i, data in enumerate(dataloader):
  real_B = Variable(data['B'])
  real_A = Variable(data['A'])
  if cuda:
    real_B = real_B.cuda()
    real_A = real_A.cuda()
  fake_A = 0.5 * (GBA(real_B).data + 1.0)
  fake_B = 0.5 * (GAB(real_A).data + 1.0)
  save_image(fake_A, 'outputA/%04d.png' % i)
  save_image(fake_B, 'outputB/%04d.png' % i)
  print('finish: [%04d/%04d]' % (i, len(dataloader)))
