from dataset import ImageDataset
import torchvision.transforms as transforms
from config import data_root, batch_size, lr, momentum1, momentum2, epoches, cyc, lid
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import ReplayBuffer

def show_running_loss(running_loss, idx):
	x = np.array([i for i in range(len(running_loss))])
	y = np.array(running_loss)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('loss curve')
	plt.xlabel('step')
	plt.ylabel('loss value')
	plt.savefig('./result/loss-{}.png'.format(idx))


transforms_ = [
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

def train(dataloader, GAB, GBA, disA, disB):
  fake_A_buffer = ReplayBuffer()
  fake_B_buffer = ReplayBuffer()
  criterion_GAN = torch.nn.MSELoss()
  criterion_cycle = torch.nn.L1Loss()
  criterion_identity = torch.nn.L1Loss()  
  optimG = torch.optim.Adam(
    itertools.chain(GAB.parameters(), GBA.parameters()), lr=lr, betas=(momentum1, momentum2)
  )
  optimDA = torch.optim.Adam(disA.parameters(), lr=lr, betas=(momentum1, momentum2))
  optimDB = torch.optim.Adam(disB.parameters(), lr=lr, betas=(momentum1, momentum2))
  gloss = []
  dloss = []
  for epoch in range(epoches):
    it = 0
    tempd = 0.0
    tempg = 0.0
    for _, data in enumerate(dataloader):
      real_A = data['A']
      real_B = data['B']
      real_A = real_A.type(torch.FloatTensor)
      real_B = real_B.type(torch.FloatTensor)
      # label
      valid = Variable(torch.tensor(np.ones((real_A.size(0), *disA.output_shape))), requires_grad=False)
      fake = Variable(torch.tensor(np.ones((real_A.size(0), *disA.output_shape))), requires_grad=False)
      valid = valid.type(torch.FloatTensor)
      fake = valid.type(torch.FloatTensor)
      optimG.zero_grad()
      # identity loss
      loss_id_A = criterion_identity(GBA(real_A), real_A)
      loss_id_B = criterion_identity(GAB(real_B), real_B)
      loss_identity = (loss_id_A + loss_id_B) / 2
      # GAN loss
      fake_B = GAB(real_A)
      loss_GAN_AB = criterion_GAN(disB(fake_B), valid)
      fake_A = GBA(real_B)
      loss_GAN_BA = criterion_GAN(disA(fake_A), valid)
      lossGAN = (loss_GAN_AB + loss_GAN_BA) / 2
      # cycle loss
      recv_A = GBA(fake_B)
      loss_cycle_A = criterion_cycle(recv_A, real_A)
      recv_B = GAB(fake_A)
      loss_cycle_B = criterion_cycle(recv_B, real_B)
      loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
      # total loss of G
      loss_G = lossGAN + cyc * loss_cycle + lid * loss_identity
      tempg += loss_G.item()

      loss_G.backward()
      optimG.step()

      optimDB.zero_grad()
      #disB loss
      loss_real = criterion_GAN(disB(real_B), valid)
      # loss_fake = criterion_GAN(disB(fake_B), fake)
      fake_B_ = fake_B_buffer.push_and_pop(fake_B)
      loss_fake = criterion_GAN(disB(fake_B_.detach()), fake)
      loss_D_B = (loss_real + loss_fake) / 2
      loss_D_B.backward()
      optimDB.step()

      optimDA.zero_grad()
      # disA loss
      loss_real = criterion_GAN(disA(real_A), valid)
      fake_A_ = fake_A_buffer.push_and_pop(fake_A)
      loss_fake = criterion_GAN(disA(fake_A_.detach()), fake)
      loss_D_A = (loss_real + loss_fake) / 2
      loss_D_A.backward()
      optimDA.step()

      tempd += ((loss_D_A + loss_D_B) / 2).item()
      it += 1
      print(it, tempd)
    gloss.append(tempg / it)
    dloss.append(tempd / it)
    print('[%3d/%3d]: dloss: %.4f, gloss: %.4f' % (epoch, epoches, dloss[-1], gloss[-1]))
  show_running_loss(gloss, 1)
  show_running_loss(dloss, 1)
  return GAB, GBA, disA, disB




if __name__ == "__main__":
  trainset = ImageDataset(data_root, transforms_, 'train')
  trainloader = DataLoader(trainset, batch_size=batch_size)
  GAB = Generator(3, 3, 3)
  GBA = Generator(3, 3, 3)
  D_A = Discriminator((3, 256, 256))
  D_B = Discriminator((3, 256, 256))
  result = train(trainloader, GAB, GBA, D_A, D_B)