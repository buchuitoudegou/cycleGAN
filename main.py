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

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def show_running_loss(running_loss, idx, prefix):
	x = np.array([i for i in range(len(running_loss))])
	y = np.array(running_loss)
	plt.figure()
	plt.plot(x, y, c='b')
	plt.axis()
	plt.title('loss curve')
	plt.xlabel('step')
	plt.ylabel('loss value')
	plt.savefig('./result/{}-loss-{}.png'.format(prefix, idx))


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
  if cuda:
    criterion_GAN = criterion_GAN.cuda()
    criterion_cycle = criterion_cycle.cuda()
    criterion_identity = criterion_identity.cuda() 
    print('cuda available')
  else:
    print('cuda disavailable')
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
      real_A = real_A.type(Tensor)
      real_B = real_B.type(Tensor)
      # label
      valid = Variable(Tensor(np.ones((real_A.size(0), *disA.output_shape))), requires_grad=False)
      fake = Variable(Tensor(np.ones((real_A.size(0), *disA.output_shape))), requires_grad=False)
      valid = valid.type(torch.FloatTensor)
      fake = valid.type(torch.FloatTensor)
      if cuda:
        valid = valid.cuda()
        fake = fake.cuda()
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
      print(it, tempd, tempg, epoch)
    gloss.append(tempg / it)
    dloss.append(tempd / it)
    print('[%3d/%3d]: dloss: %.4f, gloss: %.4f' % (epoch, epoches, dloss[-1], gloss[-1]))
  show_running_loss(gloss, 1, 'generator')
  show_running_loss(dloss, 1, 'discriminator')
  return GAB, GBA, disA, disB




if __name__ == "__main__":
  trainset = ImageDataset(data_root, transforms_, 'train')
  trainloader = DataLoader(trainset, batch_size=batch_size)
  GAB = Generator(3, 3, 9)
  GBA = Generator(3, 3, 9)
  D_A = Discriminator((3, 256, 256))
  D_B = Discriminator((3, 256, 256))
  if cuda:
    GAB = GAB.cuda()
    GBA = GBA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
  Gab, Gba, disa, disb = train(trainloader, GAB, GBA, D_A, D_B)
  torch.save(Gab.state_dict(), './result/Gab.pkl')
  torch.save(Gba.state_dict(), './result/Gba.pkl')
  torch.save(disa.state_dict(), './result/disa.pkl')
  torch.save(disb.state_dict(), './result/disb.pkl')
