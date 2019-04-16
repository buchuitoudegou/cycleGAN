import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):
  def __init__(self, in_features):
    super(ResNet, self).__init__()
    conv = [
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features),
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features)
    ]
    self.model = nn.Sequential(*conv)
  
  def forward(self, input):
    return input + self.model(input)


class Generator(nn.Module):
  def __init__(self, in_channel=3, out_channel=3, res_blocks=0):
    super(Generator, self).__init__()
    model = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(in_channel, 64, 7),
      nn.InstanceNorm2d(64),
      nn.ReLU(inplace=True),
    ]
    in_feature = 64
    out_feature = in_feature * 2
    for _ in range(2):
      model += [
        nn.Conv2d(in_feature, out_feature, 3, stride=2, padding=1),
        nn.InstanceNorm2d(out_feature),
        nn.ReLU(inplace=True)
      ]
      in_feature = out_feature
      out_feature = in_feature * 2
    for _ in range(res_blocks):
      model += [ResNet(in_feature)]
    out_feature = in_feature // 2
    for _ in range(2):
      model += [
        nn.ConvTranspose2d(in_feature, out_feature, 3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(out_feature),
        nn.ReLU(inplace=True)
      ]
      in_feature = out_feature
      out_feature = in_feature // 2
    model += [
      nn.ReflectionPad2d(3),
      nn.Conv2d(64, out_channel, 7),
      nn.Tanh()
    ]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)

class Discriminator(nn.Module):
  def __init__(self, input_shape):
    super(Discriminator, self).__init__()
    channels, height, width = input_shape
    self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

    def dis_block(in_channel, out_channel, normalize=True):
      model = [nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1)]
      if normalize:
        model.append(nn.InstanceNorm2d(out_channel))
      model.append(nn.LeakyReLU(0.2, inplace=True))
      return model

    self.model = nn.Sequential(
      *dis_block(channels, 64, False),
      *dis_block(64, 128),
      *dis_block(128, 256),
      *dis_block(256, 512),
      nn.ZeroPad2d((1, 0, 1, 0)),
      nn.Conv2d(512, 1, 4, padding=1)
    )

  def forward(self, x):
    return self.model(x)