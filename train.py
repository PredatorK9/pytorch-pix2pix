import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import argparse
import os
from dataset import MapDataset, RandomCropMap
from models import PatchGAN, Unet
from utils import apply_weights


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, type=str,
        help='Path to the dataset')
    parser.add_argument('--ckpth', required=False, type=str, default='./',
        help='Path to/for checkpoints of the model')
    parser.add_argument('--num_epoch', required=False, type=int,
        default=200, help='The number of epochs to train the model')
    parser.add_argument('--batch_size', required=False, type=int, default=1,
        help='The batch size for training the model')
    parser.add_argument('--pixel', required=False, type=float, default=100,
        help='Pixel loss (L1 loss) weight')
    parser.add_argument('--lr', required=False, type=float, default=0.0002,
        help='The learning rate of the optimizer')
    parser.add_argument('--beta1', required=False, type=float, default=0.5,
        help='Adam optimizer momentum parameters')
    parser.add_argument('--beta2', required=False, type=float, default=0.999,
        help='Adam optimizer momentum parameters')
    parser.add_argument('--cont', required=False, type=bool, default=False,
        help='Continue the training')
    parser.add_argument('--device', required=False, type=str, default='cpu',
        help='Training Device')

    arguments = parser.parse_args()

    return arguments


def getModelsandData(path, batch_size, device):
    netD = PatchGAN()
    netG = Unet()
    apply_weights(netD)
    apply_weights(netG)

    device = torch.device(device)

    transform = transforms.Compose([
        transforms.Resize((286, 572)),
        RandomCropMap(256),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5 ,0.5],
            [0.5, 0.5, 0.5]
        )
    ])

    dataset = MapDataset(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True)
    
    return netD, netG, dataloader, device


def train(num_epoch, ckpth, netD, netG, dataloader, lam,
    lr, beta1, beta2, cont, device):

    if cont:
        netD.load_state_dict(torch.load(os.path.join(ckpth, 'Discriminator.pth')))
        netG.load_state_dict(torch.load(os.path.join(ckpth, 'Generator.pth')))
        print('Checkpoints verified and loaded....\n')
    else:
        print('Starting...\n')

    netD.to(device)
    netG.to(device)

    patch_criterion = nn.BCELoss()
    pixel_criterion = nn.L1Loss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in range(1, num_epoch+1):
        for idx, (sat_image, map_image) in enumerate(dataloader):
            sat_image, map_image = sat_image.to(device), \
                map_image.to(device)

            optimizerD.zero_grad()
            optimizerG.zero_grad()

            real_output = netD(map_image, sat_image)
            real_labels = torch.ones_like(real_output).to(device)
            lossD_real = patch_criterion(real_output, real_labels)

            fake_sample = netG(map_image)
            fake_output = netD(map_image, fake_sample.detach())
            fake_labels = torch.zeros_like(fake_output).to(device)
            lossD_fake = patch_criterion(fake_output, fake_labels)

            lossD = lossD_real+lossD_fake
            lossD.backward()
            optimizerD.step()

            Gpatchout = netD(map_image, fake_sample)
            Gpatchloss = patch_criterion(Gpatchout, real_labels)
            Gpixelloss = pixel_criterion(fake_sample, sat_image)
            lossG = Gpatchloss + lam * Gpixelloss
            lossG.backward()
            optimizerG.step()

            if idx % 200 == 0:
                print(f'epoch[{epoch:03d}/{num_epoch}]=> '
                    f'lossD: {lossD.item():.4f}\tlossG: {lossG.item():.4f}')

        torch.save(netD.state_dict(), os.path.join(ckpth, 'Discriminator.pth'))
        torch.save(netG.state_dict(), os.path.join(ckpth, 'Generator.pth'))


def main():
    arguments = get_arguments()
    netD, netG, dataloader, device = getModelsandData(arguments.dataset,
                                        arguments.batch_size, arguments.device)
    train(arguments.num_epoch, arguments.ckpth, netD, netG, dataloader,
        arguments.pixel, arguments.lr, arguments.beta1, arguments.beta2,
        arguments.cont, device)

if __name__ == '__main__':
    main()