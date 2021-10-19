import time
import cv2
import sys
import torch
import config
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CULanesDataset, CityscapesDataset, tvtLaneDataset
from utils import weights_init, visualize_images
from model import Generator, Discriminator


if __name__ == "__main__":
    args = config.hyperparam()
    checkpoint_epoch = 0
    device = torch.device("cuda")
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    dataset = CityscapesDataset(args.root, transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    visualize_images(next(iter(dataloader)))

    netG = Generator(z_dim=100, out_channels=3).to(device)
    netG.apply(weights_init)

    netD = Discriminator(in_channels=3).to(device)
    netD.apply(weights_init)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5,0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5,0.999))
    criterion = nn.BCELoss()

    real_label = 1
    fake_label= 0

    if args.load_model==True:
        print("Loading models...")
        gen_checkpoint = torch.load("models/generator.pth")
        netG.load_state_dict(gen_checkpoint['model_state'])
        optimizerG.load_state_dict(gen_checkpoint['optim_state'])
        checkpoint_epoch = gen_checkpoint['epoch']+1

        disc_checkpoint = torch.load("models/discriminator.pth")
        netD.load_state_dict(disc_checkpoint['model_state'])
        optimizerD.load_state_dict(disc_checkpoint['optim_state'])
        print("Done!")
        print('-'*20)       

    print("Starting training...")
    print('-'*20)

    for epoch in range(checkpoint_epoch, checkpoint_epoch+args.epochs):
        start_time = time.time()
        for idx, real_img in enumerate(dataloader, 0):

            real_img = real_img.to(device).float()
            batch_size = real_img.size(0) #batch_size can be different for last batch in epoch
            ########################################
            # (1) DISCRIMINATOR --> REAL BATCH TRAINING, MAXIMIZE log(D(x)) + log(1-D(G(z)))
            ########################################

            netD.zero_grad()
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_img).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            ########################################
            # (2) DISCRIMINATOR --> FAKE BATCH TRAINING
            ########################################

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real+ lossD_fake
            optimizerD.step()

            ########################################
            # (3) GENERATOR --> MAXIMIZE log(D(G(z)))
            ########################################

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            ########################################
            # OUTPUT TRAINING STATS + PROGRESS
            ########################################

            if idx % (len(dataloader)//5) == 0:
                print(f"Epoch {epoch+1}/{checkpoint_epoch+args.epochs} Step {idx}/{len(dataloader)}, Loss_D: {lossD.item():.4f}, D(x): {D_x:.4f}, Loss_G: {lossG.item():.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        noise = torch.randn(64, 100, 1, 1, device=device)
        sample_8x8 = netG(noise)
        filepath = (f"samples_8x8/generated_road_scenes_epoch_0{epoch+1}.png")
        visualize_images(sample_8x8*0.5+0.5, filepath=filepath, save_image=True)

        noise = torch.randn(1, 100, 1, 1, device=device)
        sample_1x1 = netG(noise)
        filepath = (f"samples_1x1/generated_road_scenes_epoch_0{epoch+1}.png")
        visualize_images(sample_1x1*0.5+0.5, filepath=filepath, save_image=True)

        if args.save_model:
            print("Saving model....")
            gen_checkpoint = {'model_state': netG.state_dict(), 'optim_state': optimizerG.state_dict(), 'epoch': epoch}
            disc_checkpoint = {'model_state': netD.state_dict(), 'optim_state': optimizerD.state_dict(), 'epoch': epoch}
            torch.save(gen_checkpoint,f"models/generator.pth")
            torch.save(disc_checkpoint,f"models/discriminator.pth")
            print("Done!")
    print("Model training done!")
    print("-"*20)

    