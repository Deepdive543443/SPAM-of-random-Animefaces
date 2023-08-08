import torch
import torch.optim as optim
import config, shutil, os
from model import Generator, Discriminator
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm
from math import log2
import sys

torch.backends.cudnn.benchmarks = True
print(torch.cuda.get_device_name())

def tensorboard_plotting(gen, step, writer, real, fake, fixed_noise, alpha, tensorboard_step):
    gen.eval()
    with torch.no_grad():
        fixed_output = gen(fixed_noise, step, alpha)
        writer.add_image(str(step)+"fixed", make_grid(fixed_output*0.5 + 0.5), global_step=tensorboard_step)
        writer.add_image(str(step)+"random", make_grid(fake[:8]*0.5 + 0.5), global_step=tensorboard_step)
        writer.add_image(str(step)+"real", make_grid(real[:8]*0.5 + 0.5), global_step=tensorboard_step)
        if tensorboard_step % 500 == 0:
            save_image(make_grid(fixed_output*0.5 + 0.5),"cache/"+str(tensorboard_step)+"fixed.png")
    gen.train()
    return tensorboard_step + 1

def get_loader(image_size, batchsize):
    #Return the Dataloader for specific image size, return a dataloader at the end
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    dataset = ImageFolder(root=config.DATASET, transform = transform)
    if config.TEST_SPLIT:
        dataset, _ = torch.utils.data.random_split(dataset, [config.TEST_SPLIT, len(dataset) - config.TEST_SPLIT])
    dataloader = DataLoader(dataset,
                            batch_size=batchsize * 2 if config.FLOAT16 else batchsize,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.NUM_WORKERS,
                            drop_last=True)
    return dataloader, dataset

def train(gen, critic, optim_g, optim_d, g_scaler, d_scaler, loader, dataset, alpha, step, writer, fixed_noise, tensorboard_step):
    loop = tqdm(loader, leave=True) if config.BAR else loader

    for idx, (real, _) in enumerate(loop):
        real = real[:,:3,:,:].to(config.DEVICE)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, config.NOISE_DIM, 1, 1).to(config.DEVICE)
        # ===========================Training Discriminator==============================
        with torch.cuda.amp.autocast():
            fake = gen(noise, step, alpha)
            d_real_loss = critic(real, step, alpha)
            d_fake_loss = critic(fake.detach(), step, alpha)
            gp = gradient_panelty(critic, real, fake, step, alpha)
            d_loss = -(torch.mean(d_real_loss) - torch.mean(d_fake_loss)) + gp * config.LAMBDA_GP + (
                        0.001 * torch.mean(d_real_loss ** 2))

        optim_d.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(optim_d)
        d_scaler.update()

        # ===========================Training Generator==============================
        with torch.cuda.amp.autocast():
            g_fake = critic(fake, step, alpha)
            g_loss = -torch.mean(g_fake)

        optim_g.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(optim_g)
        g_scaler.update()

        alpha += batch_size / (len(dataset) * config.EPOCH * 0.5)
        alpha = min(alpha, 1)
        # loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item(), alpha=alpha)
        if idx % 50 == 0:
            tensorboard_step = tensorboard_plotting(gen, step, writer, real, fake, fixed_noise, alpha, tensorboard_step)
    return alpha, tensorboard_step

def train32(gen, critic, optim_g, optim_d, loader, dataset, alpha, step, writer, fixed_noise, tensorboard_step):
    loop = tqdm(loader, leave=True) if config.BAR else loader
    
    for idx, (real, _) in enumerate(loop):
        real = real[:,:3,:,:].to(config.DEVICE)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, config.NOISE_DIM, 1, 1).to(config.DEVICE)
        #===========================Training Discriminator==============================
        fake = gen(noise, step, alpha)
        d_real_loss = critic(real, step, alpha)
        d_fake_loss = critic(fake.detach(), step, alpha)
        gp = gradient_panelty(critic, real, fake, step, alpha)
        d_loss = -(torch.mean(d_real_loss) - torch.mean(d_fake_loss)) + gp*config.LAMBDA_GP + (0.001 * torch.mean(d_real_loss ** 2))

        optim_d.zero_grad()
        d_loss.backward()
        optim_d.step()

        # ===========================Training Generator==============================
        g_fake = critic(fake, step, alpha)
        g_loss = -torch.mean(g_fake)

        optim_g.zero_grad()
        g_loss.backward()
        optim_g.step()


        alpha += batch_size/(len(dataset) * config.EPOCH * 0.5)
        alpha = min(alpha, 1)
        # loop.set_postfix(d_loss = d_loss.item(), g_loss=g_loss.item(), alpha=alpha)
        if idx % 50 == 0:
            tensorboard_step = tensorboard_plotting(gen, step, writer, real, fake, fixed_noise, alpha, tensorboard_step)
    return alpha, tensorboard_step


def main():
    # Training pineline
    target_step = int(log2(config.TARGET_IMAGESIZE / 4))
    gen = Generator(latent_vector=config.NOISE_DIM, factors=config.FACTORS[:target_step]).to(config.DEVICE)
    critic = Discriminator(in_channels=config.NOISE_DIM, factors=config.FACTORS[:target_step]).to(config.DEVICE)
    optim_g = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optim_d = optim.Adam(critic.parameters(),lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    fixed_noise = torch.randn(8, config.NOISE_DIM, 1, 1).to(config.DEVICE)
    
    # Controlflow parameters
    image_size = config.START_IMAGESIZE
    epoch_checkpoint = 0
    alpha_checkpoint = config.ALPHA
    
    
    if config.LOAD_MODEL:
        fixed_noise = load_checkpoint("d_checkpoint.pth.tar", critic, optim_d, config.LEARNING_RATE)[3].to(config.DEVICE)
        image_size, epoch_checkpoint, alpha_checkpoint = load_checkpoint("g_checkpoint.pth.tar", gen, optim_g, config.LEARNING_RATE)
        print(f"Start training at image size:{image_size}, epoch: {epoch_checkpoint}, alpha:{alpha_checkpoint}")

    start_step = int(log2(image_size / 4))

    gen.train()
    critic.train()

    #============================Tensorboard===============================
    
    shutil.rmtree(f"cache")
    os.makedirs(f"cache")
    writer = SummaryWriter(f"cache")

    print(f"Generator:\n{gen}\nDiscriminator:{critic}\n")

    # Training loop
    for step in range(start_step, target_step+1):
        print(step, config.BATCH_SIZE[step])
        alpha = alpha_checkpoint
        tensorboard_step = 0
        loader, dataset = get_loader(image_size, config.BATCH_SIZE[step])
        for epoch in range(epoch_checkpoint, config.EPOCH):
            print_time()            
            print(f"[{epoch + 1}/{config.EPOCH}]")
            alpha, tensorboard_step = train(gen, critic, optim_g, optim_d, g_scaler, d_scaler, loader, dataset, alpha, step, writer, fixed_noise, tensorboard_step) if config.FLOAT16 else train32(gen, critic, optim_g, optim_d, loader, dataset, alpha, step, writer, fixed_noise, tensorboard_step)
            
            # Test if the training has failed(by checking the NaN values)
            test_outputs = gen(fixed_noise, step, 1)
            if (torch.isnan(test_outputs).sum() <= 0):
                if config.SAVE_MODEL:
                    save_checkpoint(critic, optim_d, image_size, epoch+1, alpha, fixed_noise, "d_checkpoint.pth.tar")
                    save_checkpoint(gen, optim_g, image_size, epoch+1, alpha, fixed_noise, "g_checkpoint.pth.tar")
                    sys.stdout.flush()
            else:
                assert ValueError ("Founded NaN values in outputs, stop training")
                break
            

        image_size *= 2
        epoch_checkpoint = 0
        alpha_checkpoint = config.ALPHA



if __name__ == "__main__":
    main()


