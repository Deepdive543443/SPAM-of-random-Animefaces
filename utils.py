import torch
import config
import time



def gradient_panelty(critic, real, fake, step, alpha):
    B,C,H,W = real.shape
    epsilon = torch.randn((B, 1, 1, 1)).repeat(1, C, H, W).to(config.DEVICE)
    interpolated_images = (epsilon*real + (1-epsilon)*fake.detach()).requires_grad_(True)
    interpolated_score = critic(interpolated_images, step, alpha)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_score,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_panelty = torch.mean((gradient_norm - 1)**2)
    return gradient_panelty

#Credit@ Aladdin Persson
def save_checkpoint(model, optimizer, image_size, epoch, alpha, noise, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "image_size": image_size,
        "epoch":epoch,
        "alpha":alpha,
        "fixed_noise":noise
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    image_size = checkpoint["image_size"]
    epoch = checkpoint["epoch"]
    alpha = checkpoint["alpha"]
    fixed_noise = checkpoint["fixed_noise"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return image_size, epoch, alpha, fixed_noise


def print_time():
    seconds = time.time()
    local_time = time.ctime(seconds)
    print("Local time:", local_time)
