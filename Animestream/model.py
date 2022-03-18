import torch
import torch.nn as nn
import config


class WSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2, transpose=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding) if transpose else nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain/(in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self,epsilon = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ProgressiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inversed = False, use_pn = True):
        super().__init__()
        layers = []
        layers.append(WSConv(in_channels, in_channels) if inversed else WSConv(in_channels, out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_pn:
            layers.append(PixelNorm())
        layers.append(WSConv(in_channels, out_channels) if inversed else WSConv(out_channels, out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_pn:
            layers.append(PixelNorm())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


#===============================Model==================================#
class Generator(nn.Module):
    def __init__(self, img_channels = 3, latent_vector = 512, factors = [1, 1, 1, 2, 2, 2, 2, 2]):
        super().__init__()
        self.init = nn.Sequential(
            WSConv(latent_vector, latent_vector, kernel_size= 4, stride=1, padding=0, transpose=True),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            WSConv(latent_vector, latent_vector),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.main = nn.ModuleList()
        self.rgb = nn.ModuleList().append(WSConv(latent_vector,img_channels, kernel_size=1, stride=1, padding=0))

        for factor in factors:
            block = ProgressiveBlock(latent_vector,latent_vector//factor)
            self.main.append(block)
            self.rgb.append(WSConv(latent_vector//factor, img_channels, kernel_size=1, stride=1, padding=0))
            latent_vector = latent_vector // factor

    def fade_in(self, alpha, upsampled, final):
        return torch.tanh(alpha * final + (1 - alpha) * upsampled)

    def forward(self, x, steps, alpha):
        x = self.init(x)
        if steps == 0:
            return self.rgb[0](x)

        for step in range(steps):
            upsampled = self.upsample(x)
            x = self.main[step](upsampled)

        upsampled_rgb = self.rgb[steps-1](upsampled)
        progressed_rgb = self.rgb[steps](x)

        return self.fade_in(alpha, upsampled_rgb, progressed_rgb)


class Discriminator(nn.Module):
    def __init__(self, img_channels = 3, in_channels = 512, factors = [1, 1, 1, 2, 2, 2, 2, 2]):
        super().__init__()
        self.final = nn.Sequential(
            WSConv(in_channels+1, in_channels),
            nn.LeakyReLU(0.2),
            WSConv(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv(in_channels, 1, kernel_size=1, stride=1,padding=0),
            nn.Flatten()
        )

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.main, self.fromRGB = nn.ModuleList(), nn.ModuleList().append(WSConv(img_channels, in_channels, kernel_size=1, stride=1, padding=0))
        for factor in factors:
            to_channels = int(in_channels / factor)
            self.fromRGB.append(WSConv(img_channels, to_channels, kernel_size=1, stride=1, padding=0))
            self.main.append(ProgressiveBlock(to_channels, in_channels, inversed=True, use_pn=False))
            in_channels = to_channels

    def fade_in(self, alpha, downsampled, progressed):
        return torch.tanh(alpha * progressed + (1 - alpha) * downsampled)

    def minibatch_std(self, x):
        std = torch.mean(torch.std(x, dim=0)).repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)

    def forward(self, x, steps, alpha):
        #for 4x4 we don't need to run through progresssive block, after toRGB and minibatchstd send it to final block directly
        if steps == 0:
            x = self.fromRGB[0](x)
            x = self.minibatch_std(x)
            return self.final(x)
        #for others we are going start with a fade in with progressive, then all progressive
        progressed = self.downsample(self.main[steps-1](self.fromRGB[steps](x)))
        downsampled = self.fromRGB[steps-1](self.downsample(x))
        x = self.fade_in(alpha, downsampled, progressed)

        for step in range(steps-2, -1, -1):
            progress = self.main[step](x)
            x = self.downsample(progress)

        x = self.minibatch_std(x)
        return self.final(x)


if __name__ == "__main__":
    from math import log2
    from torchsummary import summary
    size = 1024
    step = int(log2(size/4))
    print(step)
    latent_vector = 512

    model = Generator(latent_vector=latent_vector, factors=config.FACTORS[:step])
    rand = torch.randn(2, latent_vector, 1, 1)
    result = model(rand, step, 0.5)
    print("\nGenerator result")
    print(result.shape)

    model = Discriminator(in_channels=latent_vector, factors=config.FACTORS[:step])
    result = model(result,step,0.5)
    print("result")
    print(result)