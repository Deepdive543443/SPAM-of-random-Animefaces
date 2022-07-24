import torch
import torch.nn as nn

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

    #Step 5 alpha 1
    def forward(self, x):
        x = self.init(x)
        for step in range(6):
            upsampled = self.upsample(x)
            x = self.main[step](upsampled)

        # upsampled_rgb = self.rgb[4](upsampled)
        progressed_rgb = self.rgb[6](x)
        return torch.tanh(progressed_rgb)