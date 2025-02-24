import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, num_downs=8, ngf=64):
        super(UNetGenerator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_downs = num_downs
        self.ngf = ngf

        # Encoder
        self.down_layers = nn.ModuleList()
        for i in range(num_downs):
            in_channels = input_channels if i == 0 else ngf * 2 ** (i - 1)
            out_channels = ngf * 2**i
            self.down_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.down_layers.append(nn.BatchNorm2d(out_channels))
            self.down_layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Decoder
        self.up_layers = nn.ModuleList()
        for i in reversed(range(num_downs - 1)):
            in_channels = ngf * 2 ** (i + 1)
            out_channels = ngf * 2**i
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.up_layers.append(layer)
            # self.up_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            # self.up_layers.append(nn.BatchNorm2d(out_channels))
            # self.up_layers.append(nn.ReLU(inplace=True))

        # Output layer
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(
                ngf, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x, c):
        # Concatenate condition to input image
        x = torch.cat([x, c], dim=1)
        self.input_channels += c.shape[1]
        # Encoder
        skips = []
        for layer in self.down_layers:
            x = layer(x)
            if layer.__class__.__name__ == "Conv2d":
                skips.append(x)

        # Decoder
        skips = reversed(skips[:-1])
        for layer, skip in zip(self.up_layers, skips):
            # torch.cuda.empty_cache()
            # print("x:",x.shape)
            x = layer(x)
            # print("layer:",x.shape,"skip:",skip.shape)
            x = torch.cat([x, skip], dim=1)
            # print("before trim:",x.shape)
            # x = self.trim_layer(x,x.shape[1],x.shape[1]//2)
            x = layer(x)
            x = nn.MaxPool2d(2)(x)
            # x = self.trim_layer(x,x.shape[1],x.shape[1]//2)
            # print("after trim:",x.shape)
            # x = (x + skip) / 2.0
        # Output layer
        x = self.out_layer(x)
        return x
