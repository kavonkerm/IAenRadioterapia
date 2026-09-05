import torch
import torch.nn as nn

class DobleConv3D(nn.Module):
    """Bloque de dos convoluciones 3D consecutivas con InstanceNorm y LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bloque = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.bloque(x)


class UNet3D(nn.Module):
    """
    Arquitectura 3D U-Net para segmentación de próstata.
    Entrada esperada: (Batch, 1, 48, 160, 160)
    Salida de logits: (Batch, 1, 48, 160, 160)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters

        # Encoder (Contracción)
        self.enc1 = DobleConv3D(in_channels, f)        # Salida: f (16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = DobleConv3D(f, f * 2)             # Salida: 2f (32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = DobleConv3D(f * 2, f * 4)         # Salida: 4f (64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck (Cuello de botella)
        self.bottleneck = DobleConv3D(f * 4, f * 8)   # Salida: 8f (128)

        # Decoder (Expansión y reconstrucción espacial)
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec3 = DobleConv3D(f * 8 + f * 4, f * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec2 = DobleConv3D(f * 4 + f * 2, f * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.dec1 = DobleConv3D(f * 2 + f, f)

        # Proyección final a 1 canal binario
        self.salida = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Rama codificadora
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Base
        b = self.bottleneck(self.pool3(e3))

        # Rama decodificadora con Skip Connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.salida(d1)


if __name__ == "__main__":
    modelo = UNet3D()
    tensor_test = torch.randn(1, 1, 48, 160, 160)
    out = modelo(tensor_test)
    print("✅ Modelo 3D U-Net validado con éxito.")
    print(f"Dimensiones de entrada: {tensor_test.shape} -> Salida: {out.shape}")