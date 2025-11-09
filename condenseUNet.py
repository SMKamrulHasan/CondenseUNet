import math
import torch
import torch.nn as nn

from layers import Conv, CondenseConv, ShuffleLayer  # from your file

# ---- Dense pieces ----
class _DenseLayer(nn.Module):
    """ BN-ReLU-1x1 CondenseConv (L-Conv) -> BN-ReLU-3x3 grouped Conv -> concat """
    def __init__(self, in_ch, growth, args):
        super().__init__()
        bk = args.bottleneck
        self.conv1 = CondenseConv(in_ch, bk * growth, kernel_size=1, groups=args.group_1x1)
        self.conv2 = Conv(bk * growth, growth, kernel_size=3, padding=1, groups=args.group_3x3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, n_layers, in_ch, growth, args):
        super().__init__()
        ch = in_ch
        layers = []
        for _ in range(n_layers):
            layers.append(_DenseLayer(ch, growth, args))
            ch += growth
        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.block(x)


# ---- Helper: projection + (optional) compress after a DenseBlock ----
class _Project(nn.Module):
    """ 1x1 CondenseConv projection (learned groups) """
    def __init__(self, in_ch, out_ch, args):
        super().__init__()
        self.proj = CondenseConv(in_ch, out_ch, kernel_size=1, groups=args.group_1x1)
        self.out_ch = out_ch
    def forward(self, x): return self.proj(x)


# ---- U-Net ----
class CondenseUNet(nn.Module):
    """
    Exact implementation of your 128x128 sketch:
      Stem -> [DB1 -> Proj -> MaxPool] -> [DB2 -> Proj -> MaxPool] -> [DB3 -> Proj -> MaxPool]
      -> Bottleneck (DB4 + compress to 80)
      -> TU1 + concat + Proj(->80) + DB3_dec + Compress(80)
      -> TU2 + concat + Proj(->48) + DB2_dec + Compress(48)
      -> TU3 + concat + Proj(->32) + DB1_dec + Compress(32)
      -> 1x1 Head to num_classes
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        g1x1 = args.group_1x1
        g3x3 = args.group_3x3
        k = 16  # growth

        in_ch = getattr(args, 'in_channels', 1)    # your sketch is 128x128x1
        num_classes = args.num_classes

        # ---- Stem: 3x3(16) -> 7x7(48) ----
        self.stem1 = Conv(in_ch, 16, kernel_size=3, stride=1, padding=1, groups=1)
        self.stem2 = Conv(16, 48, kernel_size=7, stride=1, padding=3, groups=1)
        ch = 48  # 128x128x48

        # ---- Encoder DB1 (2 layers -> +32) => 80 ----
        self.db1 = _DenseBlock(2, ch, k, args)
        ch = self.db1.out_ch  # 80
        self.proj1 = _Project(ch, ch, args)       # keep 80, but enforce 1x1 CondenseConv
        self.pool1 = nn.MaxPool2d(2, 2)           # 64x64

        # ---- Encoder DB2 (3 layers -> +48) => 128 ----
        self.db2 = _DenseBlock(3, ch, k, args)    # input after pool is still 80
        ch = self.db2.out_ch  # 128
        self.proj2 = _Project(ch, ch, args)       # keep 128
        self.pool2 = nn.MaxPool2d(2, 2)           # 32x32

        # ---- Encoder DB3 (4 layers -> +64) => 192 ----
        self.db3 = _DenseBlock(4, ch, k, args)    # input after pool is 128
        ch = self.db3.out_ch  # 192
        self.proj3 = _Project(ch, ch, args)       # keep 192
        self.pool3 = nn.MaxPool2d(2, 2)           # 16x16

        # ---- Bottleneck DB4 (5 layers) + compress -> 80 ----
        self.db4 = _DenseBlock(5, ch, k, args)    # ch -> ch + 5*16
        ch_bn = self.db4.out_ch
        self.bn_compress = _Project(ch_bn, 80, args)  # your “5×16 → 80”
        ch = 80  # 16x16x80

        # ---- Decoder TU1 (to 32x32x64), concat with skip3(192) ----
        self.tu1 = nn.ConvTranspose2d(ch, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.dec1_proj = _Project(64 + 192, 80, args)     # project after concat to 80
        self.dec1_db = _DenseBlock(4, 80, k, args)        # mirror DB3 depth
        self.dec1_comp = _Project(self.dec1_db.out_ch, 80, args)
        ch = 80  # 32x32x80

        # ---- Decoder TU2 (to 64x64x64), concat with skip2(128) ----
        self.tu2 = nn.ConvTranspose2d(ch, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.dec2_proj = _Project(64 + 128, 48, args)
        self.dec2_db = _DenseBlock(3, 48, k, args)        # mirror DB2 depth
        self.dec2_comp = _Project(self.dec2_db.out_ch, 48, args)
        ch = 48  # 64x64x48

        # ---- Decoder TU3 (to 128x128x48), concat with skip1(80) ----
        self.tu3 = nn.ConvTranspose2d(ch, 48, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.dec3_proj = _Project(48 + 80, 32, args)
        self.dec3_db = _DenseBlock(2, 32, k, args)        # mirror DB1 depth
        self.dec3_comp = _Project(self.dec3_db.out_ch, 32, args)
        ch = 32  # 128x128x32

        # ---- Head ----
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1, bias=True)

        # init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.out_channels if hasattr(m, 'out_channels') else 1)
                m.weight.data.normal_(0, math.sqrt(2.0 / max(1, n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0); m.bias.data.zero_()

    def _crop_like(self, src, ref):
        """Center-crop src to ref spatial size (handles odd shapes)."""
        if src.size(-1) == ref.size(-1) and src.size(-2) == ref.size(-2):
            return src
        dh = (src.size(-2) - ref.size(-2)) // 2
        dw = (src.size(-1) - ref.size(-1)) // 2
        return src[..., dh:dh+ref.size(-2), dw:dw+ref.size(-1)]

    def forward(self, x):
        # Stem
        x = self.stem1(x)     # -> 16
        x = self.stem2(x)     # -> 48 (128x128)

        # DB1
        s1 = self.db1(x)      # -> 80
        y = self.proj1(s1)
        y = self.pool1(y)     # 64x64

        # DB2
        s2 = self.db2(y)      # -> 128
        y = self.proj2(s2)
        y = self.pool2(y)     # 32x32

        # DB3
        s3 = self.db3(y)      # -> 192
        y = self.proj3(s3)
        y = self.pool3(y)     # 16x16

        # Bottleneck
        y = self.db4(y)       # -> ch_bn
        y = self.bn_compress(y)  # -> 80

        # Decoder 1 (32x32)
        y = self.tu1(y)                        # -> 64 (32x32)
        s3c = self._crop_like(s3, y)           # align in case
        y = torch.cat([y, s3c], dim=1)         # 64 + 192
        y = self.dec1_proj(y)                  # -> 80
        y = self.dec1_db(y)                    # -> 80 + 4*16
        y = self.dec1_comp(y)                  # -> 80

        # Decoder 2 (64x64)
        y = self.tu2(y)                        # -> 64 (64x64)
        s2c = self._crop_like(s2, y)
        y = torch.cat([y, s2c], dim=1)         # 64 + 128
        y = self.dec2_proj(y)                  # -> 48
        y = self.dec2_db(y)                    # -> 48 + 3*16
        y = self.dec2_comp(y)                  # -> 48

        # Decoder 3 (128x128)
        y = self.tu3(y)                        # -> 48 (128x128)
        s1c = self._crop_like(s1, y)
        y = torch.cat([y, s1c], dim=1)         # 48 + 80
        y = self.dec3_proj(y)                  # -> 32
        y = self.dec3_db(y)                    # -> 32 + 2*16
        y = self.dec3_comp(y)                  # -> 32

        return self.head(y)  # logits N×C×128×128
