import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)


class BoxFilter(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r

    @staticmethod
    def diff_x(input, r):
        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:-r - 1]
        return torch.cat([left, middle, right], dim=2)

    @staticmethod
    def diff_y(input, r):
        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:-r - 1]
        return torch.cat([left, middle, right], dim=3)

    def forward(self, x):
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        _, _, h, w = lr_x.size()
        N = self.boxfilter(lr_x.new_ones((1, 1, h, w)))

        mean_x = self.boxfilter(lr_x) / N
        mean_y = self.boxfilter(lr_y) / N
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = F.interpolate(A, hr_x.size()[2:], mode="bilinear")
        mean_b = F.interpolate(b, hr_x.size()[2:], mode="bilinear")
        return mean_A * hr_x + mean_b


class E2EMEF(nn.Module):
    def __init__(self, config, radius=1, eps=1e-4, is_guided=True):
        super().__init__()
        self.config = config
        self.lr = self.build_lr_net()
        self.is_guided = is_guided
        if is_guided:
            self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        w_lr = self.lr(x_lr)
        w_hr = self.gf(x_lr, w_lr, x_hr) if self.is_guided else F.interpolate(
            w_lr, x_hr.size()[2:], mode="bilinear"
        )

        w_hr = torch.abs(w_hr)
        w_hr = (w_hr + self.config.EPS) / torch.sum((w_hr + self.config.EPS), dim=0)

        o_hr = torch.sum(w_hr * x_hr, dim=0, keepdim=True).clamp(0, 1)
        return o_hr, w_hr

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state[k] = v

        self.load_state_dict(new_state, strict=False)
        logger.info(f"[*] loaded checkpoint: {ckpt_path}", )

    def build_lr_net(self, norm=AdaptiveNorm, layer=5, width=24):
        layers = [
            nn.Conv2d(1, width, 3, 1, 1, bias=False),
            norm(width),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for l in range(1, layer):
            layers += [
                nn.Conv2d(width, width, 3, 1, padding=2 ** l, dilation=2 ** l, bias=False),
                norm(width),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            norm(width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, 1, 1, 1, 0),
        ]
        net = nn.Sequential(*layers)
        net.apply(self.weights_init_identity)
        return net

    @staticmethod
    def weights_init_identity(m):
        name = m.__class__.__name__
        if "Conv" in name:
            nn.init.xavier_uniform_(m.weight.data)
        elif "InstanceNorm2d" in name:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
