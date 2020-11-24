import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=False, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) + dilation - 1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=False, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class BBlock(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DBlock_com(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_com1(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock_inv1(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class GCA(nn.Module):
    def __init__(self, input_shape, in_feats=64, feats=64, kernel_size=3, out_feats=64, n_channels=3):
        super(GCA, self).__init__()

        convs_g = [nn.Conv2d(in_feats, feats, kernel_size, 2, 1),
                   nn.ReLU(True),
                   nn.Conv2d(feats, feats, kernel_size, 2, 1),
                   nn.ReLU(True)]
        self.convs_g = nn.Sequential(*convs_g)

        conv_l = [nn.Conv2d(in_feats, feats, kernel_size, 1, 1),
                  nn.ReLU(True)]
        self.conv_l = nn.Sequential(*conv_l)

        convOut = [nn.Conv2d(feats, out_feats, kernel_size, 1, 1),
                   nn.ReLU(True)]
        self.convOut = nn.Sequential(*convOut)

        print(input_shape)
        n_points = int(feats * input_shape[0]/4 * input_shape[1] / 4 /4)
        print(n_points)

        self.pool = nn.AvgPool2d(2)

        fcs = [nn.Linear(n_points, 4 * out_feats),
               nn.ReLU(True),
               nn.Linear(4 * out_feats, 2 * out_feats),
               nn.ReLU(True),
               nn.Linear(2 * out_feats, out_feats)]
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        print("input:", x.shape)
        glob_ = self.convs_g(x)
        print("glob_:", glob_.shape)
        # glob = self.pool(_glob)
        # print("glob:", glob.shape)
        glob_ = self.pool(glob_)
        glob = glob_.view(glob_.shape[0], -1)
        print("glob:", glob.shape)

        fes = self.fcs(glob)
        print("fes:", fes.shape)
        # print(fes[0,0].mean().item())

        loc = self.conv_l(x)
        print("loc:", loc.shape)
        # print(loc[0,0,:,:].mean().item())
        mul = (loc.view(loc.shape[0] * loc.shape[1], -1) * fes.view(fes.shape[0] * fes.shape[1], -1)).view(loc.shape)
        print("mul:", mul.shape)
        # print(mul[0,0,:,:].mean().item())
        # print(fes[0,0].mean().item() * loc[0,0,:,:].mean().item())

        out = self.convOut(mul)

        return out



class MWCNN(nn.Module):
    """Pytorch implementation of MWCNN [1]_.

    Notes
    -----
    This class was taken from authors `Github repository <https://github.com/lpj-github-io/MWCNNv2>`_ with minor
    modifications.

    Attributes
    ----------
    n_feats : int
        Number of filters on the first convolutional block. Default used by [1]_ is 64. The number of filters doubles
        after each Discrete Wavelet Transform application.
    n_channels : int
        Number of channels. 1 for Grayscale, 3 for RGB.
    kernel_size : int
        Size of convolution filters.

    References
    ----------
    .. [1] Liu, P., Zhang, H., Lian, W., & Zuo, W. Multi-Level Wavelet Convolutional Neural Networks. IEEE Access
    """

    def __init__(self, n_feats=64, n_channels=1, kernel_size=3):
        super(MWCNN, self).__init__()
        conv = default_conv
        n_feats = n_feats
        self.scale_idx = 0
        nColor = n_channels

        act = nn.ReLU(True)

        self.DWT = DWT()
        self.IWT = IWT()

        self.GCA = GCA((320, 320))

        m_head = [BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))

        d_l1 = [BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        # GCA_x0 = self.GCA(x0)
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx