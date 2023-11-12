import torch
import torch.nn as nn
from WT import DWT, IWT
from numpy.lib.histograms import histogram
from numpy.lib.function_base import interp
import numpy as np
from torch.autograd import Variable
from utils.frequency import FrequencyLayer
import torch.nn.functional as F
from torch.nn import init
factor=1
def histeq(im, nbr_bins=256):
    im=im.cpu()
    im=im.detach().numpy()
    imhist, bins = histogram(im.flatten(), nbr_bins)
    cdf = imhist.cumsum()
    cdf = 1.0 * cdf / cdf[-1]
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)
def double_info(input):
    b,c,h,w=input.size()
    output = torch.ones(b, c, w, h, dtype=torch.float, requires_grad=False).cuda()
    for i in range(b):
        image=input[i,:,:,:]
        image=torch.squeeze(image)
        hist = histeq(image)
        hist = torch.from_numpy(hist).float()
        hist = hist.cuda()
        result = torch.cat([image, hist], dim=0)
        result=torch.unsqueeze(result,0)
        if i==0:
            output=result
        else:
            output=torch.cat([output,result],dim=0)
    return output
def multi_info(input):
    b,c,h,w=input.size()
    output = torch.ones(b, c, w, h, dtype=torch.float, requires_grad=False).cuda()
    for i in range(b):
        image=input[i,:,:,:]
        image=torch.squeeze(image)
        hist = histeq(image)
        hist = torch.from_numpy(hist).float()
        hist = hist.cuda()
        colormap = image / (image.sum(dim=0, keepdims=True) + 1e-4)
        result = torch.cat([image, hist, colormap], dim=0)
        result=torch.unsqueeze(result,0)
        if i==0:
            output=result
        else:
            output=torch.cat([output,result],dim=0)
    return output

##########################################################################

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                 nn.PReLU(),
                                 Downsample1(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(Downsample1(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))
        self.inc=in_channels
        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)
        self.conv1 = nn.Conv2d(in_channels, self.inc, 1, 1, 0)
    def forward(self, x):
        x = self.body(x)
        x=self.conv1(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##---------- Basic ----------
def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.body = [FAB(in_size,64,64)]
        self.body = nn.Sequential(*self.body)
        if downsample:
            self.downsample = DownSample(out_size,scale_factor=2)

        self.tail = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        out = self.body(x)
        out = self.tail(out) #96，256，256
        if self.downsample:
            out_down = self.downsample(out) #96，128，128
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = UpSample(in_size,scale_factor=2)
        self.conv_block = UNetConvBlock(in_size, out_size, downsample=False)
        self.sc=SC(int(in_size/2))
    def forward(self, x, bridge):
        up = self.up(x)
        out = [up, bridge]
        out = self.sc(out)
        out = self.conv_block(out)
        return out


class SF(nn.Module):
    def __init__(self, in_channels, height=4, reduction=8, bias=False):
        super(SF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])#(1,4,96,256,256)

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class SC(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SC, self).__init__()

        self.height = height  # number of block
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):  # list4,(1,96,256,256)
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)  # (1,384,256,256)     (1,768,64,64)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2],
                                   inp_feats.shape[3])  # (1,4,96,256,256)   (1,2,384,64,64)

        feats_U = torch.sum(inp_feats, dim=1)  # (1,96,256,256) (1,384,64,64)
        feats_S = self.avg_pool(feats_U)  # (1,96,1,1)   (1,384,1,1)
        feats_Z = self.conv_du(feats_S)  # (1,12,1,1)    (1,48,1,1)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # height*Conv(1,1)  list4,(1,96,1,1)  ()
        attention_vectors = torch.cat(attention_vectors, dim=1)  # (1,384,1,1)   (1,768,1,1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)  # (1,4,96,1,1) (1,2,384,1,1)

        attention_vectors = self.softmax(attention_vectors)  # (1,4,96,1,1) (1,2,384,1,1)
        feats_V = inp_feats * attention_vectors  # (1,4,96,256,256) (1,2,384,64,64)
        feats_V=feats_V.view(batch_size,n_feats*2,H, W)
        return feats_V
##########################################################################
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv_du(channel_pool)

        return x * y

##########################################################################

class FAB(nn.Module):
    def __init__(self,inchannel,dcth=32,dctw=32):
        super(FAB, self).__init__()
        self.halfchannel = int(inchannel/2)
        self.hhalfchannel= int(inchannel/4)
        self.att=MultiSpectralAttentionLayer(self.halfchannel,dcth, dctw)

        self.wsa = SALayer()

    def forward(self, x):
        wavelet_path_in, identity_path_in = torch.chunk(x, 2, dim=1)
        wavelet_path_out=self.att(wavelet_path_in)
        identity_path_out=self.wsa(identity_path_in)
        out = torch.cat([wavelet_path_out, identity_path_out], dim=1)
        out=out+x
        return out

##########################################################################

class FaNet(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=4):
        super(FaNet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        self.bili_down = bili_resize(0.5)
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = 0
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels + wf, (2 ** i) * wf, downsample))
            prev_channels = (2 ** i) * wf

        self.up_path = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth-1))

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.conv_up.append(nn.Sequential(*[bili_resize(2 ** i), nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)]))
            prev_channels = (2 ** i) * wf

        self.final_ff = SF(in_channels=wf, height=depth)
        self.last = conv3x3(prev_channels, int(in_chn/factor), bias=True)

    def forward(self, x):

        for inter in range(1):

            img = x
            scale_img = img
            layer1_input=scale_img
            x1 = self.conv_01(layer1_input) #
            encs = []
            for num in range(1):
                for i, down in enumerate(self.down_path):
                    if i == 0:
                        x1, x1_up = down(x1)
                        encs.append(x1_up)
                    elif (i + 1) < self.depth:
                        scale_img = self.bili_down(scale_img)

                        layer2_input=scale_img
                        left_bar = self.conv_01(layer2_input)
                        x1 = torch.cat([x1, left_bar], dim=1)
                        x1, x1_up = down(x1)
                        encs.append(x1_up)
                    else:
                        scale_img = self.bili_down(scale_img)

                        layer3_input=scale_img
                        left_bar = self.conv_01(layer3_input)
                        x1 = torch.cat([x1, left_bar], dim=1)
                        x1 = down(x1)

                ms_result = [self.bottom_up(self.bottom_conv(x1))]
                for i, up in enumerate(self.up_path):
                    x1 = up(x1, self.skip_conv[i](encs[-i - 1]))
                    ms_result.append(self.conv_up[i](x1))
                msff_result = self.final_ff(ms_result)
                x1=msff_result

            x = self.last(msff_result) + img

        return x


if __name__ == "__main__":
    from thop import profile
    input = torch.ones(1, 3, 256, 256, dtype=torch.float, requires_grad=False).cuda()

    model = FaNet(in_chn=3, wf=96, depth=4).cuda()
    out = model(input)
    flops, params = profile(model, inputs=(input,))

    print('input shape:', input.shape)
    print('parameters:', params/1e6)
    print('flops', flops/1e9)
    print('output shape', out.shape)
