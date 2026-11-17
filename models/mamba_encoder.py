import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from collections import OrderedDict
from mamba_ssm import Mamba
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from mmcv.ops import DeformRoIPool
from einops.layers.torch import Reduce

_HAAR_KEYS = ('aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd')
_SQRT2 = math.sqrt(2.0)


def DWTn(image, wavelet_name='haar'):
    """Differentiable 3D Haar transform; return subbands in band-first order."""
    if wavelet_name != 'haar':
        raise ValueError('Only the Haar wavelet is supported')
    if image.ndim != 4 or any(size % 2 for size in image.shape[1:]):
        raise ValueError('Expected [B, C, H, W] with even C, H, W')

    coeffs = {'': image}
    for axis in (1, 2, 3):  # channels, height, width
        next_coeffs = {}
        for name, value in coeffs.items():
            index = [slice(None)] * value.ndim
            index[axis] = slice(0, None, 2)
            even = value[tuple(index)]
            index[axis] = slice(1, None, 2)
            odd = value[tuple(index)]
            next_coeffs[name + 'a'] = (even + odd) / _SQRT2
            next_coeffs[name + 'd'] = (even - odd) / _SQRT2
        coeffs = next_coeffs

    # [aaa for all B, aad for all B, ...] matches the existing WMB split.
    return torch.cat([coeffs[name] for name in _HAAR_KEYS], dim=0)


def IWTn(image, wavelet_name='haar'):
    """Invert the band-first 3D Haar transform without leaving autograd."""
    if wavelet_name != 'haar':
        raise ValueError('Only the Haar wavelet is supported')
    if image.ndim != 4 or image.shape[0] % 8 != 0:
        raise ValueError('Expected [8*B, C/2, H/2, W/2]')

    batch_size = image.shape[0] // 8
    coeffs = dict(zip(_HAAR_KEYS, image.split(batch_size, dim=0)))
    for axis in (3, 2, 1):  # width, height, channels
        next_coeffs = {}
        for prefix in dict.fromkeys(name[:-1] for name in coeffs):
            low, high = coeffs[prefix + 'a'], coeffs[prefix + 'd']
            even = (low + high) / _SQRT2
            odd = (low - high) / _SQRT2
            next_coeffs[prefix] = torch.stack(
                (even, odd), dim=axis + 1
            ).flatten(axis, axis + 1)
        coeffs = next_coeffs
    return coeffs['']

class DWT(nn.Module):
    def __init__(self,wavelet_name):
        super(DWT, self).__init__()
        self.wavelet_name=wavelet_name

    def forward(self, x):
        return DWTn(x,self.wavelet_name)

class IWT(nn.Module):
    def __init__(self,wavelet_name):
        super(IWT, self).__init__()
        self.wavelet_name=wavelet_name

    def forward(self, x):
        return IWTn(x,self.wavelet_name)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()

        hidden_features = hidden_dim

        self.rep_conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.rep_conv2 = nn.Conv2d(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_features) 

    def forward(self, x):
        identity = x
        x = self.project_in(x)
        x1 = x + self.rep_conv1(x) + self.rep_conv2(x)
        x1= self.bn(x1)
        x2 = self.dwconv(x)
        x2= self.bn(x2)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x + identity

class WM(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c*2, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.model1 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c,  # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

        #self.model2 = Mamba(
        #    # This module uses roughly 3 * expand * d_model^2 parameters
        #    d_model=c,  # Model dimension d_model
        #    d_state=32,  # SSM state expansion factor
        #    d_conv=4,  # Local convolution width
        #    expand=9,  # Block expansion factor
        #)
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.convb(x) + x
        #print(x.shape)
        x = self.ln(x.reshape(b, -1, c))
        #print(x.shape)
        y = self.model1(x).permute(0, 2, 1) 
        output = y.reshape(b, c, h, w)
        return self.smooth(output)

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        #print(img.shape)
        mean_c = img.mean(dim=1, keepdim=True)
        #print(mean_c.shape)
        # stx()
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    
def data_transform(X):
    return 2 * X - 1.0   

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm1(to_3d(x)), h, w)
    
class WMB(nn.Module):
    def __init__(self, dim, hidden_dim=1):
        super(WMB, self).__init__()
        self.DWT = DWT(wavelet_name='haar')#小波变换
        self.IWT = IWT(wavelet_name='haar')#小波反变换
        self.norm1 = LayerNorm(dim)
        self.illu = Illumination_Estimator(dim//2, n_fea_in=dim//2+1, n_fea_out=dim//2)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)
        self.mb = WM(dim//2)

    def forward(self, input_):
        x = input_
        n, c, h, w = x.shape
        #print(n,c,h,w)
        x = self.norm1(x)
        x = data_transform(x)
        #print(x.shape)
        input_dwt = self.DWT(x)
        input_LL, input_high = input_dwt[:n, ...], input_dwt[n:, ...]
        input_high, input_image = self.illu(input_high)
        #print(input_LL.shape)
        input_LL = self.mb(input_LL)
        #print(input_LL.shape)
        #print(torch.cat((input_LL, input_high), dim=0).shape)
        output = self.IWT(torch.cat((input_LL, input_high), dim=0))
        output = inverse_data_transform(output)
        #print(output.shape)
        x = x + output
        #print(x.shape)
        x = x + self.ffn(self.norm2(x))
        return x
    
class MambaNet(nn.Module):

    def __init__(self, f):
        self.inplanes = 16
        super(MambaNet, self).__init__()

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=f,out_channels=64,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(4,64),
                                             nn.SiLU())
        #self.wavemamba=128
        self.mamba=WMB(dim=64,hidden_dim=128)
        self.deform_fc_channels=64
        self.router = nn.Sequential(
                    nn.Linear(
                       f, self.deform_fc_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.deform_fc_channels,4))
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.deformpool = DeformRoIPool(output_size=(8,8),gamma=0.1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=x.squeeze(1)
        batch_size,D,H,W = x.shape
        center_h = (H // 2)  
        center_w = (W // 2)
        batchs = torch.arange(0,batch_size).to(x.device)
        rois_1 = torch.tensor([[0, 0,0,H,W]],dtype=torch.float32).to(x.device)
        rois_1 = rois_1.repeat(batch_size,1)
        rois_1[:,0]=batchs
        rois_2 = torch.tensor([[0, 1, 1, H-1, W-1]],dtype=torch.float32).to(x.device)
        rois_2 = rois_2.repeat(batch_size,1)
        rois_2[:,0]=batchs
        rois_3 = torch.tensor([[0, 2,2,H-2,W-2]],dtype=torch.float32).to(x.device)
        rois_3 = rois_3.repeat(batch_size,1)
        rois_3[:,0]=batchs
        rois_4 = torch.tensor([[0, 3,3,H-3,W-3]],dtype=torch.float32).to(x.device)
        rois_4 = rois_4.repeat(batch_size,1)
        rois_4[:,0] = batchs
        route_logits = self.router(x[:,:,center_h,center_w])
        #print(route_logits.shape)
        if self.training:
            route_scores = F.gumbel_softmax(route_logits, hard=True)
        else:
            route_scores = F.one_hot(route_logits.argmax(dim=-1), num_classes=4).to(route_logits.dtype)
        route_scores = route_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x1 = self.deformpool(x,rois_1)
        x2 = self.deformpool(x,rois_2)
        x3 = self.deformpool(x,rois_3)
        x4 = self.deformpool(x,rois_4)
        #print(x1.shape)
        x = x1*route_scores[:,0]+x2*route_scores[:,1]+x3*route_scores[:,2]+x4*route_scores[:,3]
        x=self.patch_embedding(x)#generate embedding B->B'
        #feature extraction
        x=self.mamba(x)#global and local 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

def mambanet(f):
    n = 3
    model = MambaNet(f)
    return model

if __name__ == '__main__':
    x = torch.randn(3,1,144,9, 9).cuda()
    f=144
    model = mambanet(f).cuda()
    print(model(x).shape)
