# 提交github
from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from math import exp

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import cv2

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        # stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        stage_kernel_stride_pad = ((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        decoder_dim = 128,
        channels = 5,
        num_classes = 4
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),      # 融合之后的特征
            nn.Conv2d(decoder_dim, num_classes, 1),    # 分类
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
    
        return self.to_segmentation(fused)

class fusion(nn.Module):
    def __init__(self,c):
        super(fusion, self).__init__()
        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(int(c), int(c/2), bias=False)
        self.fc2 = nn.Linear(int(c/2), int(c), bias=False)
        self.sigmoid = nn.Sigmoid()
           
    def forward(self,x_s,x_t):
        a = self.global_average_pool(x_s)
        a = a.view(a.shape[0], -1)
        a_1 = self.fc1(a)
        a_1 = self.fc2(a_1)
        a_1 = self.sigmoid(a_1)
        a_enhanced = a * a_1
        a_enhanced = a_enhanced.unsqueeze(2)
        a_enhanced = a_enhanced.unsqueeze(2)
        F_T = a_enhanced * x_t
        F_T_spatial = torch.mean(F_T, dim=1,keepdim = True)
        
        return F_T_spatial
    
class enhance(nn.Module):
    def __init__(self):
        super(enhance, self).__init__()
        self.global_average_pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,F_S):
        a = self.global_average_pool(F_S)
        cos_sim = (F.normalize(a, dim= 1) * F.normalize(F_S, dim=1)).sum(1)
        batch = cos_sim.shape[0]
        cos_sim = cos_sim.detach().cpu().numpy()
        cos_sim = (cos_sim*255).astype(np.uint8)
        # batch = cos_sim[0]
        res = []
        for i in range(batch):
            cos_sim_equ = cv2.equalizeHist(cos_sim[i])
            res.append(cos_sim_equ)
        res_numpy =  np.array(res)   
        res_numpy = res_numpy.astype('float32') / 255 
        cos_sim_equ = torch.from_numpy(res_numpy)
        cos_sim_equ = cos_sim_equ.unsqueeze(1).cuda()
        F_en = cos_sim_equ * F_S
        
        return F_en
# 计算距离（欧几里得）
def euclDistance(vector1, vector2):
    vector1 = vector1.cuda()
    a = vector2 - vector1
    return torch.sqrt(torch.sum(a ** 2))

# 初始化质心
def initCentroids(data, k):
    numSamples , dim = data.shape
    # k个质心，列数跟样本的列数一样
    centroids = torch.zeros((k, dim))
    for i in range(k):
        index = int(torch.Tensor(1,1).uniform_(0, numSamples))
        # index = int(torch.Tensor(index))
        # centroids[i, :] = data[index, :].detach()
        centroids[i, :] = data[index,: ]
    return centroids 
       
def kmeans(data, k):
    numSamples = data.shape[0]
    clusterData = torch.zeros((numSamples, 2))
    clusterChanged = True
    centroids = initCentroids(data, k = 11)
    # centroids = torch.tensor(centroids)
    while clusterChanged:
        clusterChanged = False
        cluster1 = []
        
        for i in range(numSamples):
            cluster2 = []
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = euclDistance(centroids[j, :], data[i,: ])
                cluster2.append(distance.item())
                
              
                if distance < minDist:
                    minDist = distance
                    clusterData[i, 1] = minDist
                    minIndex = j
            cluster2 = torch.from_numpy(np.array(cluster2)).unsqueeze(0)
            cluster1.append(cluster2)
            if clusterData[i, 0] != minIndex:
                clusterChanged = True
                clusterData[i, 0] = minIndex
        for j in range(k):
            cluster_index = torch.nonzero(clusterData[:, 0] == j)
            pointsInCluster = data[cluster_index]
            centroids[j, :] = torch.mean(pointsInCluster, axis=0)
            cluster = [j,cluster_index]
        cluster1 = torch.from_numpy(np.concatenate(cluster1))
        cluster1 = torch.softmax(cluster1, dim=-1)  # batch, n, n
        return cluster1

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret



# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)       
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(4,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[1,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[1,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out

class RAR(nn.Module):
    def __init__(self):
        super(RAR,self).__init__()
        
        self.res18_MS = ResNet18(BasicBlock)
        self.res18_PAN = ResNet18(BasicBlock)
        self.R1  = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1, bias = False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace = False)
        )
        
        self.R1024_512  = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = False),
        )
        
        self.R512_256  = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = False),
        )
        
        self.R256_128  = nn.Sequential(
            nn.Conv2d(256, 11, 3, 1, 1, bias = False),
            nn.BatchNorm2d(11),
            nn.ReLU(inplace = False),
        )
    def forward(self, ms,pan):
        # up = torch.nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)
        # ms = up(ms)  # 3*4*64*64
        # pan = self.R1(pan)
        F_MS = self.res18_MS(ms)
        F_PAN = self.res18_PAN(pan)
        F_f = torch.cat([F_MS,F_PAN],dim=1)
        F_f = self.R1024_512(F_f)
        F_f = self.R512_256(F_f)
        F_f = self.R256_128(F_f)
        
        return F_f
                    
class SAT(nn.Module):
    def __init__(self, w=0.999):
        super(SAT,self).__init__()
        self.w = w
        
        self.R1_4 = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1, bias = False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace = False)
        )
        
        self.RP4_11_stu = nn.Sequential(
            nn.Conv2d(4, 11, 3, 1, 1, bias = False),
            nn.BatchNorm2d(11),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = False)
        )
        
        self.RP4_11_teacher = nn.Sequential(
            nn.Conv2d(4, 11, 3, 1, 1, bias = False),
            nn.BatchNorm2d(11),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace = False)
        )
        
        self.stu = Segformer(
            dims = (3, 64, 128, 256),      # dimensions通道数 of each stage
            heads = (1, 1, 1, 1),           # heads of each stage
            ff_expansion = (2, 2, 2, 2),    # feedforward expansion factor of each stage
            reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
            num_layers = 1,                 # num layers of each stage
            decoder_dim = 128,              # decoder dimension
            channels = 4,
            num_classes = 11                 # number of segmentation classes
        )
        self.teacher = Segformer(
            dims = (3, 64, 128, 256),      # dimensions通道数 of each stage
            heads = (1, 1, 1, 1),           # heads of each stage
            ff_expansion = (2, 2, 2, 2),    # feedforward expansion factor of each stage
            reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
            num_layers = 1,                 # num layers of each stage
            decoder_dim = 128,              # decoder dimension
            channels = 4,
            num_classes = 11                 # number of segmentation classes
        )
        
        self.teacher_pretrain = RAR()
        self.R22_11 = nn.Sequential(
            nn.Conv2d(22, 11, 3, 1, 1, bias = False),
            nn.BatchNorm2d(11),
            nn.ReLU(inplace = False)
        )
        self.feature_enhanced = enhance()
        self.feature_fusion = fusion(11)
        self.fc_1 = nn.Linear(11*32*32,10, bias=False)
        
        self.ssim_stu = SSIM()
        self.ssim_teacher = SSIM()
        
    def momentum_update(self):
        for param_stu,param_teacher in zip(self.stu.parameters(),self.teacher.parameters()):
            param_teacher = param_teacher.data * self.w + param_stu.data * (1 - self.w)
            
                     
    def forward(self,MS_1,PAN_1):
        
        with torch.no_grad():
            self.momentum_update()
            
        up = torch.nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)
        Input_stu = up(MS_1)  
        
        org_stu = self.RP4_11_stu(Input_stu) 
        
        Input_teacher = self.R1_4(PAN_1)
        
        org_teacher = self.RP4_11_teacher(Input_teacher)
        
        F_S = self.stu(Input_stu)                                                                                   
        F_T1 = self.teacher_pretrain(Input_stu,Input_teacher)                                                                                  
        F_T2 = self.teacher(Input_teacher)
        F_T = torch.cat([F_T1,F_T2],dim=1)
        F_T = self.R22_11(F_T) 
        
        F_en = self.feature_enhanced(F_S) 
        F_en = F_en.contiguous().view(F_en.shape[0], -1)
        F_en = self.fc_1(F_en)
       
         
        F_f = self.feature_fusion(F_S,F_T)
        F_f = F_f.contiguous().view(F_f.shape[0], -1)
        F_f = kmeans(F_f,k=11).cuda()
        
        
        ssim_stu = self.ssim_stu(org_stu,F_S)
        ssim_teacher = self.ssim_teacher(org_teacher,F_T)
        L = ssim_stu - ssim_teacher
        
        return F_en, F_f, L                         

if __name__ == "__main__":
    PAN1 = torch.randn(3, 1, 64, 64).cuda()
    MS1 = torch.randn(3, 4, 16, 16).cuda()
    
    model = SAT().cuda()
    out_result_teacher, out_result_student, L = model(MS1,PAN1)
    print(out_result_teacher.shape,out_result_student.shape,L.data)