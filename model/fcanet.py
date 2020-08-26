import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet_split
from . import res2net_split

########################################[ Global ]########################################

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def get_mask_gauss(mask_dist_src, sigma):
    return torch.exp(-2.772588722*(mask_dist_src**2)/(sigma**2))

########################################[ MultiConv ]########################################

class MultiConv(nn.Module):
    def __init__(self,in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None, BatchNorm=nn.BatchNorm2d):
        super(MultiConv, self).__init__()
        self.num=len(channels)
        if kernel_sizes is None: kernel_sizes=[ 3 for c in channels]
        if strides is None: strides=[ 1 for c in channels]
        if dilations is None: dilations=[ 1 for c in channels]
        if paddings is None: paddings = [ ( (kernel_sizes[i]//2) if dilations[i]==1 else (kernel_sizes[i]//2 * dilations[i]) ) for i in range(self.num)]
        convs_tmp=[]
        for i in range(self.num):
            if channels[i]==1:
                convs_tmp.append(nn.Conv2d( in_ch if i==0 else channels[i-1] , channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i]))
            else:
                convs_tmp.append(nn.Sequential(nn.Conv2d( in_ch if i==0 else channels[i-1] , channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], dilation=dilations[i],bias=False), BatchNorm(channels[i]), nn.ReLU()))
        self.convs=nn.Sequential(*convs_tmp)
        init_weight(self)
    def forward(self, x):
        return self.convs(x)

########################################[ MyASPP ]########################################

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        init_weight(self)

    def forward(self, x):
        x = self.relu(self.bn(self.atrous_conv(x)))
        return x

class MyASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations, BatchNorm, if_global=True):
        super(MyASPP, self).__init__()
        self.if_global = if_global

        self.aspp1 = _ASPPModule(in_ch, out_ch, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(in_ch, out_ch, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        if if_global:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                                                BatchNorm(out_ch),
                                                nn.ReLU())

        merge_channel=out_ch*5 if if_global else out_ch*4

        self.conv1 = nn.Conv2d(merge_channel, out_ch, 1, bias=False)
        self.bn1 = BatchNorm(out_ch)
        self.relu = nn.ReLU()
        init_weight(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        if self.if_global:
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

########################################[ MyDecoder ]########################################

class MyDecoder(nn.Module):
    def __init__(self, in_ch, in_ch_reduce, side_ch, side_ch_reduce, out_ch, BatchNorm, size_ref='side'):
        super(MyDecoder, self).__init__()
        self.size_ref=size_ref
        self.relu = nn.ReLU()
        self.in_ch_reduce, self.side_ch_reduce = in_ch_reduce, side_ch_reduce

        if in_ch_reduce is not None:
            self.in_conv = nn.Sequential( nn.Conv2d(in_ch, in_ch_reduce, 1, bias=False), BatchNorm(in_ch_reduce), nn.ReLU())
        if side_ch_reduce is not None:
            self.side_conv = nn.Sequential( nn.Conv2d(side_ch, side_ch_reduce, 1, bias=False), BatchNorm(side_ch_reduce), nn.ReLU())

        merge_ch=  (in_ch_reduce if in_ch_reduce is not None else in_ch) + (side_ch_reduce if side_ch_reduce is not None else side_ch) 
        
        self.merge_conv = nn.Sequential(nn.Conv2d(merge_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(out_ch),
                                       nn.ReLU(),
                                       nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(out_ch),
                                       nn.ReLU())
        init_weight(self)

    def forward(self, input, side):
        if self.in_ch_reduce is not None:
            input=self.in_conv(input)
        if self.side_ch_reduce is not None:
            side=self.side_conv(side)

        if self.size_ref=='side':
            input=F.interpolate(input, size=side.size()[2:], mode='bilinear', align_corners=True)
        elif self.size_ref=='input':
            side=F.interpolate(side, size=input.size()[2:], mode='bilinear', align_corners=True)

        merge=torch.cat((input, side), dim=1)
        output=self.merge_conv(merge)
        return output

########################################[ PredDecoder ]########################################

class PredDecoder(nn.Module):
    def __init__(self,in_ch,BatchNorm, if_sigmoid=False):
        super(PredDecoder, self).__init__()
        self.if_sigmoid=if_sigmoid
        self.pred_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(in_ch//2),
                                       nn.ReLU(),
                                       nn.Conv2d(in_ch//2, in_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(in_ch//2),
                                       nn.ReLU(),
                                       nn.Conv2d(in_ch//2, 1, kernel_size=1, stride=1))
        init_weight(self)
    def forward(self, input):
        output = self.pred_conv(input)
        if self.if_sigmoid:
            output=torch.sigmoid(output)
        return output

########################################[ Net ]########################################

class FCANet(nn.Module):
    def __init__(self, backbone='resnet', BatchNorm = nn.BatchNorm2d):
        super(FCANet, self).__init__()
        if backbone=='resnet':
            self.backbone_pre = resnet_split.ResNetSplit101(16,BatchNorm,False,[0,1],2)
            self.backbone_last = resnet_split.ResNetSplit101(16,BatchNorm,False,[2,4],0)
        elif backbone=='res2net':
            self.backbone_pre = res2net_split.Res2NetSplit101(16,BatchNorm,False,[0,1],2)
            self.backbone_last = res2net_split.Res2NetSplit101(16,BatchNorm,False,[2,4],0)

        self.my_aspp = MyASPP(in_ch=2048+512,out_ch=256,dilations=[1, 6, 12, 18],BatchNorm=BatchNorm, if_global=True)
        self.my_decoder=MyDecoder(in_ch=256, in_ch_reduce=None, side_ch=256, side_ch_reduce=48,out_ch=256,BatchNorm=BatchNorm)
        self.pred_decoder=PredDecoder(in_ch=256, BatchNorm=BatchNorm)
        self.first_conv=MultiConv(257,[256,256,256,512,512,512],[3,3,3,3,3,3],[2,1,1,2,1,1])
        self.first_pred_decoder=PredDecoder(in_ch=512, BatchNorm=BatchNorm)
        
    def forward(self, input):
        [img, pos_mask_dist_src, neg_mask_dist_src, pos_mask_dist_first]=input
        pos_mask_gauss, neg_mask_gauss=  get_mask_gauss(pos_mask_dist_src,10), get_mask_gauss(neg_mask_dist_src,10)
        pos_mask_gauss_first = get_mask_gauss(pos_mask_dist_first,30)
        img_with_anno = torch.cat((img, pos_mask_gauss, neg_mask_gauss), dim=1)
        l1=self.backbone_pre(img_with_anno)
        l4=self.backbone_last(l1)
        l1_first=torch.cat((l1, F.interpolate(pos_mask_gauss_first, size=l1.size()[2:], mode='bilinear', align_corners=True)),dim=1)
        l1_first=self.first_conv(l1_first)
        l4=torch.cat((l1_first,l4),dim=1)
        x=self.my_aspp(l4)
        x=self.my_decoder(x,l1)
        x=self.pred_decoder(x)
        result = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        return result


