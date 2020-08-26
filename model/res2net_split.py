import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                        downsample=None, baseWidth=26, scale = 4, stype='normal', BatchNorm=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage' and stride>1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, dilation=dilation, padding=dilation, bias=False))
          bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.stride==1 and self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2NetSplit(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, layer_range=[0,4], aux_channel=0, baseWidth = 26, scale = 4):
        super(Res2NetSplit, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.layer_range=layer_range
        self.aux_channel = aux_channel
        self.ori_inplanes=[3, 64, 64*block.expansion, 128*block.expansion, 256*block.expansion]
        self.inplanes = self.ori_inplanes[layer_range[0]]+aux_channel

        # Modules
        if layer_range[0]<=0 and 0<=layer_range[1]:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.inplanes, 32, 3, 2, 1, bias=False),
                BatchNorm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                BatchNorm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            )
            self.bn1 = BatchNorm(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.inplanes=64

        if layer_range[0]<=1 and 1<=layer_range[1]:
            self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)

        if layer_range[0]<=2 and 2<=layer_range[1]:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)

        if layer_range[0]<=3 and 3<=layer_range[1]:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)

        if layer_range[0]<=4 and 4<=layer_range[1]:
            self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
            # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self._init_weight()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample=downsample, 
                        BatchNorm=BatchNorm, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                        BatchNorm=BatchNorm, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, input, if_final=True):
        x=input
        results=[]
        if self.layer_range[0]<=0 and 0<=self.layer_range[1]:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            l0=x
            results.append(l0)

        if self.layer_range[0]<=1 and 1<=self.layer_range[1]:
            x = self.layer1(x)
            l1=x
            results.append(l1)
        

        if self.layer_range[0]<=2 and 2<=self.layer_range[1]:
            x = self.layer2(x)
            l2=x
            results.append(l2)
            

        if self.layer_range[0]<=3 and 3<=self.layer_range[1]:
            x = self.layer3(x)
            l3=x
            results.append(l3)
            
        if self.layer_range[0]<=4 and 4<=self.layer_range[1]:
            x = self.layer4(x)
            l4=x
            results.append(l4)
            
        if if_final:
            return results[-1]
        else:
            return results

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrain_url):
        pretrain_dict = model_zoo.load_url(pretrain_url)
        model_dict = {}
        state_dict = self.state_dict()
        first_weight_names=['conv1.0.weight','layer1.0.conv1.weight','layer2.0.conv1.weight','layer3.0.conv1.weight','layer4.0.conv1.weight']
        ds_weight_names=['None','layer1.0.downsample.1.weight','layer2.0.downsample.1.weight','layer3.0.downsample.1.weight','layer4.0.downsample.1.weight']
        for k, v in pretrain_dict.items():
            if  self.aux_channel>0 and (k==first_weight_names[self.layer_range[0]]  or k==ds_weight_names[self.layer_range[0]]):
                v_tmp=state_dict[k].clone()
                v_tmp[:,:self.ori_inplanes[self.layer_range[0]],:,:]=v.clone()
                model_dict[k] = v_tmp
                continue
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def Res2NetSplit50(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = Res2NetSplit(Bottle2neck, [3, 4, 6, 3], output_stride, BatchNorm, layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        model._load_pretrained_model('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth')
    return model

def Res2NetSplit101(output_stride, BatchNorm, pretrained=False,layer_range=[0,4], aux_channel=0):
    model = Res2NetSplit(Bottle2neck, [3, 4, 23, 3], output_stride, BatchNorm, layer_range=layer_range, aux_channel=aux_channel)
    if pretrained:
        model._load_pretrained_model('https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth')
    return model


