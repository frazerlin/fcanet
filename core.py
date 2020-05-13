import cv2
import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from model.fcanet import FCANet

########################################[ Encapsulation ]########################################

def get_points_mask(size, points):
    mask=np.zeros(size[::-1]).astype(np.uint8)
    if len(points)!=0:
        points=np.array(points)
        mask[points[:,1], points[:,0]]=1
    return mask

def structural_integrity_strategy(pred, pos_mask):
    pos_mask=((pos_mask==1)&(pred==1)).astype(np.uint8)
    h,w=pred.shape
    mask=np.zeros([h+2, w+2], np.uint8)
    pred_new=pred.copy()
    pts_y, pts_x = np.where(pos_mask==1)
    pts_xy=np.concatenate((pts_x[:,np.newaxis], pts_y[:,np.newaxis]), axis=1)
    for pt in pts_xy:
        cv2.floodFill(pred_new, mask, tuple(pt),2) 
    pred_new=(pred_new==2).astype(np.uint8)
    return pred_new

def img_resize_point(img, size):
    (h, w) = img.shape
    if not isinstance(size, tuple): size=( int(w*size), int(h*size) )
    M=np.array([[size[0]/w,0,0],[0,size[1]/h,0]])
    pts_y, pts_x= np.where(img==1)
    pts_xy=np.concatenate( (pts_x[:,np.newaxis], pts_y[:,np.newaxis]), axis=1 )
    pts_xy_new= np.dot( np.insert(pts_xy,2,1,axis=1), M.T).astype(np.int64)
    img_new=np.zeros(size[::-1],dtype=np.uint8)
    for pt in pts_xy_new:
        img_new[pt[1], pt[0]]=1
    return img_new

class Resize(object):
    def __init__(self, size, mode=None,  elems_point=['pos_points_mask','neg_points_mask','first_point_mask'], elems_do=None, elems_undo=[]):
        self.size, self.mode = size, mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            if elem in self.elems_point:
                sample[elem]=img_resize_point(sample[elem],self.size)
                continue
            if self.mode is None: 
                mode = cv2.INTER_LINEAR if len(sample[elem].shape)==3 else cv2.INTER_NEAREST
            sample[elem] = cv2.resize(sample[elem], self.size, interpolation=mode)
        return sample

class CatPointMask(object):
    def __init__(self, mode='NO', paras={}, if_repair=True):
        self.mode,self.paras,self.if_repair = mode, paras, if_repair
    def __call__(self, sample):
        gt = sample['gt']
        if_gt_empty= not ((gt>127).any())
        pos_points_mask, neg_points_mask = sample['pos_points_mask'], sample['neg_points_mask']
        if self.mode == 'DISTANCE_POINT_MASK_SRC':
            max_dist=255
            if if_gt_empty:
                pos_points_mask_dist = np.ones(gt.shape).astype(np.float64)*max_dist
            else:
                pos_points_mask_dist = distance_transform_edt(1-pos_points_mask)
                pos_points_mask_dist = np.minimum(pos_points_mask_dist, max_dist)
            if neg_points_mask.any()==False:
                neg_points_mask_dist = np.ones(gt.shape).astype(np.float64)*max_dist
            else:
                neg_points_mask_dist = distance_transform_edt(1-neg_points_mask)
                neg_points_mask_dist = np.minimum(neg_points_mask_dist, max_dist)
            pos_points_mask_dist, neg_points_mask_dist = pos_points_mask_dist*255, neg_points_mask_dist*255
            sample['pos_mask_dist_src'] = pos_points_mask_dist
            sample['neg_mask_dist_src'] = neg_points_mask_dist
        return sample

class ToTensor(object):
    def __init__(self, if_div=True, elems_do=None, elems_undo=[]):
        self.if_div = if_div
        self.elems_do, self.elems_undo = elems_do, (['meta']+elems_undo)
    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do!= None  and elem not in self.elems_do :continue
            if elem in self.elems_undo:continue
            tmp = sample[elem]
            tmp = tmp[np.newaxis,:,:] if tmp.ndim == 2 else tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp).float()
            tmp = tmp.float().div(255) if self.if_div else tmp
            sample[elem] = tmp                          
        return sample

########################################[ Interface ]########################################

def init_model(model_name='fcanet',backbone='resnet',pretrained_file=None, if_cuda=True):
    print('Backbone is {}'.format(backbone))
    if model_name=='fcanet':
        model=FCANet(backbone=backbone)
    if if_cuda: model = model.cuda()
    model.eval()
    if pretrained_file is not None:
        if if_cuda:
            state_dict=torch.load(pretrained_file)
        else:
            state_dict=torch.load(pretrained_file,map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print('load from [{}]!'.format(pretrained_file))
    return model

def predict(model,img, seq_points,  if_sis=False, if_cuda=True):
    h,w,_ =img.shape
    sample={}
    sample['img']=img.copy()
    sample['gt']=(np.ones((h,w))*255).astype(np.uint8)
    sample['pos_points_mask'] = get_points_mask((w,h),seq_points[seq_points[:,2]==1,:2])
    sample['neg_points_mask'] = get_points_mask((w,h),seq_points[seq_points[:,2]==0,:2])
    sample['first_point_mask'] = get_points_mask((w,h),seq_points[0:1,:2])
    Resize((int(w*512/min(h, w)),int(h*512/min(h, w))))(sample)
    CatPointMask(mode='DISTANCE_POINT_MASK_SRC', if_repair=False)(sample)
    sample['pos_mask_dist_first'] = np.minimum(distance_transform_edt(1-sample['first_point_mask']), 255.0)*255.0
    ToTensor()(sample)
    input=[sample['img'].unsqueeze(0),  sample['pos_mask_dist_src'].unsqueeze(0), sample['neg_mask_dist_src'].unsqueeze(0), sample['pos_mask_dist_first'].unsqueeze(0)]
    if if_cuda:
        for i in range(len(input)):
            input[i]=input[i].cuda()
    with torch.no_grad(): 
        output = model(input)
    result = torch.sigmoid(output.data.cpu()).numpy()[0,0,:,:]
    result = cv2.resize(result, (w,h), interpolation=cv2.INTER_LINEAR)     
    pred = (result>0.5).astype(np.uint8)
    if if_sis: pred=structural_integrity_strategy(pred,get_points_mask((w,h),seq_points[seq_points[:,2]==1,:2]))
    return pred

