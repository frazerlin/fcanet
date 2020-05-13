
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt
from core import init_model,predict

########################################[ Dataset ]########################################
#for general dataset format
class Dataset():
    def __init__(self,dataset_path,img_folder='img',gt_folder='gt',threshold=128,ignore_label=None):
        self.index,self.threshold,self.ignore_label = 0,threshold,ignore_label
        dataset_path=Path(dataset_path)
        self.img_files = sorted((dataset_path/img_folder).glob('*.*'))
        self.gt_files = [ next((dataset_path/gt_folder).glob(t.stem+'.*')) for t in self.img_files]
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.img_files)
    def __next__(self):
        if self.index > len(self) - 1:raise StopIteration
        img_src = np.array(Image.open(self.img_files[self.index]))
        gt_src = np.array(Image.open(self.gt_files[self.index]))
        gt = gt_src[:,:,0] if gt_src.ndim==3 else gt_src
        gt = np.uint8(gt>=self.threshold)
        if self.ignore_label is not None: gt[gt_src==self.ignore_label]=255
        self.index += 1
        return img_src,gt

#special for PASCAL_VOC2012
class VOC2012():
    def __init__(self,dataset_path):
        self.index = 0
        dataset_path=Path(dataset_path)
        with open(dataset_path/'ImageSets'/'Segmentation'/'val.txt') as f:
            val_ids=sorted(f.read().splitlines())

        self.img_files,self.gt_files,self.instance_indices=[],[],[]
        print('Preprocessing!')
        for val_id in tqdm(val_ids):
            gt_ins_set=  sorted(set(np.array(Image.open( dataset_path/'SegmentationObject'/(val_id+'.png'))).flat))
            for instance_index in gt_ins_set:
                if instance_index not in [0,255]:
                    self.img_files.append(  dataset_path/'JPEGImages'/(val_id+'.jpg'))
                    self.gt_files.append(  dataset_path/'SegmentationObject'/(val_id+'.png'))
                    self.instance_indices.append(instance_index)
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.img_files)
    def __next__(self):
        if self.index > len(self) - 1:raise StopIteration
        img_src = np.array(Image.open(self.img_files[self.index]))
        gt_src = np.array(Image.open(self.gt_files[self.index]))
        gt=np.uint8(gt_src==self.instance_indices[self.index])
        gt[gt_src==255]=255
        self.index += 1
        return img_src,gt


########################################[ Evaluation ]########################################
#robot user strategy
def get_next_anno_point(pred, gt, seq_points):
    fndist_map=distance_transform_edt(np.pad((gt==1)&(pred==0),((1,1),(1,1)),'constant'))[1:-1, 1:-1]
    fpdist_map=distance_transform_edt(np.pad((gt==0)&(pred==1),((1,1),(1,1)),'constant'))[1:-1, 1:-1]
    fndist_map[seq_points[:,1],seq_points[:,0]],fpdist_map[seq_points[:,1],seq_points[:,0]]=0,0
    [usr_map,if_pos] = [fndist_map, 1] if fndist_map.max()>fpdist_map.max() else [fpdist_map, 0]
    [y_mlist, x_mlist] = np.where(usr_map == usr_map.max())
    pt_next=(x_mlist[0],y_mlist[0],if_pos)
    return pt_next

datasets_kwargs={
    'GrabCut' :{'dataset_path':'dataset/GrabCut' ,'img_folder':'data_GT','gt_folder':'boundary_GT','threshold':128,'ignore_label':128 },
    'Berkeley':{'dataset_path':'dataset/Berkeley','img_folder':'images' ,'gt_folder':'masks'      ,'threshold':128,'ignore_label':None},
    'DAVIS'   :{'dataset_path':'dataset/DAVIS'   ,'img_folder':'img'    ,'gt_folder':'gt'         ,'threshold':0.5,'ignore_label':None},
    'VOC2012' :{'dataset_path':'dataset/VOC2012'},
}
default_miou_targets={'GrabCut':0.90,'Berkeley':0.90,'DAVIS':0.90,'VOC2012':0.85}

def eval_dataset(model, dataset, max_point_num=20, record_point_num=20,if_sis=False,miou_target=None,if_cuda=True):
    global datasets_kwargs, default_miou_targets
    if dataset in datasets_kwargs:
        dataset_iter= VOC2012(**datasets_kwargs[dataset]) if dataset=='VOC2012' else  Dataset(**datasets_kwargs[dataset]) 
        miou_target = default_miou_targets[dataset] if miou_target is None else miou_target
    else:
        dataset_iter=Dataset(dataset_path='dataset/{}'.format(dataset)) 
        miou_target = 0.85 if miou_target is None else miou_target

    NoC,mIoU_NoC=0,[0]*(record_point_num+1)
    for img,gt in tqdm(dataset_iter):
        pred = np.zeros_like(gt) 
        seq_points=np.empty([0,3],dtype=np.int64)
        if_get_target=False
        for point_num in range(1, max_point_num+1):
            pt_next = get_next_anno_point(pred, gt, seq_points)
            seq_points=np.append(seq_points,[pt_next],axis=0)
            pred = predict(model,img,seq_points,if_sis=if_sis,if_cuda=if_cuda)
            miou = ((pred==1)&(gt==1)).sum()/(((pred==1)|(gt==1))&(gt!=255)).sum()         
            if point_num<=record_point_num:
                mIoU_NoC[point_num]+=miou
            if (not if_get_target) and (miou>=miou_target or point_num==max_point_num):
                NoC+=point_num
                if_get_target=True
            if if_get_target and  point_num>=record_point_num: break
        
    print('dataset: [{}] {}:'.format(dataset,'(SIS)'if if_sis else ' '))
    print('--> mNoC : {}'.format(NoC/len(dataset_iter)))
    print('--> mIoU-NoC : {}\n\n'.format(np.array([round(i/len(dataset_iter),3) for i in mIoU_NoC ])))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation for FCANet")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'res2net'], help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='VOC2012', help='evaluation dataset (default: VOC2012)')
    parser.add_argument('--sis', action='store_true', default=False, help='use sis')
    parser.add_argument('--miou', type=float, default=-1.0, help='miou_target (default: -1.0[means automatic selection])')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu (not recommended)')
    args = parser.parse_args()

    if Path('dataset/{}'.format(args.dataset)).exists():
        model = init_model('fcanet',args.backbone,'./pretrained_model/fcanet-{}.pth'.format(args.backbone),if_cuda=not args.cpu)
        eval_dataset(model,args.dataset,if_sis=args.sis, miou_target=(None if args.miou<0 else args.miou),if_cuda=not args.cpu)
    else:
        print('not found folder [dataset/{}]'.format(args.dataset))
