# FCANet
The official PyTorch implementation of CVPR 2020 paper ["Interactive Image Segmentation with First Click Attention"](http://mftp.mmcheng.net/Papers/20CvprFirstClick.pdf).

## Requirement
- PyTorch>=0.4.1  
- Opencv  
- Scipy  
- Matplotlib  

## Useage
Put the pretrained model into the folder "pretrained_model" and the unzipped datasets into the folder "dataset".

### Evalution:
```
python evaluate.py --backbone [resnet,res2net] --dataset [GrabCut,Berkeley,DAVIS,VOC2012] (--sis) 
```
### Demo with UI: 
```
python annotator.py --backbone [resnet,res2net] --input test.jpg --output test_mask.jpg (--sis)
```

*(\[x,y,z] means choices. (x) means optional.)*

## Datasets
- GrabCut ( [GoogleDrive](https://drive.google.com/open?id=1PhtmSoijh7THHZOmMt8_mwEF5Ii-3ofY) | [BaiduYun](https://pan.baidu.com/s/1u3LzLx3Xnr1kuU9HAnCN4w) pwd: **6qkj** )  
- Berkeley ( [GoogleDrive](https://drive.google.com/open?id=1vo0k3JrulK8C198lmvfbDhUok8S7gpTr) | [BaiduYun](https://pan.baidu.com/s/1B6T3aKWB2U6sIeWrQrnszQ) pwd: **6bfw** )  
- DAVIS ( [GoogleDrive](https://drive.google.com/open?id=1Cn0QvzYCIlFky5hFdeUYEtXaiTYW91rL) | [BaiduYun](https://pan.baidu.com/s/1qZTrFE7K_41CgsZyH5NJJw) pwd: **u5vd** )  
*(The three datasets we used are downloaded from [f-BRS](https://github.com/saic-vul/fbrs_interactive_segmentation))*
- PASCAL ( [official](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) )  


## Pretrained models
- fcanet-resnet ( [GoogleDrive](https://drive.google.com/open?id=1Zjn1RqAuNfG_VXjJ4X7RjX0cvuu6vFtM) | [BaiduYun](https://pan.baidu.com/s/1_Jjxb_0SU842FAj9RAEmBw ) pwd: **g989** )  
- fcanet-res2net ( [GoogleDrive](https://drive.google.com/open?id=12wgzK5Gx_ku3jSm9RcUx_OmEHU3HGfG_
) | [BaiduYun](https://pan.baidu.com/s/157Lkno7x8YSdmQzaYms-lg) pwd: **xzmx** )


## Other framework
We offer the [code](https://gitee.com/mindspore/models/tree/master/research/cv/FCANet) in constructed by MindSpore framework.


## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{lin2020fclick,
  title={Interactive image segmentation with first click attention},
  author={Lin, Zheng and Zhang, Zhao and Chen, Lin-Zhuo and Cheng, Ming-Ming and Lu, Shao-Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13339--13348},
  year={2020}
}
```
## Contact
If you have any questions, feel free to contact me via: `frazer.linzheng(at)gmail.com`.  
Welcome to visit [the project page](https://www.lin-zheng.com/fclick/) or [my home page](https://www.lin-zheng.com/).

## License
The source code is free for research and education use only. Any comercial use should get formal permission first.
