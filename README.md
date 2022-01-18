## GA-NET: Global Attention Network for Point Cloud Semantic Segmentation

We propose a global attention network, called GA-Net, to obtain global information of point clouds in an efficient way. GA-Net consists of a point-independent global attention module, and a point-dependent global attention module.

[[PDF](https://ieeexplore.ieee.org/document/9439963)]

### Overview

This repository is the author's re-implementation of GA-Net. Extensive experiments on three point cloud semantic segmentation benchmarks demonstrate that GA-Net outperforms state-of-the-art methods in most cases.

<img src='./misc/architecture.png' width="1321" alt="architexture">

Further information please contact [Shuang Deng](https://ds0529.github.io/) and [Qiulei Dong](http://vision.ia.ac.cn/Faculty/qldong/index.htm).

### Citation

Please cite this paper if you want to use it in your work:

    @article{2021ganet,
	title={GA-NET: Global Attention Network for Point Cloud Semantic Segmentation}, 
	author={Deng, Shuang and Dong, Qiulei},
	journal={IEEE Signal Processing Letters (SPL)}, 
	volume={28},
	pages={1300-1304}, 
	year={2021},
    doi={10.1109/LSP.2021.3082851}
    }
	
### Setup

Setup python environment:

```
conda create -n ganet python=3.6
source activate ganet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### Semantic3D

Download and extract the Semantic3D dataset:

```
sh utils/download_semantic3d.sh
```

Prepare the Semantic3D dataset:

```
python utils/data_prepare_semantic3d.py
```

Train:

```
python main_Semantic3D.py --gpu $your_gpu_id --mode 'train'
```

Evaluation:

```
python main_Semantic3D.py --gpu $your_gpu_id --mode 'test'
```

The trained model is stored in the folder `result/ganet/Log_2020-10-09_Semantic3D_1`.

### S3DIS

Download the S3DIS dataset from <a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here (4.09GB)</a>. Uncompress the folder and move it to 
`/your_data_folder/S3DIS`.

Prepare the S3DIS dataset:

```
python utils/data_prepare_s3dis.py
```

Train:

```
python main_S3DIS.py --model 'GANet' --test_area 5 --gpu $your_gpu_id --mode 'train'
```

Test:

```
python main_S3DIS.py --model 'GANet' --test_area 5 --gpu $your_gpu_id --mode 'test'
```

Calculate the final mean IoU results:

```
python utils/area_5_cv.py
```

The trained model is stored in the folder `result/ganet/Log_2020-10-19_S3DIS_Area_5`.

### ScanNet

Download the ScanNet dataset from <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">here (1.72GB)</a>. Uncompress the folder and move it to `/your_data_folder/scannet`.

Prepare the ScanNet dataset:

```
python utils/data_prepare_scannet.py
```

Train:

```
python main_ScanNet.py --model 'GANet' --gpu $your_gpu_id --mode 'train'
```

Test:

```
python main_ScanNet.py --model 'GANet' --gpu $your_gpu_id --mode 'test'
```

The trained model is stored in the folder `result/ganet/Log_2020-10-09_ScanNet_1`.

### Acknowledgement

The structure of this codebase is borrowed from [RandLA-Net](https://github.com/QingyongHu/RandLA-Net).