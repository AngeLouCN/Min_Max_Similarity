# Min_Max_Similarity
A contrastive learning based semi-supervised segmentation network for medical image segmentation
This repository contains the implementation of a novel contrastive learning based semi-segmentation networks to segment the surgical tools.
<div align=center><img src="https://github.com/AngeLouCN/Min_Max_Similarity/blob/main/img/mms.jpg" width="1000" height="450" alt="Result"/></div>
<p align="center"><b>Fig. 1. The architecture of Min-Max Similarity.</b></p>

**:fire: NEWS :fire:**
**The full paper is available:** [Min-Max Similarity](https://arxiv.org/abs/2203.15177)

## Environment

- python==3.6
- packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
```
conda install opencv-python pillow numpy matplotlib
```
- Clone this repository
```
git clone https://github.com/AngeLouCN/Min_Max_Similarity
```
## Data Preparation

We use three dataset to test its performance:
- Kvasir-instrument
- EndoVis'17
- Cochlear Implant

**File structure**
```
|-- data
|   |-- kvasir
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask
|   |-- EndoVis17
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask
|   |-- cochlear
|   |   |-- train
|   |   |   |--image
|   |   |   |--mask
|   |   |-- test
|   |   |   |--image
|   |   |   |--mask
```

**You can also test on some other public medical image segmentation dataset with above file architecture**

## Usage

- **Training:**
You can change the hyper-parameters like labeled ratio, leanring rate, and e.g. in ```train_mms.py```, and directly run the code.

- **Testing:**
You can change the dataset name in ```test.py``` and run the code.

## Segmentation Performance
<div align=center><img src="https://github.com/AngeLouCN/Min_Max_Similarity/blob/main/img/result_vis.jpg" width="650" height="550" alt="Result"/></div>
<p align="center"><b>Fig. 2. Visual segmentation results.</b></p>
<div align=center><img src="https://github.com/AngeLouCN/Min_Max_Similarity/blob/main/img/result.jpg" width="800" height="350" alt="Result"/></div>
<p align="center"><b>Table 1. Segmentation results.</b></p>

## Citation
```coming soon```

