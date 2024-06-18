This repository holds the PyTorch code for the paper
# A Spatial-Transformation-Based Causality-Enhanced Model for Glioblastoma Progression Diagnosis

By [Qiang Li](https://faculty.hdu.edu.cn/txgxxy/lq2/main.htm), [Xinyue Li](https://mihi.sjtu.edu.cn/lxy.html), [Hong Jiang](https://www.scholarmate.com/P/iErui2), [Xiaohua Qian](https://bme.sjtu.edu.cn/Web/FacultyDetail/46).

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

### Table of Contents
0. [Abstract](#Abstract)
0. [Required](#Required)
0. [Setting up](#Setting-up)
0. [Trainning](#Trainning)
0. [Citation](#Citation)
0. [Contact](#Contact)

### Abstract

Differentiation between pseudoprogression and true tumor progression of glioblastoma multiforme (GBM) is crucial for choosing appropriate postoperative management strategies and increasing the overall chances of patient survival. However, the current diagnostic methods, such as pathological examination and imaging-based follow-up, have inherent limitations including invasiveness and treatment delays. Hence, we developed a spatial-transformation-based causality-enhanced model (ST-CEM), which jointly achieves better data diversity and more effective use of clinical significance information. Specifically, first, a texture diverse augmentation scheme was designed based on a spatial transformation to improve the diversification of the augmented training data. Subsequently, an interference information contrastive strategy was introduced through which interference image features are actively constructed and decoupled with lesion features. Finally, a causality-enhanced mechanism was developed to promote the causal lesion features, and thereby improve the diagnostic stability of the model. Extensive experiments substantiate the effectiveness of our model in the accurate diagnosis of GBM progression under small-sample conditions. The model achieves an accuracy of 84.1%, precision of 85.8%, and recall of 90.3%, all of which outperform the existing works. Moreover, it demonstrates competitive performance on an additional lung nodule classification dataset. Our source codes will be released at https://github.com/SJTUBME-QianLab/GBM-STCEM.


### Required

Our code is based on **Python3.7** There are a few dependencies to run the code. The major libraries we depend are
- PyTorch1.10.1 (http://pytorch.org/)
- tensorboardX 2.5
- numpy 
- tqdm 

### Setting up
```
pip install -r requirements.txt
```
Attention:
Please run this project on linux.
In different pytorch environment, the model may obtain different results. 

### Trainning
Run the ```st_cem.py``` by this command:
```
python st_cem.py
```
The detailed parameters can be changed in ```st_cem.py``` 

### Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = {CA Spatial-Transformation-Based Causality-Enhanced Model for Glioblastoma Progression Diagnosis},
  author    = {Qiang Li, Xinyue Li, Hong Jiang, Xiaohua Qian*},
  year      = {2024},
}
```

### Contact
For any question, feel free to contact
```
Qiang Li : lq_1929@sjtu.edu.cn
```
