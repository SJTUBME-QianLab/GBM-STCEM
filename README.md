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

Differentiation between pseudoprogression and true tumor progression of glioblastoma (GBM) is crucial for choosing appropriate management strategies and increasing the chances of patient survival. Currently, there is a lack of non-invasive and effective methods in clinic for GBM progression diagnosis. Here, we propose an automated early diagnosis method based on diffusion tensor imaging (DTI) with a high potential for this diagnosis. A primary challenge for intelligent diagnostic methods lies in the limited accuracy and stability caused by data insufficiency and the fine-grained nature of diagnostic tasks. To address this challenge, we develop a spatial-transformation-based causality-enhanced model (ST-CEM). This model jointly improves data diversity and the effective utilization of clinically significant discriminative information. Specifically, first, a texture diverse augmentation scheme is designed based on a spatial transformation, which allows for greater texture diversification in the augmented data. Subsequently, an interference information contrastive strategy is developed, where non-lesion features that may introduce interference are actively extracted and decoupled with lesion features. Finally, a causality-enhanced mechanism is introduced to highlight the decoupled lesion features, and thereby improve the diagnostic stability of the model. Extensive experiments verified the effectiveness of our model in diagnosis of GBM progression under small-sample conditions. The proposed model achieved an accuracy of 84.1%, precision of 85.8%, and recall of 90.3%, all of which outperform the existing works. Moreover, it demonstrated competitive performance on an additional lung nodule classification dataset. Our source codes have been released at https://github.com/SJTUBME-QianLab/GBM-STCEM.


### Required

- Python3.7
- PyTorch1.10.1 (http://pytorch.org/)

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
- The TDAS details can be found in ```get_all_data.py```
- The IICS and CEM details can be found in ```torch_cnn.py```
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
