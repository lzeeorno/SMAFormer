# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
@article{zheng2024smaformer,
  title={SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation},
  author={Zheng, Fuchen and Chen, Xuhang and Liu, Weihuang and Li, Haolun and Lei, Yingtie and He, Jiahui and Pun, Chi-Man and Zhou, Shounjun},
  journal={arXiv preprint arXiv:2409.00346},
  year={2024}
}
```
# 📋SMAFormer

SMAFormer: Synergistic Multi-Attention Transformer for Medical Image Segmentation
[Vedio introduction](https://www.bilibili.com/video/BV1FLDsYqExZ/)

[Fuchen Zheng](https://lzeeorno.github.io/),  [Xuhang Chen](https://cxh.netlify.app/), Weihuang Liu, Haolun Li, Yingtie Lei, Jiahui He, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮and [Shoujun Zhou](https://people.ucas.edu.cn/~sjzhou?language=en) 📮( 📮 Corresponding authors)

**University of Macau, SIAT CAS, Huizhou University, University of Nottingham Ningbo China**

2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2024)

## 🚧 Installation 
Requirements: `Ubuntu 20.04`

1. Create a virtual environment: `conda create -n your_environment python=3.8 -y` and `conda activate your_environment `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) :`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`
Or you can use Tsinghua Source for installation
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
3. `pip install tqdm scikit-learn albumentations==1.0.3 pandas einops axial_attention`
4. `pip install xlsxwriter`
5. `python train_lits2017_png.py`


## 1. Prepare the dataset

### LiTS2017 datasets
- The LiTS2017 dataset can be downloaded here: {[LiTS2017](https://competitions.codalab.org/competitions/17094)}.
- The Synapse dataset can be downloaded here: {[Synapse multi-organ](https://www.synapse.org/Synapse:syn3193805/wiki/217789)}.

- After downloading the datasets, you should run ./data_prepare/preprocess_lits2017_png.py to convert .nii files into .png files for training. (Save the downloaded LiTS2017 datasets in the data folder in the following format.)

- './data_prepare/'
  - preprocess_lits2017_png.py
- 'net'
  - SMAFormer_LiTS.py
  - SMAFormer_Synapse.py
- './data/'
  - LITS2017
    - ct
      - .nii
    - label
      - .nii
  - trainImage_lits2017_png
      - .png
  - trainImage_lits2017_png
      - .png

### Other datasets
- Other datasets just similar to LiTS2017

## 2. Prepare the pre_trained weights
- The weights of the pre-trained SMAFormer could be downloaded from [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth).


## 3. Train the SMAFormer
```bash
cd ./
python train_synapse.py

python train_lits2017.py 
```

## 4. Train the SMAFormer
```bash
cd ./
python test_synapse.py

python test_lits2017.py 
```

## 5. Obtain the outputs
- After trianing, you could obtain the results in './results/..'
- - After testing, you could obtain the results in './test_result/..'

  

# 🧧 Acknowledgement

This work was supported in part by the National Key R\&D Project of China (2018YFA0704102, 2018YFA0704104), in part by Natural Science Foundation of Guangdong Province (No. 2023A1515010673), and in part by Shenzhen Technology Innovation Commission (No. JSGG20220831110400001), in part by Shenzhen Development and Reform Commission (No. XMHT20220104009), in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3.


