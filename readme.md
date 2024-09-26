# Polarimetry-inspired Contrastive Learning for Class-imbalanced PolSAR Image Classification

## Abstract:

In recent years, deep neural networks have significantly boosted the performance of polarimetric synthetic aperture radar (PolSAR) image classification. However, existing deep learning-based approaches still suffer from the following limitations. First, the performance of them is subject to the availability of massive annotations that are difficult to acquire for PolSAR images. Second, the class imbalance in PolSAR data greatly hinders the correct classification of minority yet equally pivotal classes. To overcome the above shortcomings, we propose a polarimetry-inspired contrastive learning (CL) PolSAR image classification (PiCL) approach, in the hope of elevating the classification accuracy by taking advantage of the polarimetric domain knowledge. First, a complex-valued CL (CVCL) framework is designed, via which powerful polarimetric representations are learned without any manual annotations. Specifically, we innovatively design two distribution-inspired positive sample generation (PSG) strategies, i.e., Wishart-distance-based PSG (WishartPSG) and noise-injection PSG (NoisePSG), to enable discriminative and domain-specific representation learning. A novel hybrid anti-imbalance scheme is further devised to tackle the class imbalance issue, which combines a contextual consistency-based pseudo-label generation (PLG) and a weighted feature-level synthetic data oversampling technique. It should be highlighted that the domain knowledge of PolSAR, including the data and noise distributions, complex-valued (CV) characteristics, and the spatial consistency prior, is fully exploited throughout our model design. Extensive experiments on four benchmark datasets demonstrated the effectiveness of the proposed model. For the Flevoland 1989 dataset, our method improves the overall accuracy (OA), average accuracy (AA), and Kappa metrics by 3.54%, 6.81%, and 7.29%, respectively, compared to the existing state-of-the-art method. 


## Dependencies and Installation
```bash
conda create -n picl python=3.7
conda activate picl
pip install -r requirements.txt
```

## Usage
### Pretraining

```
python pretrain/pretrain.py
```

### Fine-tuning

```
python fine_tune/main.py
```

The trained networks and results will be saved at `trained_model`. 

## Contact
If you have any question, please email `794466014@qq.com`

## Citation
```
@article{kuang2024polarimetry,
  title={Polarimetry-inspired Contrastive Learning for Class-imbalanced PolSAR Image Classification},
  author={Kuang, Zuzheng and Bi, Haixia and Li, Fan and Xu, Chen and Sun, Jian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  volume={62},
  number={},
  pages={1-19},
  publisher={IEEE},
  doi={10.1109/TGRS.2024.3403100}
}
```
