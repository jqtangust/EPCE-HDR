# EPCE-HDR
This is a pytorch project for the paper **High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation** by Jiaqi Tang, Xiaogang Xu, Sixing Hu and Ying-Cong Chen presented at **ECAI2023**.

## Introduction
Due to limited camera capacities, digital images usually have a narrower dynamic illumination range than real-world scene radiance. To resolve this problem, High Dynamic Range (HDR) reconstruction is proposed to recover the dynamic range to better represent real-world scenes. However, due to different physical imaging parameters, the tone-mapping functions between images and real radiance are highly diverse, which makes HDR reconstruction extremely challenging. Existing solutions can not explicitly clarify a corresponding relationship between the tone-mapping function and the generated HDR image, but this relationship is vital when guiding the reconstruction of HDR images. To address this problem, we propose a method to explicitly estimate the tone mapping function and its corresponding HDR image in one network. Firstly, based on the characteristics of the tone mapping function, we construct a model by a polynomial to describe the trend of the tone curve. To fit this curve, we use a learnable network to estimate the coefficients of the polynomial. This curve will be automatically adjusted according to the tone space of the Low Dynamic Range (LDR) image, and reconstruct the real HDR image. Besides, since all current datasets do not provide the corresponding relationship between the tone mapping function and the LDR image, we construct a new dataset with both synthetic and real images. Extensive experiments show that our method generalizes well under different tone-mapping functions and achieves SOTA performance.

[Paper link (Axriv)](https://arxiv.org/abs/2307.16426)

## Code specification

TODO

## Usage

TODO

#### Evaluation

TODO

## Citation Information

If you find the project useful, please cite:

```
@inproceedings{xu2022scene2graph,
  title={Hierarchical Image Generation via Transformer-Based Sequential Patch Selection},
  author={Xiaogang Xu, Ning Xu},
  booktitle={AAAI},
  year={2022}
}
```
