# EPCE-HDR (ECAI 2023 Oral) 
This is a pytorch project for the paper **High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation** 

by Jiaqi Tang, Xiaogang Xu, Sixing Hu and Ying-Cong Chen presented at **ECAI2023**.

**All Related Materials are Perparing.**

## Introduction
Due to limited camera capacities, digital images usually have a narrower dynamic illumination range than real-world scene radiance. To resolve this problem, High Dynamic Range (HDR) reconstruction is proposed to recover the dynamic range to better represent real-world scenes. However, due to different physical imaging parameters, the tone-mapping functions between images and real radiance are highly diverse, which makes HDR reconstruction extremely challenging. Existing solutions can not explicitly clarify a corresponding relationship between the tone-mapping function and the generated HDR image, but this relationship is vital when guiding the reconstruction of HDR images. To address this problem, we propose a method to explicitly estimate the tone mapping function and its corresponding HDR image in one network. Firstly, based on the characteristics of the tone mapping function, we construct a model by a polynomial to describe the trend of the tone curve. To fit this curve, we use a learnable network to estimate the coefficients of the polynomial. This curve will be automatically adjusted according to the tone space of the Low Dynamic Range (LDR) image, and reconstruct the real HDR image. Besides, since all current datasets do not provide the corresponding relationship between the tone mapping function and the LDR image, we construct a new dataset with both synthetic and real images. Extensive experiments show that our method generalizes well under different tone-mapping functions and achieves SOTA performance.

[Paper link (Axriv)](https://arxiv.org/abs/2307.16426)

[Poster](ECAI_Poster.pdf)

[Oral (PPT)](1024HighDynamicRange.pdf)

[Oral (Video)]()

## Contributions
If you have any questions, feel free to e-mail the author Jiaqi Tang ([jtang092@connect.ust.hk](jtang092@connect.ust.hk)).

## Dataset

TODO

## Code specification

TODO

## Usage

TODO

## Evaluation

TODO

## Citation Information

If you find the project useful, please cite:

```
@inproceedings{Tang2023HighDR,
  title={High Dynamic Range Image Reconstruction via Deep Explicit Polynomial Curve Estimation},
  author={Jiaqi Tang and Xiaogang Xu and Sixing Hu and Yingke Chen},
  booktitle={European Conference on Artificial Intelligence},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:260334575}
}
```
