# Infusion-Net Unofficial implementation in PyTorch

Implementation of ["Infusion-Net: Inter- and Intra-Weighted Cross-Fusion Network for Multispectral Object Detection"](https://www.mdpi.com/2227-7390/10/21/3966) in Special Issue [Advances in Pattern Recognition and Image Analysis](https://www.mdpi.com/journal/mathematics/special_issues/Pattern_Recognition_Image_Analysis) of _Mathematics_ **2022**, 10(21), 3966; https://doi.org/10.3390/math10213966

## Abstract

Object recognition is conducted using red, green, and blue (RGB) images in object recognition studies. However, RGB images in low-light environments or environments where other objects occlude the target objects cause poor object recognition performance. In contrast, infrared (IR) images provide acceptable object recognition performance in these environments because they detect IR waves rather than visible illumination. In this paper, we propose an inter- and intra-weighted cross-fusion network (Infusion-Net), which improves object recognition performance by combining the strengths of the RGB-IR image pairs. Infusion-Net connects dual object detection models using a high-frequency (HF) assistant (HFA) to combine the advantages of RGB-IR images. To extract HF components, the HFA transforms input images into a discrete cosine transform domain. The extracted HF components are weighted via pretrained inter- and intra-weights for feature-domain cross-fusion. The inter-weighted fused features are transmitted to each other’s networks to complement the limitations of each modality. The intra-weighted features are also used to enhance any insufficient HF components of the target objects. Thus, the experimental results present the superiority of the proposed network and present improved performance of the multispectral object recognition task.

Keywords: multispectral object detection; inter- and intra-weighted fusion; high-frequency component; discrete cosine transform

## Datasets

LLVIP: A Visible-infrared Paired Dataset for Low-light Vision [link](https://bupt-ai-cz.github.io/LLVIP/)

FLIR: [link](https://www.flir.in/oem/adas/adas-dataset-form/)

## Notes

This is an implementation based on reading the paper. There are no guarantees that the provided code is exactly what the authors used in the paper. For official implementation contact the authors

## Contributing

Feel free to contribute by making a pull request. Any contribution or feedback is encouraged.

