# References

This document outlines the methodological grounding and original papers relevant to the components utilized in this project.

## 1. 3D Face Reconstruction
**Method:** Deep3DFaceRecon_pytorch
**Paper:** Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. "Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set." IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), 2019.
**URL:** [https://arxiv.org/abs/1903.08527](https://arxiv.org/abs/1903.08527)
**Relevance:** This project uses the `Deep3DFaceRecon_pytorch` implementation to extract a 3D Morphable Model (3DMM) surface mesh from a monocular image. This model outputs the initial geometry from which we extract depth, gradient, and curvature features.

## 2. Face Parsing and Segmentation
**Method:** BiSeNet
**Paper:** Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation." European Conference on Computer Vision (ECCV), 2018.
**URL:** [https://arxiv.org/abs/1808.00897](https://arxiv.org/abs/1808.00897)
**Relevance:** We use a pretrained BiSeNet model (`face-parsing.PyTorch`) to parse the face, specifically to isolate facial regions and define non-face (occlusion) pixels for the inpainting step.

## 3. Image Inpainting
**Method:** Telea Inpainting (OpenCV)
**Paper:** Alexandru Telea. "An Image Inpainting Technique Based on the Fast Marching Method." Journal of Graphics Tools, 2004.
**URL:** [https://www.semanticscholar.org/paper/An-Image-Inpainting-Technique-Based-on-the-Fast-Telea/67f0bfcbab3a38ea39c43d94ff54df81f7c13fb5](https://www.semanticscholar.org/paper/An-Image-Inpainting-Technique-Based-on-the-Fast-Telea/67f0bfcbab3a38ea39c43d94ff54df81f7c13fb5)
**Relevance:** We apply Telea's algorithm via `cv2.inpaint` to fill in occluded or non-face regions detected by BiSeNet before passing the image to the 3D reconstruction model.

## 4. Face Anti-Spoofing (PAD) Motivation
**Paper:** Zezheng Wang, Zitong Yu, Chenxu Zhao, Xiangyu Zhu, Yunxiao Qin, Qiusheng Zhou, Feng Zhou, and Zhen Lei. "Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing." IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
**URL:** [https://arxiv.org/abs/2003.08061](https://arxiv.org/abs/2003.08061)
**Relevance:** Provides the foundational motivation for utilizing spatial depth and gradient information to detect presentation attacks. *Note: This project does not directly reproduce their deep multi-frame RSGB/STPM architecture, but draws inspiration to evaluate explicitly extracted 3D mesh geometric features (depth, gradient, curvature) for Presentation Attack Detection (PAD).*
