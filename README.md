
 
<div align="center">

<!-- TITLE -->
# **Test-time Adaptation with Slot-Centric Models**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2203.11194-b31b1b.svg)](https://arxiv.org/abs/2203.11194)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://mihirp1998.github.io/project_pages/slottta/)
</div>


<!-- DESCRIPTION -->
## Abstract
Current supervised visual detectors, though impressive within their training distribution, often fail to segment out-of-distribution scenes into their constituent entities. Recent test-time adaptation methods use auxiliary self-supervised losses to adapt the network parameters to each test example independently and have shown promising results towards generalization outside the training distribution for the task of image classification. In our work, we find evidence that these losses can be insufficient for instance segmentation tasks, without also considering architectural inductive biases. For image segmentation, recent slot-centric generative models break such dependence on supervision by attempting to segment scenes into entities in a self-supervised manner by reconstructing pixels. Drawing upon these two lines of work, we propose Slot-TTA, a semi-supervised instance segmentation model equipped with a slot-centric inductive bias, that is adapted per scene at test time through gradient descent on reconstruction or novel view synthesis objectives. We show that test-time adaptation in Slot-TTA greatly improves instance segmentation in out-of-distribution scenes. We evaluate Slot-TTA in several 3D and 2D scene instance segmentation benchmarks and show substantial out-of-distribution performance improvements against state-of-the-art supervised feed-forward detectors and self-supervised test-time adaptation methods.

## Code
Coming soon!


<!-- CITATION -->
## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{prabhudesai2022test,
title={Test-time Adaptation with Slot-Centric Models},
author={Prabhudesai, Mihir and Goyal, Anirudh and Paul, Sujoy and van Steenkiste, Sjoerd and Sajjadi, Mehdi SM and Aggarwal, Gaurav and Kipf, Thomas and Pathak, Deepak and Fragkiadaki, Katerina},
journal={arXiv preprint arXiv:2203.11194},
year={2022}
}
```
