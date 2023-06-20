
 
<div align="center">

<!-- TITLE -->
# **Test-time Adaptation with Slot-Centric Models**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2203.11194-b31b1b.svg)](https://arxiv.org/abs/2203.11194)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://slot-tta.github.io/)
</div>

<img src="images/slot-tta_model_fig.png" alt="Model figure"/>


<!-- DESCRIPTION -->
## Abstract
Current supervised visual detectors, though impressive within their training distribution, often fail to segment out-of-distribution scenes into their constituent entities. Recent test-time adaptation methods use auxiliary self-supervised losses to adapt the network parameters to each test example independently and have shown promising results towards generalization outside the training distribution for the task of image classification. In our work, we find evidence that these losses can be insufficient for instance segmentation tasks, without also considering architectural inductive biases. For image segmentation, recent slot-centric generative models break such dependence on supervision by attempting to segment scenes into entities in a self-supervised manner by reconstructing pixels. Drawing upon these two lines of work, we propose Slot-TTA, a semi-supervised instance segmentation model equipped with a slot-centric inductive bias, that is adapted per scene at test time through gradient descent on reconstruction or novel view synthesis objectives. We show that test-time adaptation in Slot-TTA greatly improves instance segmentation in out-of-distribution scenes. We evaluate Slot-TTA in several 3D and 2D scene instance segmentation benchmarks and show substantial out-of-distribution performance improvements against state-of-the-art supervised feed-forward detectors and self-supervised test-time adaptation methods.

## Code
Below we opensource the code for our 2D RGB experiments, we'll soon opensource the code for our pointcloud experiments.
### Installation
We use Pytorch 2.0 for all our experiments. We use [wandb](https://wandb.ai/) for logging the results.
Install conda (instructions [here](https://docs.conda.io/en/latest/miniconda.html)).
Create a conda environment using the below commands:
```
conda create -n slot_tta python=3.8.5
conda activate slot_tta
```
Install the required pip packages using the below command:
```
pip install -r requirement.txt
```
### Dataset

For 2D RGB experiments, we use CLEVR and CLEVR-Tex as our train-test splits respectively. You can download a small set (1000 examples) of Clevr dataset from [here](https://drive.google.com/file/d/1SJTeu51qMAa6nfeLfRYyks_o6T0PuFiP/view?usp=sharing)).
Rename the folder to `clevr_train` for debugging.

```mv clevr_train_small clevr_train```

You can find the full ClevrTex dataset we used for testing [here](https://drive.google.com/file/d/1KRhnVG05fj3jfKtcq6bdqGCAa5_lYCau/view?usp=sharing):
Download, unzip and update the `root_folder` variable, found [here](https://github.com/mihirp1998/Slot-TTA/blob/932ed4fa7b66e93898e4fcc5bc5fd99a32851438/config/config.yaml#L23) in the code accordingly.

We'll soon opensource the full CLEVR dataset we use for training, note that we're simply using the CLEVR dataset found [here](https://github.com/deepmind/multi_object_datasets#clevr-with-masks), by reformatting it from tfrecords to pickled files. 

### Training Code

```
python main.py +experiment=clevr_train
```


### Test-Time Adaptation Code

We provide the pre-trained model of our 2D CLEVR experiment over [here](https://github.com/mihirp1998/Slot-TTA/checkpoint/clevr_train/checkpoint.pt)
In order to load with your own checkpoint, simply update the `load_folder` variable as shown below.
For intermediate TTA result visualization, set `deep_tta_vis` variable to True

```
python main.py +experiment=clevrtex_tta load_folder=checkpoint/clevr_train/checkpoint.pt
```


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
