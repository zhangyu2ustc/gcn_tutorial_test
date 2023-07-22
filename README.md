neurolibre : [![Binder](http://binder-wksh2.conp.cloud/badge_logo.svg)](http://binder-wksh2.conp.cloud/v2/gh/zhangyu2ustc/gcn_tutorial_test/master?filepath=notebooks%2F)
public binder : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zhangyu2ustc/gcn_tutorial_test/master?filepath=notebooks%2F)

# GCN_tutorial
this tutorial covers the basics of _**graph laplacian**_ and _**graph convolutional networks**_ and how to apply these tools to neuroimging data

## Introduction
Brain graphs provide a relatively simple way of modeling the human brain connectome, by associating nodes with brain regions, and defining edges via anatomical or functional connections. 
![Optimizing brain graphs and GNN for brain decoding](data/脑图谱的选择与解码.png)
#### Graph Laplacian
> Based on this architecture, a non-linear embedding tool, called graph Laplacian, can be used to project the high dimensional brain activities onto subspaces of the graph Laplacian eigenbasis.
This method has gained more and more attention in neuroscience studies, for instance identifying functional areas and networks, generating connectivity gradients and harmonics, and even predicting atrophy patterns of dementia. 
#### Graph Convolutional Networks
> Recently, graph convolutional networks (GCN) was proposed, which combines the graph Laplacian theory with deep learning architectures by extending convolution operations onto the graph domain. 
> This approach has shown some promising findings in neuroscience applications, for instance parcellating brain areas and detecting alterations in AD and Autism. 

> In our recent study, we applied GCN[1,2] to annotate the spatiotemporal dynamics of brain dynamics of human cognitive functions using a short series of fMRI volumes. 
I will use this as a case study to illustrate how to apply GCN to brain imaging.



\
\

## Code
 * ```notebooks``` includes all the functions and modules you need for this tutorial

 * ```notebooks/Tutorials_GCN_practice2_graph-Laplacian_GCN.ipynb``` the main notebook

 * ```notebooks/model.py``` contains the model definition, including fully-connected, 1stGCN [1] and ChebyNet [2]

 * ```notebooks/utils.py``` contains helpful functions

## Setup
### Prepare the environment
 You could either clone the **github repo** or directly using the enviorment prebuild through **binder**

  * open a terminal and type:
```
git clone https://github.com/zhangyu2ustc/gcn_tutorial_test.git
``` 

 * Or Click the link to [binder](https://mybinder.org/v2/gh/zhangyu2ustc/gcn_tutorial_test/master?filepath=notebooks%2F) in your browser


### Check the notebook 
  Go to notebook folder and find the .ipynb [notebook](https://github.com/zhangyu2ustc/gcn_tutorial_test/blob/master/notebooks/Tutorials_GCN_practice2_graph-Laplacian_GCN.ipynb)

  Note: It could take a while for the binder to build the enviornment (around 5 minutes). Be patient !  




\
\
## Slides
 check the presentation slides here: GCN_tutorial_slides [@MAIN2019](https://drive.google.com/file/d/1Gu28WcHXlwjXQSSmqZZwIcESHff_j-J4/view?usp=sharing)    [@OHBM2020](https://drive.google.com/file/d/1mBl_nKBRm1peIu3ouz6kqkMlWJpJQw6V/view?usp=sharing) [@OHBM2023](https://drive.google.com/file/d/1W3So2MxNS9c3wR6E-zRm9LV0-prmVuPR/view?usp=drive_link)

## References
<a id="1">[1]</a> Zhang, Yu, and Pierre Bellec. "Functional Decoding using Convolutional Networks on Brain Graphs." 2019 Conference on Cognitive Computational Neuroscience, Berlin, Germany [PDF](https://ccneuro.org/2019/proceedings/0001137.pdf)

<a id="2">[2]</a> Zhang, Yu, and Pierre Bellec. "Functional Annotation of Human Cognitive States using Graph Convolution Networks." 2019 Conference on Neural Information Processing Systems (NeurIPS) Neuro-AI workshop - Real Neurons & Hidden Units, Vancouver, Canada [PDF](https://openreview.net/pdf?id=HJenmmF8Ir)

<a id="3">[3]</a> Yu Zhang, Loïc Tetrel, Bertrand Thirion, and Pierre Bellec. 2021. “Functional Annotation of Human Cognitive States Using Deep Graph Convolution.” NeuroImage 231 (May): 117847. [PDF](https://doi.org/10.1016/j.neuroimage.2021.117847)

<a id="4">[4]</a> Yu Zhang, Nicolas Farrugia and Pierre Bellec. 2022. “Deep learning models of cognitive processes constrained by human brain connectomes.” Medical Image Analysis 80 (August): 102507. [PDF](https://doi.org/10.1016/j.media.2022.102507)
