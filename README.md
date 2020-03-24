# High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks 
**[H. Wang, X. Wu, Z. Huang, and E. P. Xing. "High frequency component helps explain the generalization of convolutional neural networks." CVPR 2020 (Oral).](https://arxiv.org/abs/1905.13545)**

## Highlights
|The central hypothesis of our paper: within a data collection, there are correlations between the highfrequency components and the “semantic” component of the images. As a result, the model will perceive both high-frequency components as well as the “semantic” ones, leading to generalization behaviors counterintuitive to human (e.g., adversarial examples).|<img src="main.png" alt="main hypothesis of the paper" width="1600" height="whatever">  
 |:--|---|

<img src="intro.png" alt="HFC helps explain CNN generaliation" width="1000" height="whatever">

**Eight testing samples selected from CIFAR10 that help explain that CNN can capture the high-frequency image: the model (ResNet18) correctly predicts the original image (1st column in each panel) and the highfrequency reconstructed image (3rd column in each panel), but incorrectly predict the low-frequency reconstructed image (2nd column in each panel). The prediction confidences are also shown. Details are in the paper.**

### Code Structures

### Before using the code
