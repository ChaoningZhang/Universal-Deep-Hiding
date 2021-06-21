# UDH: Universal Deep Hiding for Steganography, Watermarking, and Light Field Messaging
This is the repository for the NeurIPS 2020 paper titled [UDH: Universal Deep Hiding for Steganography, Watermarking, and Light Field Messaging](https://papers.nips.cc/paper/2020/file/73d02e4344f71a0b0d51a925246990e7-Paper.pdf).

## Abstract
Neural networks have been shown effective in deep steganography for hiding a full image in another. However, the reason for its success remains not fully clear. Under the existing cover (*C*) dependent deep hiding (DDH) pipeline, it is challenging to analyze how the secret (*S*) image is encoded since the encoded message cannot be analyzed independently. We propose a novel universal deep hiding (UDH) meta-architecture to disentangle the encoding of *S* from *C*. We perform extensive analysis and demonstrate that the success of deep steganography can be attributed to a frequency discrepancy between *C* and the encoded secret image. Despite *S* being hidden in a cover-agnostic manner, strikingly, UDH achieves a performance comparable to the existing DDH. Beyond hiding one image, we push the limits of deep steganography. Exploiting its property of being *universal*, we propose universal watermarking as a timely solution to address the concern of the exponentially increasing amount of images/videos. UDH is robust to a pixel intensity shift on the container image, which makes it suitable for challenging application of light field messaging (LFM). This is the first work demonstrating the success of (DNN-based) hiding a full image for watermarking and LFM.

## Setup 
We performed our experiments with `PyTorch v.0.4.1` for the main UDH, DDH, and watermarking, and `PyTorch v.1.0.0` as well as `torchgeometry v.0.1.2` for LFM.

### Datasets 
#### ImageNet
 1. Follow the common setup to make ImageNet compatible with pytorch as described in [here](https://github.com/pytorch/examples/tree/master/imagenet).
 2. Set the path to the pytorch ImageNet dataset folder in the main file.

## Experiments
### Train
To train the main UDH model, run the script `bash ./scripts/train_main_udh.sh`.
To train the main DDH model, run the script `bash ./scripts/train_main_ddh.sh`.
To train the UDH model for watermarking, run the script `bash ./scripts/train_watermarking.sh`.
To train the UDH model for LFM, run the script `bash ./scripts/train_lfm.sh`.

Weights and other details of the training instances are saved into the `./training/` folder.

### Test
To get the main qualitative and quantitative results, run the script `bash ./scripts/test_main.sh`.
To get the results for watermarking, run the script `bash ./scripts/test_watermarking.sh`.

For test scripts, training folders should be specified with the argument `--test`. 
For DDH training folder in the `test_main.sh` script, the argument is `--test_diff`.

Utility code is provided in the `./util/` folder for LFM processing: `align_images.py` and `cr2png.py`.

## Pre-trained weights
Pre-trained weights for different models can be accessed [here](https://cloud.frameau.xyz/index.php/s/dFGagRWystSss7D). 
Download the folder `training` and put it inside the project folder. Now the test scripts can be run without training.

*Update 21.06.21*: LFM, hiding 6 secret images in 3 cover images, hiding 2 colored images in 1 grayscale image, and hiding with multiple encoders and decoders weights are availabe [here](https://cloud.frameau.xyz/index.php/s/YDowXx5L8LqTaoM). These weights were retrained for your convenience (not original weights used in the paper).

## JPEG
Code for the pseudo-differentiable JPEG used for watermarking is available [here](https://github.com/ChaoningZhang/Pseudo-Differentiable-JPEG).

## License & Disclaimer
This code is strictly for non-commercial academic use only. 

## Citation
```
@inproceedings{zhang2020udh,
  title={UDH: Universal Deep Hiding for Steganography, Watermarking, and Light Field Messaging},
  author={Zhang, Chaoning and Benz, Philipp and Karjauv, Adil and Sun, Geng and Kweon, In-So},
  booktitle={34th Conference on Neural Information Processing Systems, NeurIPS 2020},
  year={2020},
  organization={Conference on Neural Information Processing Systems}
}
```
