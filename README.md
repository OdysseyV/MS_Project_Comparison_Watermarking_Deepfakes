# MS_Project_Comparison_Watermarking_Deepfakes
### Report: https://github.com/OdysseyV/MS_Project_Comparison_Watermarking_Deepfakes/blob/main/Villagomez_Odyssey_MS_Final_Report.pdf

### Abstract
Fake imaging and videos have exploded into popularity over the past 5 years
since the release of the first deepfake video in 2017. These deepfake media increasingly
pose a threat to society by questioning the integrity and authenticity of images. This
project compares two existing watermarking techniques that have developed to not only
detect deepfake images, but to also defend against them proactively This study compares
the two watermarking methods to test which performs better on image manipulation
detection as well as their ability to prevent deepfake images. The first study, “Neekhara,
et al., FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and
Countering Deepfakes” had an AUC score of 0.9992, when classifying real or fake
images, while the second study, “CMUA-Watermark: A Cross-Model Universal
Adversarial Watermark for Combating Deepfakes” had an AUC score of 0.5. The first
study demonstrates how a pre-trained watermark encoder-decoder model can detect
deepfake images with high accuracy. This mechanism allows for anyone to recognize
fake or real imaging by looking for a watermark.

### References: Code based on: 

#### FaceSigns Watermark: https://github.com/paarthneekhara/FaceSignsDemo
@article{facesigns2022,
  title={{FaceSigns: Semi-Fragile Neural Watermarks for Media Authentication and Countering Deepfakes}},
  author={Neekhara, Paarth and Hussain, Shehzeen and Zhang, Xinqiao and Huang, Ke and McAuley, Julian and Koushanfar, Farinaz},
  journal={arXiv:2204.01960},
  year={2022}
}

#### CMUA Watermark: https://github.com/VDIGPKU/CMUA-Watermark
@misc{huang2021cmuawatermark,
      title={CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes}, 
      author={Hao Huang and Yongtao Wang and Zhaoyu Chen and Yuze Zhang and Yuheng Li and Zhi Tang and Wei Chu and Jingdong Chen and Weisi Lin and Kai-Kuang Ma},
      year={2021},
      eprint={2105.10872},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

#### FaceSwap: https://github.com/guipleite/CV2-Face-Swap

## Dataset from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
