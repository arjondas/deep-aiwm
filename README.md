# A Deep Learning–based Audio-in-Image Watermarking Scheme

#### Some notes one dataset preparation
* Need to import your own Audio and Image Dataset. For our training run we used the Speech Commands Dataset from Kaggle and resampled with a sampling rate of 8192. For images we used the MS COCO Dataset rescaled to 128x128 pixels.
* Since we are training the model in TPU, it is necessary to offload the dataset into the memory first. That's why we are using preprocessed image and audio numpy blocks for dataset.
* Audio dataset expected numpy shape: (Dataset_length, 8192, 1)
* Image dataset expected numpy shape: (Dataset_length, 128, 128, 3)
* After Data preparation, install the dependencies from requirements.txt

## Cite
[Deep Audio-in-Image Watermarking Paper](https://arxiv.org/abs/2110.02436):
```
@inproceedings{das2021deep,
  title={A Deep Learning-based Audio-in-Image Watermarking Scheme},
  author={Das, Arjon and Zhong, Xin},
  booktitle={2021 International Conference on Visual Communications and Image Processing (VCIP)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```