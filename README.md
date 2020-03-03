# Semantic Segmentation of facies in the seismic images - SeismicNet

## Reference: 
The pipeline of this work is taken from: https://github.com/olivesgatech/facies_classification_benchmark 

which is the implementation of the paper: https://arxiv.org/abs/1901.07659

## Dataset

To download the training and testing data, run the following commands in the terminal: 

```bash
# download the files: 
wget https://www.dropbox.com/s/p6cbgbocxwj04sw/data.zip
# check that the md5 checksum matches: 
openssl dgst -md5 data.zip # Make sure the result looks like this: MD5(data.zip)= bc5932279831a95c0b244fd765376d85, otherwise the downloaded data.zip is corrupted. 
# unzip the data:
unzip data.zip 
# create a directory where the train/val/test splits will be stored:
mkdir data/splits
```

Alternatively, you can click [here](https://www.dropbox.com/s/p6cbgbocxwj04sw/data.zip) to download the data directly. Make sure you have the following folder structure in the `data` directory after you unzip the file: 

```bash
data
├── splits
├── test_once
│   ├── test1_labels.npy
│   ├── test1_seismic.npy
│   ├── test2_labels.npy
│   └── test2_seismic.npy
└── train
    ├── train_labels.npy
    └── train_seismic.npy
```

The train and test data are in NumPy `.npy` format ideally suited for Python. You can open these file in Python as such: 

```python
import numpy as np
train_seismic = np.load('data/train/train_seismic.npy')
```

**Make sure the testing data is only used once after all models are trained. Using the test set multiple times makes it a validation set.**

We also provide fault planes, and the raw horizons that were used to generate the data volumes in addition to the processed data volumes before splitting to training and testing. If you're interested in this data, you can download it from [here](https://www.dropbox.com/s/cvfrud3kp3o69ar/raw.zip). In addition, you can download the well log files from [here](https://www.dropbox.com/s/vupljhjd3pqr8du/logs.zip). 

  
### Prerequisites:

The version numbers are the exact ones I've used, but newer versions should works just fine. 

```
Pillow == 5.2.0
matplotlib == 2.0.2
numpy == 1.15.1
tensorboardX == 1.4 # from here: https://github.com/lanpa/tensorboardX
torch == 0.4.1
torchvision == 0.2.1
tqdm == 4.14.0
scikit-learn == 0.18.1
```

### Training: 

To train the patch-based model with a different batch size and with augmentation,  you can run:

```bash
python train.py --batch_size 32 --aug True
```

Unless you specify the options you want, the default ones (listed in `patch_train.py` will be used). Similarly, you can do the same thing for the section-based model. 

### Testing:

To test a model, you have to specify the path to the trained model. For example, you can run: 

```bash
python test.py --model_path 'path/to/trained_model.pkl' 
```

In order to be consistent with the results of the paper, we suggest you keep all the test options to their deafault values (such as `test_stride` , `crossline` and `inline` ). Feel free to change the test `split` if you do not want to test on both test splits, and make sure you update `train_patch_size` if it was changed during training. Once the test code is finished, it will print the results in the terminal. You can also view the test results, both images and metrics, in Tensorboard.  


