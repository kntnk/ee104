# CIFAR-10 Convolutional Neural Network  
Implemented CNN using the CIFAR-10 dataset and achieved >90% validation accuracy.

## Used Methods
- Added more layers.  
- Increased Dropout.  
- Data Augumentation.  
- Batch Normalization.  
- Adjusted learning rate.  


# Local environment  
- macOS Monterey Version 12.0.1  


# Setup of the environment on M1 Mac  
SOURCE: (https://makeoptim.com/en/deep-learning/tensorflow-metal)  

## 1. Switch zsh to bach on M1 mac (arm64).  
```
chsh -s /bin/bash  
```

## 2. Download miniforge from (https://github.com/conda-forge/miniforge)  
```
bash Miniforge3-MacOSX-arm64.sh  
source ~/miniforge3/bin/activate  
```

## 3. Enable the conda environment.  
```
conda create -n tensorflow python=3.9.5  
conda activate tensorflow  
conda install -c apple tensorflow-deps  
```

## 4. Uninstall existing tensorflow-macos and tensorflow-metal.  
```
python -m pip uninstall tensorflow-macos  
python -m pip uninstall tensorflow-metal  
```

## 5. Upgrade tensorflow-deps.  
```
conda install -c apple tensorflow-deps --force-reinstall  
conda install -c apple tensorflow-deps --force-reinstall -n tensorflow  
```

## 6. Install tensorflow.  
```
python -m pip install tensorflow-macos  
python -m pip install tensorflow-metal  
```

## 7. Install required python package.  
```
conda install -y matplotlib  
```

## 8. Uninstall numpy and setuptools to solve importing the numpy C-extensions failed.  
```
pip uninstall -y numpy  
pip uninstall -y setuptools  
pip install setuptools  
pip install numpy  
```

The numpy version will be updated from 1.19 to 1.21 and will see the error message, 
such as numpy 1.21 is incompatible, but should be able to import tensorflow.  


# Other packages  
```
pip install np_utils  
```


# Execution of program  
```
python3 cnn_cifar.py  
```