# DeepCover
Uncover the truth behind AI

[![DeepCover Video](images/deepcover.gif)](https://www.youtube.com/watch?v=vTyfOBAGm_o)

DeepCover is a tool for [testing](https://dl.acm.org/doi/abs/10.1145/3358233) and 
[debugging](https://arxiv.org/abs/1908.02374) Deep Neural Network (DNNs).

## To start running the Statistical Fault Localization (SFL) based explaining:
```
python ./sfl-src/sfl.py --mobilenet-model --inputs data/ --outputs outs
```

# More options
```
python ./sfl-src/sfl.py --help
usage: sfl.py [-h] [--model MODEL] [--inputs DIR] [--outputs DIR]
              [--measures zoltar, tarantula ... [zoltar, tarantula ... ...]]
              [--measure zoltar, tarantula ...] [--mnist-dataset]
              [--cifar10-dataset] [--grayscale] [--vgg16-model]
              [--inception-v3-model] [--xception-model] [--mobilenet-model]
              [--attack] [--text-only] [--input-rows INT] [--input-cols INT]
              [--input-channels INT] [--top-classes INT]
              [--adversarial-ub FLOAT] [--adversarial-lb FLOAT]
              [--adversarial-value FLOAT] [--testgen-factor FLOAT]
              [--testgen-size INT] [--testgen-iterations INT]
```

# Dependencies
We suggest create an environment using `conda`, `tensorflow>=2.0.0`
```
conda create --name deepcover
source activate deepcover
conda install keras
conda install opencv
conda install pillow
```

# Publications
```
@inproceedings{schk2020,
AUTHOR = { Sun, Youcheng
and Chockler, Hana
and Huang, Xiaowei
and Kroening, Daniel},
TITLE = {Explaining Image Classifiers using Statistical Fault Localization},
BOOKTITLE = {European Conference on Computer Vision (ECCV)},
YEAR = { 2020 },
}
```

# Miscellaneous
[Roaming Panda Dataset](http://www.roaming-panda.com/)
