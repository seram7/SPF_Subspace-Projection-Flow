# SPF(Subspace Projection Flow)


## Environment

conda is recommended.

```
# create environment using python
conda create -n latent python=3.10 -y

# install pytorch
# go https://pytorch.org/get-started/previous-versions/ and find appropriate install command
# for example:
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# install other dependencies
pip install -r requirements.txt

```

If you encounter an error similar to `libtiff.so.5: cannot open shared object file: No such file or directory`, try the following (from [github issue](https://github.com/rm-hull/luma.led_matrix/issues/154)):

```
sudo apt update
sudo apt install libtiff5
```


## Dataset settings
### Raw Data

To follow the experiment part in the paper, which does not require finetuning backbone network, it is enough to download `extracted feature of data` in the below.

Raw data is required for finetuning new backbone network. 
If you don't have data in your `datasets`,
the code will try to download some datasets using PyTorch.

Imagenet, Texture, Places365 iNaturalist, SSB_hard, NINCO, Openimage_O datasets should be downloaded manually and be placed on directories written below

```

Imagenet: datasets/imagenet_1K
    Train: datasets/Imagenet_1K/train, datasets/datalists/train_imagenet.txt
        images should be placed in subdirectory with class name (ex: datasets/imagenet_1K/n01440764/xxxx.jpg)
    Validation: datasets/imagenet_1K/val, datasets/datalists/val_imagenet.txt
    Test: datasets/imagenet_1k/val, datasets/datalists/val_imagenet.txt

Texture(ViM, OpenOODv1.5): datasets/texture
    images should be placed in subdirectory with class name
    .directory file in waffled subdirectory should be removed before use

iNaturalist(ViM, OpenOODv1.5): datasets/iNaturalist/images

SSB_hard(OpenOODv1.5): datasets/ssb_hard
    images should be placed in subdirectory with class name (ex: datasets/ssb_hard/n00470682/xxxx.jpg)

NINCO(OpenOODv1.5): datasets/ninco
    images should be placed in subdirectory with class name (ex: datasets/ninco/amphiuma_means/amphiuma_means_000_10045958.jpeg)

Openimage_O(OpenOODv1.5): datasets/openimage_o/images
    OpenOODv1.5 splits Openimage_O into Validation set and Test set.
    Validation: datasets/datalists/val_openimage_o.txt
    Test: datasets/datalists/test_openimage_o.txt


```
To use sampled dataset from ViM, download datalists folder from ViM official github and place the files in datasets/datalists directory
To use dataset from OpenOODv1.5, refer to https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download

### Extracted Features of Data
Features of all Dataset are extracted with finetuned backbone network using CIDER(https://github.com/deeplearning-wisc/cider)

```
data_space = 'http://robot2.snu.ac.kr:8001'

cifar100 : {data_space}/results/cifar100_backbone_sphere/finetune
imagenet1k: {data_space}/results/imagenet_backbone_sphere/finetune
```



## Directories

```
├── configs: config yaml files
├── datasets: data files for datasets (e.g., CIFAR-10)
│   └─ datalists: data lists .txt files
├── loader: codes for loading data
├── models
├── results: model runs are saved here
└── trainers
```

## Training

Example: 
```
python train.py model=cifar100/flow_sphere_dsm data=sphere_cifar100 eval=sphere_ood_cifar100 run=dev device=0
python train.py model=imagenet/flow_sphere_dsm data=sphere_imagenet eval=sphere_ood_imagenet run=dev device=0 
```

## Feature and statistics Extraction

Extract features from trained backbone:
```
python extract_feature.py --trained=cifar100_backbone_sphere --run=finetune --data=cifar100,ood_cifar100
```

## Evaluation

Example:
```
python evaluate.py --model=vim --data=cifar100 --run=dev
```

Evalutate models with extracted features:
```
python evaluate.py --trained=cifar100_flow_sphere_dsm --run=dev (--eval=sphere_ood_cifar100) (--iter=24000)
    ㄴ without --eval, i.e. to not use '--eval', eval dataset should be determined at training time
    ㄴ without --iter, model_best.pkl will be evaluated 
```

## Tensorboard


Install tensorboard with `pip install tensorboard` (added in requirements.txt)

Then run,
```
tensorboard --logdir <결과 파일 저장된 디렉토리> --port <남이 안 쓰는 포트> --bind_all
```

Open a browser, and then go into `<host>:<port>`. For example, `robot3.snu.ac.kr:55556`.
