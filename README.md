# Transfer Learning using Adversarially Robust ImageNet Models

This repository contains the code and models necessary to replicate the results of our NeurIPS2020 submission:

**Do Adversarially Robust ImageNet Models Transfer Better?** <br>
*Under Review*


### Getting started
Our code relies heavily on the [MadryLab](http://madry-lab.ml/) public [robustness library](https://github.com/MadryLab/robustness), which will be installed automatically as you follow the below instructions.
1.  Clone our repo: `git clone https://github.com/anonymousneuripsauthor/paper-8887`

2.  Install dependencies:
    ```
    conda create -n transfer-learning python=3.7
    conda activate transfer-learning
    pip install -r requirements.txt
    ```


### Running transfer learning experiments
The entry point of our code is [main.py](src/main.py) (see the file for arguements).

1- Download one of the pretrained robust ImageNet models, say an L2-robust ResNet-18 with &epsilon; = 3. For a full list of models, see the [section below](#download-our-robust-imagenet-models)!
  ```
  mkdir pretrained-models & 
  wget -O pretrained-models/resnet-18-l2-eps3.ckpt "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D"
  ```
2- Run the following script
  ```
  python src/main.py --arch resnet18 \
    --dataset cifar10 \
    --data /tmp \
    --out-dir outdir \
    --exp-name cifar10-transfer-demo \
    --epochs 150 \
    --lr 0.01 \
    --step-lr 30 \
    --batch-size 64 \
    --weight-decay 5e-4 \
    --adv-train 0 \
    --model-path pretrained-models/resnet-18-l2-eps3.ckpt
  ```

That's it!

**Datasets:** 
* aircraft ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/fgvc-aircraft-2013b.tar.gz?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* birds ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/birdsnap.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
)), 
* caltech101 **(Automatically downloaded when you are the code)**
* caltech256 **(Automatically downloaded when you are the code)**
* cifar10 **(Automatically downloaded when you are the code)**
* cifar100 **(Automatically downloaded when you are the code)**
* dtd ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/dtd.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* flowers ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/flowers.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* food ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/food.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* pets ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/pets.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* stanford_cars ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/stanford_cars.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))
* SUN397 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/SUN397.tar?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D
))

To use any of these datasets in the code:
1. Download (click or use `wget`) and extract the desired dataset somewhere, e.g. 
    ```
    tar -xvf pets.tar
    mkdir datasets
    mv pets ./datasets
    ```
2. add the dataset name and path as arguments , e.g. 
    ```
    python src/main.py --arch resnet18  ... --dataset pets --data ./datasets/pets
    ```

**Architectures:**
You can use any of the following architectures simply by passing them as arguments to the code e.g. 
```
python src/main.py --arch resnet50 ...
```
```python
archs = [resnet18, 
         resnet50, 
         wide_resnet50_2, 
         wide_resnet50_4, 
         densenet,
         mnasnet,
         mobilenet,
         resnext50_32x4d,
         shufflenet,
         vgg16_bn
         ]
```

### Download our robust ImageNet models
##### ResNet-18 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)  | [&epsilon; = 0.01](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.01.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.03](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.03.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.05](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.05.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.1](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.25.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 5.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)

* Linf
  * [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps1.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 2.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps2.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 4.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps4.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 8.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_linf_eps8.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### ResNet-50 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.01](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.01.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.03](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.03.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.05](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.05.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.1](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.25.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 5.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)

* Linf
  * [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps1.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 2.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps2.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 4.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps4.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 8.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_linf_eps8.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### Wide-ResNet-50-2 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.01](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.01.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.03](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.03.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.05](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.05.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.1](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.25](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.25.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 5.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_l2_eps5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)

* Linf
  * [&epsilon; = 0.5](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps0.5.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps1.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 2.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps2.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 4.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps4.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 8.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_2_linf_eps8.0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### Wide-ResNet-50-4 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 1.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps1.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/wide_resnet50_4_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)



##### DenseNet 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/densenet_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/densenet_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### MNASNET 
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mnasnet_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mnasnet_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### MobileNet-v2
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mobilenet_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/mobilenet_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)



##### ResNeXt50_32x4d
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnext50_32x4d_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnext50_32x4d_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### ShuffleNet
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/shufflenet_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/shufflenet_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)


##### VGG16_bn
* L2 
  * [&epsilon; = 0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D) | [&epsilon; = 3.0](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/vgg16_bn_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D)

