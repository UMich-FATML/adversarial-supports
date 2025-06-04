# On sensitivity of meta-learning to support data

This repository contains the code for the paper [*On sensitivity of meta-learning to support data*](https://openreview.net/forum?id=Tv0O_cAdKtW) by Agarwal et al. The paper appeared at NeurIPS 2021.

### Abstract

Meta-learning algorithms are widely used for few-shot learning. For example, image recognition systems that readily adapt to unseen classes after seeing only a few labeled examples. Despite their success, we show that modern meta-learning algorithms are extremely sensitive to the data used for adaptation, i.e. support data. In particular, we demonstrate the existence of (unaltered, in-distribution, natural) images that, when used for adaptation, yield accuracy as low as 4% or as high as 95% on standard few-shot image classification benchmarks. We explain our empirical findings in terms of class margins, which in turn suggests that robust and safe meta-learning requires larger margins than supervised learning.


## Usage

### Installation

1. Install dependencies using `pip install -r requirements.txt`
2. Download the dataset files: [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing), [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing), [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
3. For each dataset loader, specify the path to the data directory. For example, in `data/CIFAR_FS.py` line 30, set the path to the CIFAR_FS data directory


### Training 
1. To train MetaOptNet, R2D2, and ProtoNet models
```
python train.py --save-path "<savedir>" --train-shot <nshot> --head <head> --network <net> --dataset <dataset> --val-episode 500
```

2. To train MAML and Meta-Curvature models
```
# MAML training
cd MAML/
python maml-train-<dataset>.py "<nshot>"


# Meta-Curvature training
cd MAML/
python train-MC.py "<dataset>" "<nshot>"
```

### Train adversarially
```
python train_adv.py --train-shot "1" --train-query "50" --val-episode "500" --val-query "50" --test-way "5" --save-path "<savepath>" --network "<net>" --head "<head>" --dataset "<dataset>" --load_model "<modelpath>" --subsample_imgs "100" --attack_rounds "3" --lr "0.01"
```


### Test (normal)
1. To test MetaOptNet, R2D2, and ProtoNet models
```
python test.py --load "<modelpath>" --episode 1000 --way 5 --shot <nshot> --query 50 --head <head> --network <net> --dataset <dataset> --outdir "<outputdir>"
```

2. To test MAML and Meta-Curvature models
```
# MAML testing
cd MAML/
python maml-test-<dataset>.py "<nshot>" "<modelpath>"


# Meta-Curvature testing
cd MAML/
python test-MC.py "<dataset>" "<nshot>" "<modelpath>"
```


### Test (adversarially)
1. To test MetaOptNet, R2D2, and ProtoNet models
```
python test_adv.py --ntasks <ntasks> --load "<modelpath>" --n_adv_rounds 3 --head <head> --network <net> --dataset <dataset> --seed 0 --outdir "<outputdir>" --way 5 --shot <nshot> --phase test --minmax_acc min
```

2. To test MAML and Meta-Curvature models
```
# MAML testing
cd MAML/
python maml-test-adv-<dataset>.py "<nshot>" "<modelpath>" "min"


# Meta-Curvation testing
cd MAML/
python test-adv-MC.py "<dataset>" "<nshot>" "<modelpath>" "<seed>" "min"
```