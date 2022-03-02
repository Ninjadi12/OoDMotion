# Generative Model-Enhanced Human Motion Prediction
This is the code for the paper



Anthony Bourached, Ryan-Rhys Griffiths, Robert Gray, Ashwani Jha, Parashkev Nachev.
[_Generative Model-Enhanced Human Motion Prediction_](https://arxiv.org/abs/2010.11699). Under review at ICLR 2021. Accepted at NeurIPS workshop on Interpretable Inductive Biases and Physically Structured Learning.


## Dependencies
Some older versions may work. But we used the following:

* cuda 10.1
* Python 3.6.9
* [Pytorch](https://github.com/pytorch/pytorch) 1.6.0
* [progress 1.5](https://pypi.org/project/progress/)

## Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

## Training commands
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train on angle space, in-distribution, H3.6M:
```bash
python3 main.py --data_dir "[Path To Your H36M data]/h3.6m/dataset/" --variational --lambda 0.003 --n_z 8 --dropout 0.3 --lr_gamma 1.0 --input_n 10 --output_n 10 --dct_n 20
```
in-distribution (CMU):
```bash
python3 main.py --dataset 'cmu_mocap' --data_dir "[Path To Your CMU data]/cmu_mocap/" --variational --lambda 0.003 --n_z 8 --dropout 0.3 --lr_gamma 1.0 --input_n 10 --output_n 25 --dct_n 35
```
to train on 3D space for CMU, simply change the ```--dataset 'cmu_mocap'``` to ```--dataset 'cmu_mocap_3d```. This flag is 'h3.6m' by default.

To train on 'walking' and test out-of-distribution (for h3.6M), include the extra flag:
```bash
--out_of_distribution 'walking' 
```
identically to train on 'basketball' and test out-of-distribution (for CMU), include the extra flag:
```bash
--out_of_distribution 'basketball' 
```
The same models may be trained (or used for inference independent of how they were trained) without the VGAE branch by removing the 
```
--variational
``` 
flag.

### Hyperparameter search can be conducted via:
```
python3 hyperparameter_search.py --num_trials 10 --epoch 100 --variational
```

### Inference on latent spaces for trained model, saves to latents.csv (also save DCT inputs, to inputs.csv)
```python
python3 interpretability.py --dataset 'cmu_mocap' --model_path "[Path To Your Trained Model].pth.tar"
```

## Citing

If you use our code, and/or build on our work, please cite our paper:

```
@article{https://doi.org/10.1002/ail2.63,
author = {Bourached, Anthony and Griffiths, Ryan-Rhys and Gray, Robert and Jha, Ashwani and Nachev, Parashkev},
title = {Generative Model-Enhanced Human Motion Prediction},
journal = {Applied AI Letters},
volume = {n/a},
number = {n/a},
pages = {},
doi = {https://doi.org/10.1002/ail2.63},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/ail2.63},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/ail2.63},
abstract = {Abstract The task of predicting human motion is complicated by the natural heterogeneity and compositionality of actions, necessitating robustness to distributional shifts as far as out-of-distribution (OoD). Here we formulate a new OoD benchmark based on the Human3.6M and CMU motion capture datasets, and introduce a hybrid framework for hardening discriminative architectures to OoD failure by augmenting them with a generative model. When applied to current state-of-theart discriminative models, we show that the proposed approach improves OoD robustness without sacrificing in-distribution performance. We suggest human motion predictors ought to be constructed with OoD challenges in mind, and provide an extensible general framework for hardening diverse discriminative architectures to extreme distributional shift. The code is available at https: //github.com/bouracha/OoDMotion.}
}


```

## Acknowledgments

The codebase is built on that of https://github.com/wei-mao-2019/LearnTrajDep and depends heavily on their work in [_Learning Trajectory Dependencies for Human Motion Prediction_](https://arxiv.org/abs/1908.05436) (ICCV 2019), and [_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://arxiv.org/abs/2007.11755) (ECCV 2020). Thus please also cite:

```
@inproceedings{wei2019motion,
  title={Learning Trajectory Dependencies for Human Motion Prediction},
  author={Wei, Mao and Miaomiao, Liu and Mathieu, Salzemann and Hongdong, Li},
  booktitle={ICCV},
  year={2019}
}
```

and

```
@article{mao2020history,
  title={History Repeats Itself: Human Motion Prediction via Motion Attention},
  author={Mao, Wei and Liu, Miaomiao and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2007.11755},
  year={2020}
}
```

## Licence

MIT
