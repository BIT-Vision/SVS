# Codes for AAAI 2024 paper "Finding Visual Saliency in Continuous Spike Stream"

This repository contains the official codes for AAAI 2024 paper `Finding Visual Saliency in Continuous Spike Stream`.
## Requirements
- torch >= 1.8.0
- torchvison >= 0.9.0
- ...

To installl requirements, run:
```bash
conda create -n svs python==3.7
pip install -r requirements.txt
```

## Data Organization
### SVS Dataset
Download the [SVS](https://pan.baidu.com/s/1SqaCEWXCK_ng8mzNW1veXg?pwd=w2ba)[w2ba] dataset, then organize data as following format:
```
root_dir
    SpikeData
        |----00001
        |     |-----spike_label_format
        |     |-----spike_numpy
        |     |-----spike_repr
        |     |-----label
        |----00002
        |     |-----spike_label_format
        |     |-----spike_numpy
        |     |-----spike_repr
        |     |-----label
        |----...
```
Where `label` contains the saliency labels, `spike_numpy` contains the compress spike data, `spike_repr` contains the interval spike representation, `spike_label_format` contains instance labels.

## Training

### Training on SVS dataset
To train the model on SVS dataset, just modify the dataset root `$cfg.DATA.ROOT` in `config.py`, `--step` is used for multi-step, `--clip` is used for multi-step loss, then run following command:
```bash
python train.py --gpu ${GPU-IDS} --exp_name ${experiment} --step --clip
```
## Testing
Download the model pretrained on SVS dataset [multi_step](https://pan.baidu.com/s/1yx5oieoAw5Mmhj2dqyjy2w?pwd=vn2x)[vn2x].

```bash
python inference.py --checkpoint ${./multi_step.pth} --results ${./results/SVS} --step
```

Download the model pretrained on SVS dataset [single_step](https://pan.baidu.com/s/1WHuwUdZNmzeVSqj2xo_o0g?pwd=scc0)[scc0].
```bash
python inference.py --checkpoint ${./single_step.pth} --results ${./results/SVS}
```

The results will be saved as indexed png file at `${results}/SVS`.

Additionally, you can modify some setting parameters in `config.py` to change configuration.

# Acknowledgement
This codebase is built upon [official DCFNet repository](https://github.com/Roudgers/DCFNet) and [official Spikformer repository](https://github.com/ZK-Zhou/spikformer).
We modify the code from [eval-co-sod](https://github.com/zzhanghub/eval-co-sod) to evaluate the results.
