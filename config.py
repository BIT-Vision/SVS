from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.OUTPUT_DIM = 1
cfg.MODEL.TYPE = 'RST'

cfg.MODEL.ACTIVATION_TYPE = 'lif'  #lif_atten lif

# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.LR = 2e-4
cfg.TRAIN.WEIGHT_DECAY = 2e-5
cfg.TRAIN.BACKBONE_MULTIPLIER = 1

cfg.TRAIN.EPOCH = 20
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.NUM_WORKER = 8

cfg.TRAIN.SKIP_SAVE = 4

cfg.DATA = edict()
cfg.DATA.ROOT = '../../..'
cfg.DATA.DATA_NAME = 'SpikeData'

cfg.DATA.DATA_MODE = 'spike'

cfg.MODEL.INPUT_DIM = 3
cfg.DATA.DATASET_CONFIG = './dataset_configs/spike.yaml'

cfg.DATA.SIZE = (256, 256)
cfg.DATA.CLIP_LEN = 1
cfg.DATA.PREPARE = False
cfg.DATA.INTERVAL = 1


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
