from .coco import build as build_coco, DatasetCOCO
from .lvis import build as build_lvis, DatasetLVIS
from .fss import build as build_fss, DatasetFSS
from .deepglobe import build as build_dg, DatasetDeepglobe
from .isic import build as build_isic, DatasetISIC
from .lung import build as build_lung, DatasetLung
from .pascal_part import build as build_pascal_part, DatasetPASCALPart
from .paco_part import build as build_paco_part, DatasetPACOPart
from .ade20k import build as build_ade20k, SemADE
from .sarcoma import build as build_sarcoma, DatasetSarcoma
from .msd import build as build_msd, DatasetMSD
from .msd_all import build as build_msd_all, DatasetMSDAll
from .btcv import build as build_btcv
from .msd_vol import build as build_msd_vol

def build_dataset(dataset_file: str, image_set: str, args=None):    
    if dataset_file == 'sarcoma':
        return build_sarcoma(image_set, args)
    elif dataset_file == 'msd':
        return build_msd(image_set, args)
    elif dataset_file == 'msd_all':
        return build_msd_all(image_set, args)
    elif dataset_file == 'msd_vol':
        return build_msd_vol(image_set, args)
    elif dataset_file == 'btcv':
        return build_btcv(image_set, args)
    
    raise ValueError(f'dataset {dataset_file} not supported')
