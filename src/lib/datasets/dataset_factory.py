from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP

from .ai_challenger_kp import AIChallengeKp
from .coco_hp import COCOHPNew
from .ai_challenger_kp_ori import AIChallengeKpOri

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):

  if task == 'easy_multi_pose':
    return COCOHPNew

  elif task == 'ai_challenge':
    return AIChallengeKp

  elif task == 'ai_challenge_hp':
    return AIChallengeKpOri

  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

