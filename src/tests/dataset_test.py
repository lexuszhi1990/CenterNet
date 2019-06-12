
import os.path as osp
import sys
import torch
import torch.utils.data

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

lib_dir = '/app/src/lib'
# Add lib to PYTHONPATH
add_path(lib_dir)

from opts import opts


from datasets.dataset_factory import get_dataset

def test_det_dataset():

    Dataset = get_dataset('coco', 'ctdet')
    opt = opts().init('ctdet --arch res_18 --head_conv 64 --exp_id coco_res --batch_size 32 --master_batch 15 --lr 1.25e-4'.split(' '))
    data_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    for iter_id, batch in enumerate(data_loader):

        import pdb; pdb.set_trace()

test_det_dataset()


