
### setup env

apt-get update
apt install -y cython libglib2.0-dev libsm6 libxext6 libxrender-dev
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
pip install -i https://mirrors.aliyun.com/pypi/simple/ onnx onnx-simplifier pycocotools

docker run --ipc host --rm -v /mnt/data-4t/workspace/david/CenterNet:/app -v /mnt/data-4t/coco/cocoapi:/app/data/coco -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime-centernet bash

docker run --rm -v /mnt/data-4t/workspace/david/CenterNet:/app -v /mnt/data-4t/coco/cocoapi:/app/data/coco -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime bash

docker run --rm -v /mnt/data-4t/workspace/david/CenterNet:/app -v /mnt/data-4t/coco/cocoapi:/app/data/coco -w /app -it pytorch/pytorch:0.4.1-cuda9-cudnn7-devel bash

docker run --rm -v /mnt/data-4t/workspace/david/CenterNet:/app -v /mnt/data-4t/coco/cocoapi:/app/data/coco -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-devel bash


### train kp

python main.py multi_pose --exp_id res_18_1x --dataset coco_hp --batch_size 32 --master_batch 9 --lr 5e-4 --arch res_18 --gpus 0 --num_workers 2

python demo.py multi_pose --arch res_18 --demo ../images/17790319373_bd19b24cfc_k.jpg --load_model ../exp/multi_pose/res_18_1x/model_best.pth
