

apt update
apt install -y python-opencv
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

docker run --rm --runtime nvidia --ipc host -v /home/fulingzhi/CenterNet:/app -v /ml/dataset/coco:/app/data/coco -it pytorch/pytorch:0.4.1-cuda9-cudnn7-devel-dev bash




docker run --rm --runtime nvidia --ipc host -v /home/fulingzhi/CenterNet:/app -v /ml/dataset/coco:/app/data/coco -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-devel-centernet bash

### train squeezenet

python main.py multi_pose --exp_id squeezenet_1x --dataset coco_hp --arch squeeze --batch_size 16 --lr 1e-3 --gpus 1 --num_workers 2 --down_ratio 2

python main.py multi_pose --exp_id squeezenet_1x --dataset coco_hp --arch squeeze --batch_size 16 --lr 1.25e-4 --gpus 1 --num_workers 2 --down_ratio 4 --debug 4

python demo.py multi_pose --demo example.jpg --arch squeeze --load_model ../exp/multi_pose/squeezenet_1x/model_best.pth --gpus 2 --vis_thresh 0.1

python run_ckpt_onnx.py multi_pose --demo example.jpg --arch squeeze --load_model ../exp/multi_pose/squeezenet_1x/model_best.pth --gpus -1



### train dla

python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16

python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume

python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth

CUDA_VISIBLE_DEVICES=1 python -m tests.demo_test
CUDA_VISIBLE_DEVICES=3 python tests/dataset_test.py


CUDA_VISIBLE_DEVICES=1 python test.py multi_pose --exp_id hg --dataset coco_hp --arch hourglass --keep_res --load_model ../models/multi_pose_hg_3x.pth --flip_test
CUDA_VISIBLE_DEVICES=1 python test.py multi_pose --exp_id dla --keep_res --load_model ../models/multi_pose_dla_3x.pth --flip_test


### train ctdet

train:

`python main.py ctdet --arch res_18 --head_conv 64 --exp_id coco_res --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 1`

eval:

```
python test.py ctdet --exp_id coco_res --arch res_18 --keep_res --resume --gpus 2

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.238
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.534
```

demo:

```
python demo.py ctdet --exp_id coco_res --arch res_18 --demo ../images/33823288584_1d21cf0a26_k.jpg --load_model ../exp/ctdet/coco_res/model_best.pth

python demo.py ctdet --exp_id coco_res --arch res_18 --demo results/input_v1.png --load_model ../exp/ctdet/coco_res/model_best.pth
```




```
import cv2

desired_size = 512
im_pth = "../images/33823288584_1d21cf0a26_k.jpg"

im = cv2.imread(im_pth)
old_size = im.shape[:2] # old_size is in (height, width) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

# new_size should be in (width, height) format

im = cv2.resize(im, (new_size[1], new_size[0]))

delta_w = desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)

color = [0, 0, 0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

diff=(ori_im - new_im)
cv2.imwrite('diff.png', diff)
```


### hourglass net for kp

python test.py multi_pose --exp_id hg --dataset coco_hp --arch hourglass --keep_res --load_model ../models/multi_pose_hg_3x.pth --flip_test --gpus 1

tot 0.180s (0.170s) |load 0.000s (0.000s) |pre 0.001s (0.001s) |net 0.165s (0.152s) |dec 0.006s (0.008s) |post 0.008s (0.009s)

Evaluate annotation type *keypoints*
DONE (t=15.27s).
Accumulating evaluation results...
DONE (t=0.83s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.640
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.856
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.702
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.901
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.767
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.802
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=33.94s).
Accumulating evaluation results...
DONE (t=5.84s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.676
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.776
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.851

python demo.py multi_pose --demo example.jpg --load_model ../models/multi_pose_hg_3x.pth --arch hourglass --gpus 2

python run_ckpt_onnx.py multi_pose --demo example.jpg --load_model ../models/multi_pose_hg_3x.pth --arch hourglass --gpus -1

