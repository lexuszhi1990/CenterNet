

apt update
apt install -y python-opencv
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

docker run --rm --runtime nvidia --ipc host -v /home/fulingzhi/CenterNet:/app -v /ml/dataset/coco:/app/data/coco -it pytorch/pytorch:0.4.1-cuda9-cudnn7-devel-dev bash



### train dla

python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16

python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume

python demo.py multi_pose --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/multi_pose_dla_3x.pth

CUDA_VISIBLE_DEVICES=1 python -m tests.demo_test
CUDA_VISIBLE_DEVICES=3 python tests/dataset_test.py




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
new_im = cv2.copyMakeBorder(im, top-2, bottom+2, left, right, cv2.BORDER_CONSTANT, value=color)
diff=(ori_im - new_im)
cv2.imwrite('diff.png', diff)
```
