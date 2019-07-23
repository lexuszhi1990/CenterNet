

### setup env
apt-get update
apt install -y cython libglib2.0-dev libsm6 libxext6 libxrender-dev
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
pip install -i https://mirrors.aliyun.com/pypi/simple/ onnx onnx-simplifier pycocotools

docker run --rm --runtime nvidia --ipc host -v /home/fulingzhi/CenterNet:/app -v /ml/dataset/coco:/app/data/coco -v /home/fulingzhi/ncnn-dev:/ncnn -w /app -it pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7-centernet bash

docker run --rm --runtime nvidia --ipc host -v /home/fulingzhi/CenterNet:/app -v /ml/dataset/coco:/app/data/coco -v /ml/dataset/ai_challenger:/app/data/ai_challenger -v /home/fulingzhi/ncnn-dev:/ncnn -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-devel-centernet bash

docker run --rm --runtime nvidia --ipc host -v /mnt/data-4t/workspace/david/CenterNet:/app -v /mnt/data-4t/coco/cocoapi:/app/data/coco -v /mnt/data-4t/workspace/david/ncnn:/ncnn -w /app -it pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime-centernet bash

### example cmd

python -m lib.models.networks.squeezenet

cp /app/src/example-sim.onnx . && ../tools/onnx/onnx2ncnn example-sim.onnx example.param example.bin && ../tools/ncnn2mem example.param example.bin example.id.h example.mem.h

### setup ncnn

1. install protobuf
https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

    sudo apt-get -y install autoconf automake libtool curl make g++ unzip libopencv-dev
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git submodule update --init --recursive
    ./autogen.sh

    ./configure
    make
    make check
    sudo make install
    sudo ldconfig # refresh shared library cache.

2. build with GPU support
    export VULKAN_SDK=/ncnn/ncnn/vulkansdk_1.1.92.1/x86_64
    export Vulkan_DIR=/ncnn/thirdparty/vulkansdk_1.1.92.1
    cmake -DNCNN_VULKAN=ON ..

3. generate ncnn model
    python3 -m onnxsim example.onnx example-sim.onnx
    onnx2ncnn example-sim.onnx example.param example.bin
    ncnn2mem example.param example.bin example.id.h example.mem.h


### training

1. ctdet res_18

```
train:
python main.py ctdet --arch res_18 --head_conv 64 --exp_id coco_res --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 1

eval:
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

demo:

python demo.py ctdet --exp_id coco_res --arch res_18 --demo ../images/33823288584_1d21cf0a26_k.jpg --load_model ../exp/ctdet/coco_res/model_best.pth
python demo.py ctdet --exp_id coco_res --arch res_18 --demo results/input_v1.png --load_model ../exp/ctdet/coco_res/model_best.pth
```

2. kp hourglass net

```
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
```

3. kp squeezenet

```
pretest model:

python -m lib.models.networks.squeezenet
python3 -m onnxsim example.onnx example-sim.onnx
onnx2ncnn example-sim.onnx example.param example.bin
ncnn2mem example.param example.bin example.id.h example.mem.h

cp /app/src/example-sim.onnx . && ../tools/onnx/onnx2ncnn example-sim.onnx example.param example.bin && ../tools/ncnn2mem example.param example.bin example.id.h example.mem.h

train:
python main.py multi_pose --exp_id squeeze_new_1x --arch squeeze --dataset coco_hp --batch_size 32  --lr 5e-4 --gpus 2 --num_workers 2 --down_ratio 4 --lr_step 60,90

python test.py multi_pose --exp_id squeezenet_1x --dataset coco_hp --arch squeeze --keep_res --load_model ../exp/multi_pose/squeezenet_1x/model_best.pth --flip_test --gpus 2

python run_ckpt_onnx.py multi_pose --demo example.jpg --arch squeeze --load_model ../exp/multi_pose/squeeze_new_1x/model_best.pth --gpus -1

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.148
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.357
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.100
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.128
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.606
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.375
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=45.12s).
Accumulating evaluation results...
DONE (t=9.49s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.095
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.132
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.614

android:
    network               : 1177ms
    network + post_process: 1371ms
```

python main.py multi_pose --exp_id squeeze_0.5_coco --arch squeeze --dataset coco_hp --batch_size 64  --lr 5e-3 --num_workers 2 --input_res 224 --down_ratio 2 --lr_step 30,60,90 --gpus 3 --debug 4

python main.py ai_challenge --exp_id squeeze_0.5_ai_challenge --arch squeeze --dataset ai_challenge --batch_size 64  --lr 5e-3 --num_workers 2 --input_res 224 --down_ratio 2 --lr_step 30,60,90 --gpus 3


### test new dataset load

python -m lib.datasets.coco_kp multi_pose --arch squeeze
