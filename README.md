# Libtorch_YOLOv3_train_demo
A libtorch implementation of YOLOv3, supports training on custom dataset,  evaluation and detection.
## Requirements
1. LibTorch v1.5.0
2. CUDA v10.1
3. OpenCV v3.4.10
## Build
cd build\
cmake -DCMAKE_PREFIX_PATH=path/to/libtorch -DCMAKE_ECLIPSE_VERSION=4.15 -G "Eclipse CDT4 - Unix Makefiles" ../src
# Recommened
Use more "standard" method: torch.jit(TorchScript), and do pre & post process in LibTorch.
Note that this implementation does not support multi scale training yet, poor performance on diffcult tasks.
