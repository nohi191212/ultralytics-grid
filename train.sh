python train_attr.py \
    --config ultralytics/cfg/models/26/yolo26-attr.yaml \
    --pretrained /mnt/HithinkOmniSSD/user_workspace/caisihang/mymodel/YOLO26s-face/yolo26s-face.pt \
    --data /mnt/HithinkOmniSSD/user_workspace/caisihang/project/人脸检测数据集/hybrid_test/data.yaml \
    --project /mnt/workspace/checkpoint/yolo26s-face-260512-test \
    --device cpu