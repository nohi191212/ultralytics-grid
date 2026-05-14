python train_attr.py \
  --config ultralytics/cfg/models/26/yolo26s-attr.yaml \
  --pretrained /mnt/HithinkOmniSSD/user_workspace/caisihang/project/人脸检测数据集/merged_yolo/yolo26s.pt \
  --data /mnt/HithinkOmniSSD/user_workspace/caisihang/project/人脸检测数据集/hybrid_12k/data.yaml \
  --project /mnt/HithinkOmniSSD/user_workspace/caisihang/project/ultralytics/runs/0514 \
  --device 0,1 