python train_attr_resume.py \
  --config ultralytics/cfg/models/26/yolo26s-attr.yaml \
  --pretrained /mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/260515_detect_face/0514/exp/weights/last.pt \
  --data /mnt/HithinkOmniSSD/user_workspace/caisihang/project/人脸检测数据集/hybrid_12k/data.yaml \
  --project /mnt/HithinkOmniSSD/user_workspace/caisihang/project/ultralytics/runs/0515 \
  --device 0