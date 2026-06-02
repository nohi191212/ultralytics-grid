python train_attr_resume.py \
  --config ultralytics/cfg/models/26/yolo26s-attr.yaml \
  --pretrained /mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/260515_detect_face/0514/exp/weights/last.pt \
  --data /mnt/HithinkOmniSSD/user_workspace/caisihang/dataset/OmniRobotFaceDetect/data.yaml \
  --project /mnt/HithinkOmniSSD/user_workspace/caisihang/checkpoints/260518_detect_face \
  --device 0,1,2,3
