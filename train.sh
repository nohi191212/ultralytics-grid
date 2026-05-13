/opt/conda/bin/python train_two_stage.py \
  --config ultralytics/cfg/models/26/yolo26s-attr.yaml \
  --pretrained /mnt/dataset/OmniRobotFaceDetect/yolo26s-face.pt \
  --data /mnt/dataset/OmniRobotFaceDetect/data.yaml \
  --project /mnt/workspace/checkpoint/two_stage_0512 \
  --device 0,1 \
  --stage1-epochs 30 \
  --stage2-epochs 70 \
  --imgsz 640 \
  --batch 32