from ultralytics import YOLO
from ultralytics.utils import LOGGER

best = "/mnt/HithinkOmniSSD/user_workspace/caisihang/project/ultralytics/runs/two_stage_0512/stage2_full/weights/best.pt"
DATA = "/mnt/HithinkOmniSSD/user_workspace/caisihang/dataset/OmniRobotFaceDetect/data.yaml"

LOGGER.setLevel("WARNING")  # suppress info-level detection output

model2 = YOLO(str(best))
results = model2.val(data=DATA, imgsz=640)

acc = results.results_dict
for key, label in [("metrics/gender_acc", "gender_acc"), ("metrics/race_acc", "race_acc"), ("metrics/body_acc", "body_acc")]:
    v = acc.get(key)
    if v is not None:
        print(f"{label}: {float(v):.4f}")
    else:
        print(f"{label}: N/A")