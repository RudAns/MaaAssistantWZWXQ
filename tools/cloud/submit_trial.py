"""提交 PAI DLC 试训任务（2026-10-15）：代码包 build/cloud_trial（source_dir 自动上传），checkpoint 从 OSS 挂载到 /ml/input/data/ckpt。
  .venv-pai/Scripts/python.exe tools/cloud/submit_trial.py [--instance ecs.gn7i-c8g1.2xlarge]
"""
import argparse
from pai.estimator import Estimator
from pai.image import retrieve
from pai.session import get_default_session

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="ecs.gn7i-c8g1.2xlarge")     # A10 24G（免费试用）
ap.add_argument("--ckpt", default="laya/ckpt/R32_s1_top8/")
ap.add_argument("--max-hours", type=float, default=1.0)
a = ap.parse_args()
s = get_default_session()
img = retrieve("PyTorch", framework_version="latest", accelerator_type="GPU").image_uri
est = Estimator(image_uri=img, command="bash run.sh", source_dir="build/cloud_trial", instance_type=a.instance,
                instance_count=1, max_run_time=int(a.max_hours * 3600), base_job_name="laya-cloud-trial")
ckpt = f"oss://{s.oss_bucket_name}/{a.ckpt}"
job = est.fit(inputs={"ckpt": ckpt}, wait=False)
print("submitted", getattr(job, "training_job_id", job))
print("console", s.console_uri)
