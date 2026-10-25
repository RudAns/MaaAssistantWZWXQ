"""Laya 学生的云端训练（阿里云 PAI DLC，2026-10-15 起尝试云端训练）。

本机 3060 训一轮 R32 约 75 分钟，A10 约 35 分钟（试训实测快 2.1 倍）；更重要的是**不占本机显卡，对局时也能训**。
线上推理、测试集对比、回放测试仍在本机做（下载回来的 checkpoint 和本机训出来的一样用）。

  # 提交（参数和 tools/laya/train.py 一样，--train-args 原样传给 train.py）
  .venv-pai/Scripts/python.exe tools/cloud/laya_cloud.py submit --run R33_s1_top8 --ver r9zm \
      --labels data/laya/labels/r9zm_all/labels_all.jsonl --init data/laya/runs/R32_s1_top8/checkpoint \
      --train-args "--stage top8 --epochs 2 --upsample-ids data/laya/datasets/r9zm/upsample_ids.json --upsample 4 ..."
  # 盯到结束（结束后自动取回 checkpoint 到 data/laya/runs/<run>/）
  .venv-pai/Scripts/python.exe tools/cloud/laya_cloud.py watch --run R33_s1_top8

上传：代码包（tools/laya、agent/decision、data/*.json、laya 库源码、标签、--train-args 里引用到的本地文件）走 source_dir；
数据集 points.jsonl 和 init checkpoint 按内容 md5 存 OSS（没变就不重传），挂载到 /ml/input/data/{ds,ckpt}。
任务记录在 build/cloud_jobs.jsonl。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parents[2]
JOBS = P / "build" / "cloud_jobs.jsonl"
A10 = "ecs.gn7i-c8g1.2xlarge"


def md5(f: Path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with f.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def session():
    from pai.session import get_default_session
    return get_default_session()


def oss_put_dir(local: Path, key: str, s) -> str:
    """local 目录 / 文件 → oss://bucket/key（key 以 / 结尾表示目录）；已有 _DONE 标记就跳过。"""
    import pai.common.oss_utils as o
    b = s.oss_bucket
    if b.object_exists(key + "_DONE"):
        print("OSS 已有，跳过上传：", key)
    else:
        t = time.time()
        o.upload(str(local), key, bucket=b)
        b.put_object(key + "_DONE", b"ok")
        print(f"上传 {local} → {key}（{time.time() - t:.0f}s）")
    return f"oss://{s.oss_bucket_name}/{key}"


def build_bundle(run: str, ver: str, labels: Path, extra: list[Path]) -> Path:
    B = P / "build" / f"cloud_{run}"
    if B.exists():
        shutil.rmtree(B)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak*")
    for sub in ("tools/laya", "agent/decision"):
        shutil.copytree(P / sub, B / sub, ignore=ign)
    (B / "agent/__init__.py").write_text("", encoding="utf-8")
    (B / "data").mkdir(parents=True, exist_ok=True)
    for f in (P / "data").glob("*.json"):
        shutil.copy2(f, B / "data" / f.name)
    for f in [labels, *extra]:
        rel = f.resolve().relative_to(P)
        (B / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, B / rel)
    # laya 库只装在主 .venv（PAI SDK 在 .venv-pai），直接拷源码
    shutil.copytree(P / ".venv/Lib/site-packages/laya", B / "pylib/laya", ignore=ign)
    return B


def submit(a) -> int:
    s = session()
    labels = (P / a.labels).resolve()
    init = (P / a.init).resolve()
    pts = P / "data/laya/datasets" / a.ver / "points.jsonl"
    targs = shlex.split(a.train_args)
    # --train-args 里引用到的本地文件（如 --upsample-ids）一起打包
    extra = [P / x for x in targs if not x.startswith("-") and (P / x).is_file()]
    B = build_bundle(a.run, a.ver, labels, extra)
    ds_uri = oss_put_dir(pts, f"laya/datasets/{a.ver}/{md5(pts)[:12]}/", s)
    ck_uri = oss_put_dir(init, f"laya/ckpt/{init.parent.name}-{md5(init / 'model.safetensors')[:12]}/", s)
    rel_labels = labels.relative_to(P).as_posix()
    cmd = " ".join(shlex.quote(x) for x in targs)
    (B / "run.sh").write_text(f"""#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib:$PYTHONPATH" PYTHONIOENCODING=utf-8
nvidia-smi --query-gpu=name,memory.total --format=csv || true
pip install -q -i https://mirrors.aliyun.com/pypi/simple "transformers==5.17.0" "tokenizers==0.23.2" "safetensors==0.8.0" "huggingface_hub"
mkdir -p data/laya/datasets/{a.ver}
ln -sf /ml/input/data/ds/points.jsonl data/laya/datasets/{a.ver}/points.jsonl
T0=$(date +%s)
python -X utf8 -u tools/laya/train.py --run {a.run} --ver {a.ver} --labels {rel_labels} --init /ml/input/data/ckpt {cmd}
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
mkdir -p /ml/output/model
cp -r data/laya/runs/{a.run} /ml/output/model/
echo CLOUD_DONE
""", encoding="utf-8", newline="\n")
    from pai.estimator import Estimator
    from pai.image import retrieve
    img = retrieve("PyTorch", framework_version="latest", accelerator_type="GPU").image_uri
    est = Estimator(image_uri=img, command="bash run.sh", source_dir=str(B), instance_type=a.instance, instance_count=1,
                    max_run_time=int(a.max_hours * 3600), base_job_name=f"laya-{a.run}".replace("_", "-").lower(),
                    output_path=f"oss://{s.oss_bucket_name}/laya/runs/{a.run}/")
    job = est.fit(inputs={"ds": ds_uri, "ckpt": ck_uri}, wait=False)
    jid = getattr(job, "training_job_id", None) or str(job)
    rec = {"run": a.run, "job": jid, "ver": a.ver, "labels": rel_labels, "init": str(init.relative_to(P)), "instance": a.instance,
           "train_args": a.train_args, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    with JOBS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("已提交", jid, a.instance)
    return 0


def find_job(run: str) -> dict:
    rows = [json.loads(l) for l in JOBS.open(encoding="utf-8")] if JOBS.exists() else []
    rows = [r for r in rows if r["run"] == run]
    if not rows:
        sys.exit(f"build/cloud_jobs.jsonl 里没有 {run}")
    return rows[-1]


def logs(s, jid: str) -> list[str]:
    out, page = [], 1
    while True:
        r = s.training_job_api.list_logs(jid, page_size=200, page_number=page)
        items = [str(x) for x in (getattr(r, "items", r) or [])]
        out += items
        if len(items) < 200:
            return out
        page += 1


def fetch(s, run: str) -> int:
    """OSS 上的结构是 laya/runs/<run>/<任务名>/model/<run>/…（SDK 自动插了一层任务名），取最新那个任务的，逐个文件下载。"""
    import oss2
    dst = P / "data/laya/runs" / run
    if (dst / "checkpoint" / "model.safetensors").exists():
        print("本机已有", dst); return 0
    tag = f"/model/{run}/"
    objs = [o for o in oss2.ObjectIterator(s.oss_bucket, prefix=f"laya/runs/{run}/") if tag in o.key and not o.key.endswith("/")]
    if not objs:
        print("OSS 上没有找到产物：", f"laya/runs/{run}/"); return 1
    job_dir = max({o.key.split(tag)[0] for o in objs})            # 任务名带时间戳，取最新
    t = time.time()
    for o in objs:
        if not o.key.startswith(job_dir + tag):
            continue
        f = dst / o.key[len(job_dir + tag):]
        f.parent.mkdir(parents=True, exist_ok=True)
        s.oss_bucket.get_object_to_file(o.key, str(f))
    ok = (dst / "checkpoint/model.safetensors").exists()
    print(f"已取回 {dst}（{time.time() - t:.0f}s）", "checkpoint OK" if ok else "（没有 checkpoint！）")
    return 0 if ok else 1


def watch(a) -> int:
    from pai.job import TrainingJob
    s = session()
    rec = find_job(a.run)
    jid, last, t0 = rec["job"], None, time.time()
    while True:
        try:
            st = TrainingJob.get(jid).model_dump().get("Status")
        except Exception as ex:          # 接口偶尔断连（2026-10-15 实测 RemoteDisconnected），重试即可，任务本身不受影响
            print(time.strftime("%H:%M:%S"), "查询状态失败，30 秒后重试：", repr(ex)[:120], flush=True)
            time.sleep(30)
            continue
        if st != last:
            print(time.strftime("%H:%M:%S"), st, flush=True)
            last = st
        if st in ("Succeed", "Succeeded", "Failed", "Stopped"):
            break
        if time.time() - t0 > a.max_hours * 3600:
            print("盯任务超时"); return 1
        time.sleep(30)
    L = logs(s, jid)
    (P / "build" / f"cloud_{a.run}.log").write_text("\n".join(L), encoding="utf-8")
    for line in L:
        if line.startswith(("epoch", "TEST", "TRAIN_SECONDS", "CLOUD_DONE", "Traceback")) or "Error" in line:
            print("  ", line[:200])
    if st not in ("Succeed", "Succeeded") or not any("CLOUD_DONE" in x for x in L):
        print("任务没有成功，日志在 build/cloud_%s.log" % a.run); return 1
    return fetch(s, a.run)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--run", required=True); p.add_argument("--ver", required=True)
    p.add_argument("--labels", required=True); p.add_argument("--init", required=True)
    p.add_argument("--train-args", default="")
    p.add_argument("--instance", default=A10); p.add_argument("--max-hours", type=float, default=4.0)
    w = sub.add_parser("watch"); w.add_argument("--run", required=True); w.add_argument("--max-hours", type=float, default=6.0)
    f = sub.add_parser("fetch"); f.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "submit":
        return submit(a)
    if a.cmd == "watch":
        return watch(a)
    return fetch(session(), a.run)


if __name__ == "__main__":
    sys.exit(main())
