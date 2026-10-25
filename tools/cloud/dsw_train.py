"""在 DSW 实例上训练 Laya（2026-10-15：DSW 每月 250 计算时免费；用完必须及时关机）。

不需要远程登录：实例开机时自动执行（user_command.on_start）OSS 挂载目录里的任务脚本，结果写回 OSS；
本机盯 OSS 上的 DONE 标记，一出现就**立刻关机**并下载结果。实例另设了闲置 30 分钟自动关机兜底（dsw.py）。

  # 一次性：给实例挂 OSS、设开机命令（实例要处于停止状态）
  .venv-pai/Scripts/python.exe tools/cloud/dsw_train.py setup
  # 训练（参数同 laya_cloud.py submit）；会开机 → 等 DONE → 关机 → 下载到 data/laya/runs/<run>/
  .venv-pai/Scripts/python.exe tools/cloud/dsw_train.py run --run R33_s1_top8 --ver r9zm --labels ... --init ... --train-args "..."
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import tarfile
import time
from pathlib import Path

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / "tools" / "cloud"))
import dsw  # noqa: E402
import laya_cloud as LC  # noqa: E402

MOUNT = "/mnt/laya"                  # OSS 前缀 laya/ 挂在实例的这里
JOB = "laya/dsw_job/"               # OSS 上的任务目录
NAME = "laya-train"
ON_START = (f"mkdir -p {MOUNT}/dsw_job; if [ -f {MOUNT}/dsw_job/run.sh ] && [ ! -f {MOUNT}/dsw_job/DONE ]; then "
            f"nohup bash {MOUNT}/dsw_job/run.sh > /mnt/workspace/dsw_job.log 2>&1 & fi")


def setup() -> int:
    from alibabacloud_pai_dsw20220101 import models as dm
    s, c = dsw.client()
    i = dsw.find(NAME)
    if i.status not in ("Stopped", "Failed"):
        print("实例要先停机：", i.status); return 1
    uri = f"oss://{s.oss_bucket_name}.oss-{s.region_id}-internal.aliyuncs.com/laya/"
    req = dm.UpdateInstanceRequest(
        datasets=[dm.UpdateInstanceRequestDatasets(uri=uri, mount_path=MOUNT, mount_access="RW", dynamic=False)],
        user_command=dm.UpdateInstanceRequestUserCommand(on_start=dm.UpdateInstanceRequestUserCommandOnStart(content=ON_START)))
    r = c.update_instance(i.instance_id, req)
    print("已设置：OSS", uri, "→", MOUNT, "；开机命令 →", ON_START[:80], "…", r.body.to_map().get("Success"))
    return 0


RUN_SH = r"""#!/bin/bash
# DSW 训练任务（tools/cloud/dsw_train.py 生成）。结束时写 DONE（内容 ok / fail）。
set -o pipefail
J={mount}/dsw_job
W=/mnt/workspace/laya_job
finish() {{ [ -f $J/DONE ] && return 0; echo "$1" > $J/DONE.tmp; cp /mnt/workspace/dsw_job.log $J/out.log 2>/dev/null; mv $J/DONE.tmp $J/DONE; }}
trap 'finish fail; exit 1' ERR
date > $J/STARTED
( while [ ! -f $J/DONE ]; do cp /mnt/workspace/dsw_job.log $J/out.log 2>/dev/null; sleep 60; done ) &
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# 训练环境装在持久盘上（第一次装，之后复用）
# 2026-10-16：原来只看 venv 在不在 → 一个空 venv 让安装被跳过（No module named torch）；现在按"导入得了"判断
[ -x /mnt/workspace/venv/bin/python ] || python3 -m venv /mnt/workspace/venv
if ! /mnt/workspace/venv/bin/python -c "import torch, transformers" 2>/dev/null; then
  /mnt/workspace/venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ torch==2.8.0     || /mnt/workspace/venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple torch
  /mnt/workspace/venv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple "transformers==5.17.0" "tokenizers==0.23.2" "safetensors==0.8.0" huggingface_hub numpy
fi
PY=/mnt/workspace/venv/bin/python
$PY -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())"
rm -rf $W && mkdir -p $W && tar xzf $J/code.tar.gz -C $W
cd $W
mkdir -p data/laya/datasets/{ver} /mnt/workspace/ckpt
cp {mount}/{ds_key}points.jsonl data/laya/datasets/{ver}/points.jsonl
rsync -a {mount}/{ck_key} /mnt/workspace/ckpt/{ck_name}/ 2>/dev/null || cp -r {mount}/{ck_key}. /mnt/workspace/ckpt/{ck_name}/
export PYTHONPATH="$W/pylib:$PYTHONPATH" PYTHONIOENCODING=utf-8
T0=$(date +%s)
$PY -X utf8 -u tools/laya/train.py --run {run} --ver {ver} --labels {labels} --init /mnt/workspace/ckpt/{ck_name} {targs}
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
mkdir -p {mount}/dsw_out/{run}
cp -r data/laya/runs/{run}/. {mount}/dsw_out/{run}/
finish ok
"""


def _status(tries: int = 8) -> str:
    """查实例状态；阿里云接口偶尔断连（2026-10-16 R33 因一次 RemoteDisconnected 中途关机），重试，查不到返回 "?"。"""
    for k in range(tries):
        try:
            return dsw.find(NAME).status
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M:%S"), f"查状态失败（{type(e).__name__}），重试 {k + 1}/{tries}", flush=True)
            time.sleep(15)
    return "?"


def _exists(b, key: str, tries: int = 8):
    for k in range(tries):
        try:
            return b.object_exists(key)
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M:%S"), f"查 OSS 失败（{type(e).__name__}），重试 {k + 1}/{tries}", flush=True)
            time.sleep(15)
    return None


def run(a) -> int:
    from alibabacloud_pai_dsw20220101 import models as dm  # noqa: F401
    s, c = dsw.client()
    b = s.oss_bucket
    labels = (P / a.labels).resolve()
    init = (P / a.init).resolve()
    pts = P / "data/laya/datasets" / a.ver / "points.jsonl"
    targs = shlex.split(a.train_args)
    extra = [P / x for x in targs if not x.startswith("-") and (P / x).is_file()]
    B = LC.build_bundle(a.run, a.ver, labels, extra)
    tgz = P / "build" / f"dsw_{a.run}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(B, arcname=".")
    ds_key = f"datasets/{a.ver}/{LC.md5(pts)[:12]}/"
    ck_name = f"{init.parent.name}-{LC.md5(init / 'model.safetensors')[:12]}"
    LC.oss_put_dir(pts, "laya/" + ds_key, s)
    LC.oss_put_dir(init, f"laya/ckpt/{ck_name}/", s)
    import pai.common.oss_utils as o
    for k in ("DONE", "out.log", "STARTED"):
        if b.object_exists(JOB + k):
            b.delete_object(JOB + k)
    o.upload(str(tgz), JOB + "code.tar.gz", bucket=b)
    sh = RUN_SH.format(mount=MOUNT, ver=a.ver, ds_key=ds_key, ck_key=f"ckpt/{ck_name}/", ck_name=ck_name, run=a.run,
                       labels=labels.relative_to(P).as_posix(), targs=" ".join(shlex.quote(x) for x in targs))
    b.put_object(JOB + "run.sh", sh.encode("utf-8"))
    rec = {"run": a.run, "where": "dsw", "ver": a.ver, "labels": labels.relative_to(P).as_posix(), "init": str(init.relative_to(P)),
           "train_args": a.train_args, "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    with LC.JOBS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    i = dsw.find(NAME)
    if i.status in ("Running",):
        print("实例已在运行 —— 开机命令不会再执行，先关机再开：", i.status); return 1
    # 开机：最多 3 次，失败隔 2 分钟；3 次都失败 → 退出码 3，调用方改用本机 GPU 训练
    started = False
    for k in range(3):
        c.start_instance(i.instance_id)
        print(time.strftime("%H:%M:%S"), f"第 {k + 1} 次开机 …", flush=True)
        for _ in range(16):
            time.sleep(15)
            st = _status()
            if st == "Running":
                started = True
                break
            if st == "Failed":
                break
        if started:
            break
        print(time.strftime("%H:%M:%S"), f"第 {k + 1} 次开机失败（{_status()}）", flush=True)
        if k < 2:
            time.sleep(120)
    if not started:
        i = dsw.find(NAME)
        if i.status not in ("Stopped", "Failed"):
            c.stop_instance(i.instance_id, dm.StopInstanceRequest())
        print("3 次开机都失败 → 改用本机 GPU 训练（退出码 3）")
        return 3
    print(time.strftime("%H:%M:%S"), "已开机，等任务 DONE …", flush=True)
    t0, last = time.time(), None
    try:
        while time.time() - t0 < a.max_hours * 3600:
            st = _status()
            if st != last:
                print(time.strftime("%H:%M:%S"), "实例状态", st, flush=True); last = st
            if st in ("Failed", "Stopped"):
                print("实例意外停了 / 开机失败"); break
            if st == "Running" and _exists(b, JOB + "STARTED") is False:
                ran = ran if "ran" in dir() else time.time()
                if time.time() - ran > 12 * 60:
                    print("开机 12 分钟还没看到 STARTED（开机命令没执行？）→ 关机"); break
            if _exists(b, JOB + "DONE"):
                print(time.strftime("%H:%M:%S"), "任务结束：", b.get_object(JOB + "DONE").read().decode().strip(), flush=True)
                break
            time.sleep(30)
    finally:
        # **无论成败都关机**（避免计时浪费）
        for k in range(10):
            try:
                i = dsw.find(NAME)
                if i.status not in ("Stopped", "Stopping", "Failed"):
                    c.stop_instance(i.instance_id, dm.StopInstanceRequest())
                break
            except Exception as e:  # noqa: BLE001
                print("关机请求失败，重试：", type(e).__name__, flush=True); time.sleep(15)
        for _ in range(60):
            if _status() in ("Stopped", "Failed"):
                break
            time.sleep(10)
        print(time.strftime("%H:%M:%S"), "关机后状态：", _status(), flush=True)
    if b.object_exists(JOB + "out.log"):
        (P / "build" / f"dsw_{a.run}.log").write_bytes(b.get_object(JOB + "out.log").read())
        for line in (P / "build" / f"dsw_{a.run}.log").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("epoch", "TEST", "TRAIN_SECONDS", "torch", "Traceback")) or "Error" in line:
                print("  ", line[:200])
    if not (b.object_exists(JOB + "DONE") and b.get_object(JOB + "DONE").read().decode().strip() == "ok"):
        print("任务没有成功，日志 build/dsw_%s.log" % a.run); return 1
    import oss2
    dst = P / "data/laya/runs" / a.run
    pre = f"laya/dsw_out/{a.run}/"
    for ob in oss2.ObjectIterator(b, prefix=pre):
        if ob.key.endswith("/"):
            continue
        f = dst / ob.key[len(pre):]
        f.parent.mkdir(parents=True, exist_ok=True)
        b.get_object_to_file(ob.key, str(f))
    print("已取回", dst, "checkpoint OK" if (dst / "checkpoint/model.safetensors").exists() else "（没有 checkpoint）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    p = sub.add_parser("run")
    p.add_argument("--run", required=True); p.add_argument("--ver", required=True)
    p.add_argument("--labels", required=True); p.add_argument("--init", required=True)
    p.add_argument("--train-args", default=""); p.add_argument("--max-hours", type=float, default=4.0)
    a = ap.parse_args()
    return setup() if a.cmd == "setup" else run(a)


if __name__ == "__main__":
    sys.exit(main())
