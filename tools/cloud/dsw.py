"""DSW 实例管理（2026-10-15：DSW 每月 250 计算时免费，用完要及时关机）。
  .venv-pai/Scripts/python.exe tools/cloud/dsw.py list|status|start|stop [--name laya-train]
"""
import argparse, sys, time
from pai.session import get_default_session
from alibabacloud_tea_openapi import models as om
from alibabacloud_pai_dsw20220101.client import Client
from alibabacloud_pai_dsw20220101 import models as dm
from alibabacloud_credentials.client import Client as CredClient


def client():
    s = get_default_session()
    return s, Client(om.Config(credential=CredClient(s.credential_config), region_id=s.region_id,
                               endpoint=f"pai-dsw.{s.region_id}.aliyuncs.com"))


def instances():
    s, c = client()
    return c.list_instances(dm.ListInstancesRequest(workspace_id=str(s.workspace_id), page_size=50)).body.instances or []


def find(name):
    xs = [i for i in instances() if i.instance_name == name]
    return xs[0] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "status", "start", "stop"])
    ap.add_argument("--name", default="laya-train")
    a = ap.parse_args()
    if a.cmd == "list":
        for i in instances():
            print(i.instance_id, i.instance_name, i.status, i.ecs_spec, i.payment_type, (i.reason_message or "")[:120])
        return 0
    i = find(a.name)
    if not i:
        print("没有实例", a.name); return 1
    _, c = client()
    if a.cmd == "status":
        print(i.instance_id, i.status, i.ecs_spec, (i.reason_message or "")[:200]); return 0
    if a.cmd == "start":
        c.start_instance(i.instance_id); print("已请求开机", i.instance_id); return 0
    if a.cmd == "stop":
        c.stop_instance(i.instance_id, dm.StopInstanceRequest())
        for _ in range(40):
            st = find(a.name).status
            if st in ("Stopped", "Failed"):
                break
            time.sleep(15)
        print("关机后状态：", find(a.name).status); return 0


if __name__ == "__main__":
    sys.exit(main())
