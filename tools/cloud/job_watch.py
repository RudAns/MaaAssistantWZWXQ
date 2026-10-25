"""盯一个 PAI 训练任务直到结束，打印状态变化，结束后把日志写到 build/<job>.log。
  .venv-pai/Scripts/python.exe tools/cloud/job_watch.py <job_id>"""
import sys, time
from pai.job import TrainingJob
from pai.session import get_default_session
jid = sys.argv[1]
s = get_default_session()
last = None
t0 = time.time()
while True:
    st = TrainingJob.get(jid).model_dump().get("Status")
    if st != last:
        print(time.strftime("%H:%M:%S"), st, flush=True); last = st
    if st in ("Succeed", "Succeeded", "Failed", "Stopped"):
        break
    if time.time() - t0 > 3 * 3600:
        print("watch timeout"); break
    time.sleep(20)
lines, page = [], 1
while True:
    r = s.training_job_api.list_logs(jid, page_size=200, page_number=page)
    items = list(getattr(r, "items", r) or [])
    if not items:
        break
    lines += [str(x) for x in items]; page += 1
    if len(items) < 200:
        break
open(f"build/{jid}.log", "w", encoding="utf-8").write("\n".join(lines))
print("logs lines", len(lines))
