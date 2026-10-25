"""talent_ref_noise 结果的报告：配对检验结论（talent_eval.verdict）两组种子是否一致。"""
import json, sys
sys.path.insert(0, "tools/sim")
import talent_eval as T
D = json.load(open(sys.argv[1], encoding="utf-8"))
both = agree = only = 0
for i, x in D.items():
    a, b = T.verdict(x["a"]), T.verdict(x["b"])
    if a and b:
        both += 1; agree += a == b
    elif a or b:
        only += 1
print(f"配对检验：两次都有结论 {both}（一致 {agree}，相反 {both - agree}），只有一次有结论 {only}，共 {len(D)}")
