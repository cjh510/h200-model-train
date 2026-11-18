import re, statistics, sys
from pathlib import Path
PEAK=989.4; N=8.03e9; WARM=5
RE=re.compile(r"^w(?P<world>\d+)_8B_tp(?P<tp>\d+)_pp(?P<pp>\d+)_mbs(?P<mbs>\d+)_s(?P<seq>\d+)_gb(?P<gbs>\d+)_(?P<prec>\w+)")
pat_ms=re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
pat_tf=re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
pat_loss=re.compile(r"(?:lm loss|label loss):\s*([0-9.eE+-]+)")
pat_gn=re.compile(r"grad norm:\s*([0-9.eE+-]+)")
pat_nan=re.compile(r"number of nan iterations:\s*([0-9]+)")
rows=[]
for p in sorted(Path("logs").glob("w8_*.log")):
    m=RE.match(p.stem)
    if not m: continue
    f=m.groupdict(); txt=p.read_text(errors="ignore")
    its=[l for l in txt.splitlines() if "elapsed time per iteration" in l]
    ms=[float(pat_ms.search(l).group(1)) for l in its]
    tf=[float(pat_tf.search(l).group(1)) for l in its]
    gn=[float(pat_gn.search(l).group(1)) for l in its if pat_gn.search(l)]
    loss=[float(pat_loss.search(l).group(1)) for l in its if pat_loss.search(l)]
    nan=sum(int(pat_nan.search(l).group(1)) for l in its if pat_nan.search(l))
    tail=ms[WARM:]; tt=tf[WARM:]
    step=statistics.mean(tail)/1000
    seq=int(f["seq"]); gbs=int(f["gbs"]); world=int(f["world"])
    ours=6*N*gbs*seq/(step*world*PEAK*1e12)*100
    mega=statistics.mean(tt)/PEAK*100
    rows.append(dict(name=f"{f['tp']}/{f['pp']}/{f['prec']}",seq=seq,mbs=int(f["mbs"]),gbs=gbs,
        step=step,msd=statistics.pstdev(tail)/statistics.mean(tail)*100,
        tok=gbs*seq/world/step,ours=ours,mega=mega,attn=mega/ours if ours else 0,
        gn=statistics.mean(gn[WARM:]) if gn else 0,loss=loss[-1] if loss else 0,nan=nan))
rows.sort(key=lambda r:-r["mega"])
print(f"{'档(tp/pp/精)':<16}{'seq':>8}{'mbs':>4}{'GBS':>5}{'step s':>9}{'抖动%':>7}{'tok/s/GPU':>10}{'MFU-6N':>8}{'MFU-Mega':>9}{'attn/6N':>9}{'gradnorm':>10}{'loss':>7}{'nan':>4}")
for r in rows:
    print(f"{r['name']:<16}{r['seq']:>8,}{r['mbs']:>4}{r['gbs']:>5}{r['step']:>9.3f}{r['msd']:>7.1f}{r['tok']:>10,.0f}"
          f"{r['ours']:>7.1f}%{r['mega']:>8.1f}%{r['attn']:>8.2f}x{r['gn']:>10.2f}{r['loss']:>7.2f}{r['nan']:>4}")
