#!/usr/bin/env python3
"""FineVideo per-second object finder (YOLO11l, 5 classes).

python run.py --input <folder of .parquet> --output results.jsonl

Design: one worker process per GPU. Each worker has a decode thread (PyAV, pts-based
seconds) feeding a queue and an inference loop (batched, FP16, Ultralytics defaults).
Output is written to <output>.part<g>, progress to <output>.done<g> (resumable), and
merged into <output> at the end. Default --stride 1 follows the spec exactly.
"""
import argparse, glob, io, json, math, os, queue, sys, threading, time
import multiprocessing as mp

NAMES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_IDS = [0, 2, 16, 63, 67]
ID2IDX = {c: i for i, c in enumerate(COCO_IDS)}
DEFAULT_INPUT = "/kaggle/input/datasets/musabsarmad/finevideo/data"


def _rot_k(stream, frame):
    """np.rot90 k for the video's rotation tag (UNTESTED sign: verify on a rotated video)."""
    try:
        r = getattr(frame, "rotation", 0)  # display-matrix angle, CCW degrees
        if r:
            return int(round(r / 90)) % 4
    except Exception:
        pass
    try:
        m = stream.metadata.get("rotate")  # legacy tag, CW degrees
        if m:
            return (-int(round(float(m) / 90))) % 4
    except Exception:
        pass
    return 0


def decode_frames(data, stride):
    """Yield (second, BGR ndarray or None if skipped by stride). Seconds are pts-based,
    counted from the first decoded frame."""
    import av
    import numpy as np
    c = av.open(io.BytesIO(data))
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate) if s.average_rate else 25.0
        k = None
        t0 = None
        for n, f in enumerate(c.decode(s)):
            t = f.time if f.time is not None else n / fps
            if t0 is None:
                t0 = t
            sec = max(0, int(math.floor(t - t0 + 1e-6)))
            if n % stride:
                yield sec, None
                continue
            if k is None:
                k = _rot_k(s, f)
            arr = f.to_ndarray(format="bgr24")
            if k:
                arr = np.ascontiguousarray(np.rot90(arr, k))
            yield sec, arr
    finally:
        c.close()


def producer(files, q, a, done):
    import pyarrow.parquet as pq
    nvid = 0
    for path in files:
        name = os.path.basename(path)
        row = 0
        try:
            pf = pq.ParquetFile(path)
            for b in pf.iter_batches(batch_size=1, columns=["mp4"]):
                vid = f"{name}:{row}"
                row += 1
                if vid in done:
                    continue
                if a.max_videos and nvid >= a.max_videos:
                    q.put(("stop",))
                    return
                nvid += 1
                data = b.column(0)[0].as_py()
                if not data:
                    q.put(("error", vid, "could not decode video: empty"))
                    continue
                q.put(("begin", vid))
                maxsec, nfr, batch, bsec, err = -1, 0, [], [], None
                try:
                    for sec, arr in decode_frames(data, a.stride):
                        nfr += 1
                        maxsec = max(maxsec, sec)
                        if arr is not None:
                            batch.append(arr)
                            bsec.append(sec)
                            if len(batch) == a.batch:
                                q.put(("batch", vid, batch, bsec))
                                batch, bsec = [], []
                    if batch:
                        q.put(("batch", vid, batch, bsec))
                    if nfr == 0:
                        err = "could not decode video"
                except Exception as e:
                    if batch:
                        q.put(("batch", vid, batch, bsec))
                    err = (f"could not decode video: {e}" if nfr == 0 else f"decode failed: {e}")[:200]
                q.put(("end", vid, maxsec, err))
        except Exception as e:
            q.put(("error", f"{name}:{row}", f"could not read parquet: {type(e).__name__}: {e}"[:200]))
    q.put(("stop",))


def make_line(vid, sec, rec):
    if not rec or rec[0] == 0:
        return {"video": vid, "second": sec, "objects": [], "people": 0}
    n, cc, pp = rec
    objs = [NAMES[i] for i in range(5) if cc[i] * 2 >= n]
    pp = sorted(pp)
    m = len(pp)
    people = pp[m // 2] if m % 2 else (pp[m // 2 - 1] + pp[m // 2]) // 2
    return {"video": vid, "second": sec, "objects": objs, "people": people}


def run_worker(g, ngpu, a):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    dev = f"cuda:{g}" if torch.cuda.is_available() else "cpu"
    half = dev != "cpu"
    files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))[g::ngpu]
    part, donep = f"{a.output}.part{g}", f"{a.output}.done{g}"
    done = set(open(donep).read().split()) if os.path.exists(donep) else set()
    if os.path.exists(part):  # keep only lines of fully finished videos
        keep = []
        for ln in open(part):
            try:
                if json.loads(ln)["video"] in done:
                    keep.append(ln if ln.endswith("\n") else ln + "\n")
            except Exception:
                pass
        open(part, "w").writelines(keep)
    out, dn = open(part, "a"), open(donep, "a")
    dump = open(f"{a.dump_frames}.{g}", "a") if a.dump_frames else None
    model = YOLO(a.weights)

    q = queue.Queue(maxsize=6)
    threading.Thread(target=producer, args=(files, q, a, done), daemon=True).start()

    st, consec, nfr, nvid = None, 0, 0, 0
    t_start = last = time.time()
    t_wait = t_inf = 0.0
    while True:
        w = time.time()
        item = q.get()
        t_wait += time.time() - w
        k = item[0]
        if k == "stop":
            break
        if k == "begin":
            st = {"secs": {}, "err": None}
            continue
        if k == "batch":
            _, vid, frames, secs = item
            if st["err"]:
                continue
            try:
                t = time.time()
                res = model.predict(frames, imgsz=640, conf=0.4, iou=0.7, half=half,
                                    device=dev, classes=COCO_IDS, verbose=False)
                for r, sec in zip(res, secs):
                    cnt = [0] * 5
                    if r.boxes is not None:
                        for c in r.boxes.cls.tolist():
                            cnt[ID2IDX[int(c)]] += 1
                    rec = st["secs"].setdefault(sec, [0, [0] * 5, []])
                    rec[0] += 1
                    rec[2].append(cnt[0])
                    for i in range(5):
                        rec[1][i] += 1 if cnt[i] else 0
                    if dump:
                        dump.write(json.dumps({"v": vid, "s": sec, "c": cnt}) + "\n")
                t_inf += time.time() - t
                nfr += len(frames)
            except Exception as e:
                st["err"] = f"inference failed: {type(e).__name__}: {e}"[:200]
        else:  # "end" or "error"
            vid = item[1]
            if k == "error":
                err, lines = item[2], []
            else:
                err = item[3] or st["err"]
                secs = st["secs"]
                top = (max(secs) if secs else -1) if err else max(item[2], max(secs) if secs else -1)
                lines = [make_line(vid, s, secs.get(s)) for s in range(top + 1)]
            if err:
                lines.append({"video": vid, "error": err})
            out.write("".join(json.dumps(l) + "\n" for l in lines))
            out.flush()
            dn.write(vid + "\n")
            dn.flush()
            if dump:
                dump.flush()
            nvid += 1
            consec = consec + 1 if err else 0
            if consec >= 20:
                print(f"[gpu{g}] 20 consecutive failures, aborting", flush=True)
                sys.exit(2)
        now = time.time()
        if now - last > 30:
            last = now
            print(f"[gpu{g}] videos={nvid} frames={nfr} fps={nfr/(now-t_start):.0f} "
                  f"wait_for_decode={t_wait:.0f}s inference={t_inf:.0f}s", flush=True)
    el = time.time() - t_start
    print(f"[gpu{g}] DONE videos={nvid} frames={nfr} fps={nfr/max(el,1e-9):.0f} "
          f"wait_for_decode={t_wait:.0f}s inference={t_inf:.0f}s elapsed={el:.0f}s", flush=True)


def default_weights():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11l.pt")
    return here if os.path.exists(here) else "yolo11l.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--weights", default=default_weights())
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (spec = 1)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--max-videos", type=int, default=0, help="per worker, for timing tests")
    ap.add_argument("--dump-frames", default="", help="prefix for per-frame count dump (stride 1 only)")
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if a.dump_frames and a.stride != 1:
        sys.exit("--dump-frames requires --stride 1")
    if not os.path.exists(a.weights):
        sys.exit(f"weights file not found: {a.weights} (run the preparation "
                 "command in the README first; there is no internet at run time)")
    import torch
    if not torch.cuda.is_available() and not a.allow_cpu:
        sys.exit("CUDA is not available; refusing to run on CPU "
                 "(use --allow-cpu to override)")
    ngpu = a.gpus or max(1, torch.cuda.device_count())
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=run_worker, args=(g, ngpu, a)) for g in range(ngpu)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    tmp = a.output + ".tmp"
    with open(tmp, "w") as fo:
        for g in range(ngpu):
            pth = f"{a.output}.part{g}"
            if os.path.exists(pth):
                with open(pth) as fi:
                    for ln in fi:
                        fo.write(ln if ln.endswith("\n") else ln + "\n")
    os.replace(tmp, a.output)
    sys.exit(1 if any(p.exitcode for p in ps) else 0)


if __name__ == "__main__":
    main()
