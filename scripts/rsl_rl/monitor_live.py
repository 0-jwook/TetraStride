#!/usr/bin/env python3
"""
실시간 학습 모니터 — TFEvents 파일을 직접 파싱해서 새 iteration이 생길 때마다 출력.
사용법:
  python scripts/rsl_rl/monitor_live.py            # 자동으로 최신 실험 폴더 탐색
  python scripts/rsl_rl/monitor_live.py <tfevents_path>
"""

import os
import sys
import struct
import time
import glob
from datetime import datetime
from collections import defaultdict

WATCH_TAGS = {
    "Train/mean_reward":          "reward(클수록 잘함)",
    "Train/mean_episode_length":  "ep_len(에피소드 길이)",
    "Policy/mean_std":            "std(0.1~0.3 건강)",
    "Loss/value":                 "value_loss(높으면 불안정)",
    "Loss/surrogate":             "surrogate_loss",
    "Loss/entropy":               "entropy",
    "Loss/learning_rate":         "lr",
}

EXPERIMENT_BASES = [
    "logs/rsl_rl/spot_micro_stance",
    "logs/rsl_rl/spot_micro_trot",
    "logs/rsl_rl/spot_micro_locomotion",
]


def find_latest_tfevents(repo_root):
    best = None
    for rel in EXPERIMENT_BASES:
        base = os.path.join(repo_root, rel)
        pattern = os.path.join(base, "*", "events.out.tfevents.*")
        for f in glob.glob(pattern):
            if best is None or os.path.getmtime(f) > os.path.getmtime(best):
                best = f
    return best


def read_records(path, offset):
    records = []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            while True:
                len_buf = f.read(8)
                if len(len_buf) < 8:
                    break
                data_len = struct.unpack("<Q", len_buf)[0]
                f.read(4)  # crc
                data = f.read(data_len)
                f.read(4)  # crc
                if len(data) < data_len:
                    break
                records.append(data)
                offset += 8 + 4 + data_len + 4
    except Exception:
        pass
    return records, offset


def read_varint(buf, pos):
    result, shift = 0, 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def read_field(buf, pos):
    if pos >= len(buf):
        return None, None, pos
    tag_wire, pos = read_varint(buf, pos)
    fn = tag_wire >> 3
    wt = tag_wire & 0x7
    if wt == 0:
        val, pos = read_varint(buf, pos)
        return fn, val, pos
    elif wt == 2:
        length, pos = read_varint(buf, pos)
        val = buf[pos:pos + length]; pos += length
        return fn, val, pos
    elif wt == 5:
        val = buf[pos:pos + 4]; pos += 4
        return fn, val, pos
    elif wt == 1:
        val = buf[pos:pos + 8]; pos += 8
        return fn, val, pos
    return None, None, len(buf)


def parse_event(data):
    """Event proto → (step, tag, scalar) or (None, None, None)."""
    step = None
    tag = None
    scalar = None
    pos = 0
    while pos < len(data):
        fn, val, pos = read_field(data, pos)
        if fn is None:
            break
        # field 2 = step (int64, varint)
        if fn == 2 and isinstance(val, int):
            step = val
        # field 5 = summary (bytes)
        elif fn == 5 and isinstance(val, (bytes, bytearray)):
            sp = 0
            while sp < len(val):
                sfn, sval, sp = read_field(val, sp)
                if sfn is None:
                    break
                # field 1 of Summary = Value (bytes)
                if sfn == 1 and isinstance(sval, (bytes, bytearray)):
                    vp = 0
                    while vp < len(sval):
                        vfn, vval, vp = read_field(sval, vp)
                        if vfn is None:
                            break
                        # field 1 = tag (string)
                        if vfn == 1 and isinstance(vval, (bytes, bytearray)):
                            tag = vval.decode("utf-8", errors="replace")
                        # field 2 = simple_value (float32, fixed32 → 4 bytes)
                        elif vfn == 2 and isinstance(vval, (bytes, bytearray)) and len(vval) == 4:
                            scalar = struct.unpack("<f", vval)[0]
    return step, tag, scalar


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))

    if len(sys.argv) > 1:
        tfevents = sys.argv[1]
    else:
        tfevents = find_latest_tfevents(repo_root)

    if not tfevents:
        print("TFEvents 파일을 찾을 수 없습니다.")
        sys.exit(1)

    exp_name = os.path.basename(os.path.dirname(os.path.dirname(tfevents)))
    run_name = os.path.basename(os.path.dirname(tfevents))
    print(f"[모니터] {exp_name}/{run_name}")
    print(f"[모니터] 시작: {datetime.now().strftime('%H:%M:%S')}\n")

    offset = 0
    # step → {tag: scalar}
    step_buf = defaultdict(dict)
    last_printed_step = -1
    # step → first_seen_time (for timeout fallback)
    step_first_seen = {}

    while True:
        # Stage 전환 시 새 파일 감지
        if len(sys.argv) <= 1:
            latest = find_latest_tfevents(repo_root)
            if latest and latest != tfevents and os.path.getmtime(latest) > os.path.getmtime(tfevents):
                exp_name = os.path.basename(os.path.dirname(os.path.dirname(latest)))
                run_name = os.path.basename(os.path.dirname(latest))
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Stage 전환: {exp_name}/{run_name} ===\n")
                tfevents = latest
                offset = 0
                step_buf.clear()
                last_printed_step = -1

        records, offset = read_records(tfevents, offset)

        now = time.time()
        for data in records:
            step, tag, scalar = parse_event(data)
            if step is None or tag is None or scalar is None:
                continue
            if tag not in WATCH_TAGS:
                continue
            step_buf[step][tag] = scalar
            if step not in step_first_seen:
                step_first_seen[step] = now

        # 완성된 step 출력 (last_printed_step 이후 것만)
        for step in sorted(step_buf.keys()):
            if step <= last_printed_step:
                continue
            vals = step_buf[step]
            has_main = "Train/mean_reward" in vals and "Train/mean_episode_length" in vals
            # timeout: 5초 지났으면 있는 것만으로 출력
            timed_out = (now - step_first_seen.get(step, now)) > 5.0
            if has_main or (timed_out and len(vals) > 0):
                ts = datetime.now().strftime("%H:%M:%S")
                parts = []
                for tag in ["Train/mean_reward", "Train/mean_episode_length",
                            "Policy/mean_std", "Loss/value", "Loss/entropy"]:
                    if tag in vals:
                        label = WATCH_TAGS[tag]
                        parts.append(f"{label}: {vals[tag]:.4f}")
                if parts:
                    print(f"[{ts}] iter={step:5d}  " + "  |  ".join(parts), flush=True)
                last_printed_step = step

        # 버퍼에서 오래된 step 정리
        old_steps = [s for s in step_buf if s < last_printed_step - 10]
        for s in old_steps:
            del step_buf[s]
            step_first_seen.pop(s, None)

        time.sleep(1)


if __name__ == "__main__":
    main()
