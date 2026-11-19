"""1 → 7단계를 한 번에 실행

    python run_all.py           # 전체 (약 5~10분, 2단계 롤링 CV 가 가장 오래 걸림)
    python run_all.py 6 7       # 원하는 단계만 (앞 단계 결과가 output/ 에 있어야 함)

로그는 output/run_log.txt 에도 남음.
"""
import importlib
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

STEPS = {"1": "step1_데이터정리", "2": "step2_매각확률", "3": "step3_낙찰가배수",
         "4": "step4_경매시험", "5": "step5_배당회수액", "6": "step6_시뮬레이션", "7": "step7_시뮬레이터"}


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for t in self.s: t.write(x)
    def flush(self):
        for t in self.s: t.flush()


if __name__ == "__main__":
    from common import OUT_ROOT
    picks = [a for a in sys.argv[1:] if a in STEPS] or list(STEPS)
    sys.argv = sys.argv[:1]
    log = open(OUT_ROOT / "run_log.txt", "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    t0 = time.time()
    print(f"\n######## 실행 {time.strftime('%Y-%m-%d %H:%M')} 단계 {picks}")
    for k in picks:
        importlib.import_module(STEPS[k]).main()
        print()
    print(f"완료 ({time.time() - t0:.0f}초) → 결과는 output/ 폴더")
