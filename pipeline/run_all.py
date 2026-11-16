"""1 → 2 → 3 → 4단계를 한 번에 실행 (약 20초)

    python run_all.py           # 전체
    python run_all.py 2 3       # 원하는 단계만 (예: 엑셀만 바뀌었으면 3)
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import step1_bid_model
import step2_calibration
import step3_dividend
import build_dashboard

STEPS = {"1": step1_bid_model.main, "2": step2_calibration.main, "3": step3_dividend.main, "4": build_dashboard.main}

if __name__ == "__main__":
    picks = [a for a in sys.argv[1:] if a in STEPS] or list(STEPS)
    sys.argv = sys.argv[:1]          # 각 단계에 옵션이 넘어가지 않게
    t0 = time.time()
    for k in picks:
        STEPS[k]()
        print()
    print(f"완료 ({time.time() - t0:.0f}초) → 결과는 output/ 폴더")
