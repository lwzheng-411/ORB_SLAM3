#!/bin/bash
# =============================================================================
# 实验：验证不同共视关键帧数量对 SLAM 精度的影响
# 环境变量 LBA_MAX_COVIS 控制共视关键帧数量限制
# =============================================================================

set -e

cd /home/zlw/End2End/ORB_SLAM3

# 编译
echo "正在编译 ORB_SLAM3..."
cd build
make -j4
cd ..
echo "编译完成。"

# 实验配置
DATASET="${DATASET_NAME:-MH01}"
DATASET_PATH="dataset/${DATASET}"
# NOTE (English): To resume an interrupted run, set COVIS_RESULTS_DIR to an existing folder.
if [ -n "${COVIS_RESULTS_DIR:-}" ]; then
    RESULTS_DIR="${COVIS_RESULTS_DIR}"
else
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    RESULTS_DIR="experiments/covis_results_${DATASET}_${RUN_ID}"
fi
mkdir -p "$RESULTS_DIR"

# 不同的共视关键帧限制
# NOTE (English): Include small limits to determine the minimum safe window size.
COVIS_LIMITS=(0 1 2 3 4 5 10 15 20 30)  # 0 = 不限制

echo ""
echo "============================================================"
echo "实验：共视关键帧数量对精度的影响"
echo "数据集: $DATASET"
echo "结果目录: $RESULTS_DIR"
echo "============================================================"
echo ""

for LIMIT in "${COVIS_LIMITS[@]}"; do
    if [ "$LIMIT" -eq 0 ]; then
        LIMIT_NAME="unlimited"
    else
        LIMIT_NAME="max${LIMIT}"
    fi
    
    # Resume: skip finished limits unless FORCE_RERUN=1
    if [ "${FORCE_RERUN:-0}" != "1" ] && [ -s "$RESULTS_DIR/traj_${LIMIT_NAME}.txt" ]; then
        echo "----------------------------------------"
        echo "跳过: LBA_MAX_COVIS=$LIMIT ($LIMIT_NAME) - 已存在 traj_${LIMIT_NAME}.txt"
        echo "----------------------------------------"
        echo ""
        continue
    fi

    echo "----------------------------------------"
    echo "运行: LBA_MAX_COVIS=$LIMIT ($LIMIT_NAME)"
    echo "----------------------------------------"
    
    # 设置环境变量
    export LBA_MAX_COVIS=$LIMIT
    export USE_HW_SOLVER=0
    export DUMP_JSON=0
    export DUMP_BASELINE=0
    export SLAM_NO_VIEWER=1
    
    # 运行 SLAM (EuRoC format outputs: f_<tag>.txt / kf_<tag>.txt)
    # NOTE (English): Use a unique tag per LIMIT so outputs are not overwritten.
    TAG="covis_${DATASET}_${LIMIT_NAME}"
    FRAMES_OUT="f_${TAG}.txt"
    KF_OUT="kf_${TAG}.txt"
    
    # 准备时间戳文件
    TIMES_FILE="$RESULTS_DIR/times_${DATASET}.txt"
    if [ ! -s "$TIMES_FILE" ]; then
        ls dataset/${DATASET}/mav0/cam0/data | sed 's/\.png$//' | sort > "$TIMES_FILE"
    fi
    
    # 运行 Mono-Inertial SLAM
    ./Examples/Monocular-Inertial/mono_inertial_euroc \
        Vocabulary/ORBvoc.txt \
        Examples/Monocular-Inertial/EuRoC.yaml \
        "$DATASET_PATH" \
        "$TIMES_FILE" \
        "$TAG" 2>&1 | tee "$RESULTS_DIR/log_${LIMIT_NAME}.txt" || true
    
    # 保存轨迹
    if [ -f "$FRAMES_OUT" ]; then
        cp "$FRAMES_OUT" "$RESULTS_DIR/traj_${LIMIT_NAME}.txt"
        echo "轨迹保存到: $RESULTS_DIR/traj_${LIMIT_NAME}.txt"
    else
        echo "WARN: 未找到输出文件: $FRAMES_OUT"
    fi
    if [ -f "$KF_OUT" ]; then
        cp "$KF_OUT" "$RESULTS_DIR/kf_traj_${LIMIT_NAME}.txt"
    else
        echo "WARN: 未找到输出文件: $KF_OUT"
    fi
    
    # 保存 LBA 统计
    if [ -f "LBA_Stats.txt" ]; then
        cp LBA_Stats.txt "$RESULTS_DIR/lba_stats_${LIMIT_NAME}.txt"
    fi
    
    echo ""
done

# 评估精度
echo "============================================================"
echo "精度评估 (ATE - Absolute Trajectory Error)"
echo "============================================================"
echo ""

# Ground truth
GT_FILE="dataset/${DATASET}/mav0/state_groundtruth_estimate0/data.csv"

python3 << EOF
import os
import numpy as np

results_dir = "$RESULTS_DIR"
gt_file = "$GT_FILE"

def _parse_ns(ts_str: str) -> int:
    # NOTE (English): EuRoC trajectory files are written with setprecision(6) and may include ".000000".
    # We parse the integer nanoseconds part to avoid float precision loss at ~1e18.
    if "." in ts_str:
        ts_str = ts_str.split(".", 1)[0]
    return int(ts_str)

def load_euroc_traj_xyz(filename):
    """Load EuRoC format trajectory: ts(ns) tx ty tz qx qy qz qw"""
    if not os.path.exists(filename):
        return None
    ts = []
    xyz = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                ts.append(_parse_ns(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                continue
    if not ts:
        return None
    ts = np.array(ts, dtype=np.int64)
    xyz = np.array(xyz, dtype=np.float64)
    return ts, xyz

def load_euroc_gt_xyz(filename):
    """Load EuRoC GT: timestamp(ns), p_RS_R_x,y,z,..."""
    if not os.path.exists(filename):
        return None
    ts = []
    xyz = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ts.append(int(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                continue
    if not ts:
        return None
    return np.array(ts, dtype=np.int64), np.array(xyz, dtype=np.float64)

def match_by_timestamp(ts_a, xyz_a, ts_b, xyz_b, max_dt_ns=2_000_000_00):
    """Nearest-neighbor match A->B by timestamp within tolerance (default 0.2s)."""
    # NOTE (English): We keep it simple and robust; arrays are assumed sorted by time.
    idx_b = 0
    pairs_a = []
    pairs_b = []
    for i in range(len(ts_a)):
        t = ts_a[i]
        while idx_b + 1 < len(ts_b) and abs(ts_b[idx_b + 1] - t) <= abs(ts_b[idx_b] - t):
            idx_b += 1
        if abs(ts_b[idx_b] - t) <= max_dt_ns:
            pairs_a.append(xyz_a[i])
            pairs_b.append(xyz_b[idx_b])
    if len(pairs_a) < 10:
        return None, None
    return np.array(pairs_a), np.array(pairs_b)

def umeyama_se3(X, Y):
    """Compute SE3 (R,t) aligning Y to X: minimize ||X - (R Y + t)||."""
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    Xc = X - mu_X
    Yc = Y - mu_Y
    S = (Yc.T @ Xc) / X.shape[0]
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_X - R @ mu_Y
    return R, t

def ate_rmse(est_xyz, gt_xyz):
    err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean": float(np.mean(err)),
        "std": float(np.std(err)),
        "max": float(np.max(err)),
        "count": int(len(err)),
    }

# Load ground truth
gt = load_euroc_gt_xyz(gt_file)
if gt is None:
    print("无法加载 ground truth")
else:
    gt_ts, gt_xyz = gt
    print(f"{'配置':<15} {'RMSE (m)':<12} {'Mean (m)':<12} {'Std (m)':<12} {'Max (m)':<12} {'点数':<10}")
    print("-" * 75)
    
    # NOTE (English): Keep python list in sync with bash COVIS_LIMITS.
    for limit in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30]:
        name = "unlimited" if limit == 0 else f"max{limit}"
        traj_file = os.path.join(results_dir, f"traj_{name}.txt")
        traj = load_euroc_traj_xyz(traj_file)
        if traj is None:
            print(f"{name:<15} 无轨迹文件/空文件")
            continue
        ts_e, xyz_e = traj

        # Match and align
        Xe, Xg = match_by_timestamp(ts_e, xyz_e, gt_ts, gt_xyz, max_dt_ns=20_000_000)  # 20ms
        if Xe is None:
            print(f"{name:<15} 时间戳匹配失败")
            continue
        R, t = umeyama_se3(Xg, Xe)  # align est->gt: Xg ≈ R*Xe + t
        Xe_aligned = (Xe @ R.T) + t
        m = ate_rmse(Xe_aligned, Xg)
        print(f"{name:<15} {m['rmse']:<12.4f} {m['mean']:<12.4f} {m['std']:<12.4f} {m['max']:<12.4f} {m['count']:<10}")

print()
EOF

echo ""
echo "实验完成！结果保存在: $RESULTS_DIR"
