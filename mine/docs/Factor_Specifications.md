# 因子规格完整说明

## 单个因子的列数统计

| 因子类型 | 连接变量 | 每变量列数 | 总列数 | 行数 |
|---------|---------|-----------|-------|------|
| Camera  | Pose + Landmark | 6 + 3 | **9列** | 2行 |
| IMU     | Pose_i + Pose_j | 6 + 6 | **12列** | 6行 |
| Prior   | Pose | 6 | **6列** | 6行 |

**单个因子最多 12 列**（IMU Between）

---

## 稀疏块结构与 QR 消元顺序（仅 Pose3）

### 1) 变量布局策略

**推荐布局**（按时间+类型分组）:
```
列 0..5:     x1 (Pose3, 第1帧)
列 6..11:    x2 (Pose3, 第2帧)
列 12..17:   x3 (Pose3, 第3帧)
...
列 K..K+2:   y1 (Landmark 1)
列 K+3..K+5: y2 (Landmark 2)
...
```

**为什么这样布局**:
- Pose 按时间顺序排列，便于滑窗边缘化（老帧在前）
- Landmark 集中在后，便于 Schur complement 消元

### 2) QR 消元顺序（两种策略）

**策略A: Schur Complement（推荐，适合大量路标）**
1. **第一轮**: 对 Landmark 列做 QR，消除路标自由度
   - 输入: A[:, 0:K-1] 为 Pose 块, A[:, K:N-1] 为 Landmark 块
   - Givens/Householder 旋转只作用在 Landmark 列，得到 R 上三角
   - 同时更新 Pose 块与 b，得到"边缘化路标后的 Pose-only 系统"
2. **第二轮**: 对剩余 Pose 列做 QR
   - 此时矩阵已是 Pose-only，尺寸更小
   - 按列 0..K-1 顺序消元（或按滑窗倒序）
3. **回代**: 先解 Pose，再解 Landmark

**策略B: 直接顺序消元（简单，适合小规模）**
- 直接对列 0..N-1 做 QR，不分组
- 优点: 硬件实现简单（流式 Givens/Householder）
- 缺点: 不利用稀疏性，计算量稍高

**你的硬件建议**: 先用策略B验证功能，后续优化可实现策略A的分组消元。

### 3) 稀疏性示例（图1场景）

**因子→列的稀疏模式**:
```
f1 (Camera x1,y1):  行0..1,  列 [0..5, 18..20]   非零块: pose1(6) + lm1(3)
f2 (Camera x1,y2):  行2..3,  列 [0..5, 21..23]   非零块: pose1(6) + lm2(3)
f3 (Camera x2,y2):  行4..5,  列 [6..11, 21..23]  非零块: pose2(6) + lm2(3)
f4 (IMU x1→x2):     行6..11, 列 [0..11]          非零块: pose1(6) + pose2(6)
f5 (IMU x2→x3):     行12..17,列 [6..17]          非零块: pose2(6) + pose3(6)
f6 (Prior x1):      行18..23,列 [0..5]           非零块: pose1(6)
```

**矩阵形态** (24行×24列，`*`表示非零块):
```
        0..5  6..11 12..17 18..20 21..23
行0-1   [*]    .      .     [*]     .      f1
行2-3   [*]    .      .      .     [*]     f2
行4-5    .    [*]     .      .     [*]     f3
行6-11  [*]   [*]     .      .      .      f4
行12-17  .    [*]    [*]     .      .      f5
行18-23 [*]    .      .      .      .      f6
```

**QR 后的 R 矩阵**（上三角）:
```
        0..5  6..11 12..17 18..20 21..23
        [R11] [R12] [R13]  [R14]  [R15]
         .    [R22] [R23]  [R24]  [R25]
         .     .    [R33]  [R34]  [R35]
         .     .     .     [R44]  [R46]
         .     .     .      .     [R55]
```

---

## 正则方程构建（可选，直接 QR 更稳）

若你的硬件先构建 H = J^T J、g = J^T b，再做 Cholesky：

**块稀疏 H**（对称，只存上三角）:
```
H_pose1_pose1 = J_f1[:, 0:6]^T J_f1[:, 0:6] + J_f2[:, 0:6]^T J_f2[:, 0:6] 
              + J_f4[0:6, 0:6]^T J_f4[0:6, 0:6] + J_f6^T J_f6
H_pose1_pose2 = J_f4[0:6, 6:12]^T J_f4[6:12, 0:6]
...
```

**不推荐**: 因为 H 会损失数值精度；直接对白化后的 J 做 QR 更稳定，且你的硬件 Axb 已按行输出 J、b。

---

## 列数上限与硬件参数

**单因子最大列数**: 12（IMU）
**全局矩阵最大列数**: 取决于滑窗大小
- 例: 5帧滑窗 + 20个路标 → N = 5×6 + 20×3 = 90列
- 你的 `Axb.sv` 参数 `N` 应设为全局列数上限（如 N=96 或 128）

**单因子最大行数**: 6（IMU/Prior）
**全局矩阵最大行数**: 取决于因子数量
- 例: 20个Camera(2行) + 4个IMU(6行) + 1个Prior(6行) = 40+24+6 = 70行
- 你的 `Axb.sv` 参数 `M` 应设为行数上限（如 M=96 或 128）

---

## 因子列块详细映射

### Camera Factor (2行×9列)
```
jacobian[0][0..2]:  ∂r_u/∂landmark (Jl row0, 3列)
jacobian[0][3..8]:  ∂r_u/∂pose (Jpose row0, 6列: trans 3 + rot 3)
jacobian[1][0..2]:  ∂r_v/∂landmark (Jl row1, 3列)
jacobian[1][3..8]:  ∂r_v/∂pose (Jpose row1, 6列)
```
写入时:
- `matrix_buf[factor_row_base+0][col_var1+0..2]` ← Jl row0
- `matrix_buf[factor_row_base+0][col_var0+0..5]` ← Jpose row0
- `matrix_buf[factor_row_base+1][col_var1+0..2]` ← Jl row1
- `matrix_buf[factor_row_base+1][col_var0+0..5]` ← Jpose row1

### IMU Factor (6行×12列)
```
jacobian[0..2][0..5]:   ∂rt/∂pose_i (平移残差对 pose_i，6列: trans 3 + rot 3)
jacobian[0..2][6..11]:  ∂rt/∂pose_j (平移残差对 pose_j，6列)
jacobian[3..5][0..5]:   ∂rR/∂pose_i (旋转残差对 pose_i，6列)
jacobian[3..5][6..11]:  ∂rR/∂pose_j (旋转残差对 pose_j，6列)
```
写入时每行:
- `matrix_buf[factor_row_base+k][col_var0+0..5]` ← 对 pose_i 的块
- `matrix_buf[factor_row_base+k][col_var1+0..5]` ← 对 pose_j 的块

### Prior Factor (6行×6列)
```
jacobian[0..2][0..2]:  ∂et/∂t_i = Rp^T (平移块)
jacobian[0..2][3..5]:  ∂et/∂φ_i = 0
jacobian[3..5][0..2]:  ∂eR/∂t_i = 0
jacobian[3..5][3..5]:  ∂eR/∂φ_i = I
```
写入时:
- `matrix_buf[factor_row_base+k][col_var0+0..5]` ← 单 pose 的6列

---

## 硬件参数建议

```systemverilog
parameter N = 128,  // 支持 ~20帧 + 10路标
parameter M = 128,  // 支持 ~40个Camera + 10个IMU + Prior
parameter NUM_MUL = 16,
parameter NUM_ADD = 12,
parameter NUM_SUB = 8
```

若你的 FPGA/ASIC 资源紧张，可降低 N、M；若要支持更大场景（如100帧滑窗），需增大到 N=256+。

