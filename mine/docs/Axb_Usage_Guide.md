# Axb 统一因子 SIMD 使用指南

## 架构概述

**Axb.sv** 是一个统一的因子线性化与白化模块，支持三种因子类型：
- **Camera Factor**: 单目重投影 (2x9: 3 landmark + 6 pose)
- **IMU Factor**: Pose3-Between (6x12: 6 pose_i + 6 pose_j)
- **Prior Factor**: Pose3先验 (6x6: 1 pose)

**核心特性**:
- 单一因子输入端口 + 类型选择码
- 共享浮点PE池 (16 mul, 12 add, 8 sub, 1 sqrt, 1 div, 1 atan2)
- FSM微程序自动调度运算，输出行白化后的 A、b

---

## 端口说明

### 因子输入端口
```systemverilog
input  wire                 factor_valid,    // 拉高以下发一个因子
output reg                  factor_ready,    // 低表示忙，高表示可接受
input  wire [1:0]           factor_type,     // 0=Camera, 1=IMU, 2=Prior
input  wire [7:0]           factor_row_base, // 该因子在全局矩阵的起始行
input  wire [2:0]           factor_rows,     // 行数 (2 or 6)
input  wire [7:0]           col_var0,        // pose_i 或 landmark 列基址
input  wire [7:0]           col_var1,        // pose_j 或 landmark 列基址
input  wire [31:0]          alpha [0:5],     // 行白化系数 = w_robust / sigma
```

### 几何/量测载荷（联合字段）
```systemverilog
// Camera使用
input  wire [31:0]          cam_fx, cam_fy, cam_cx, cam_cy,
input  wire [31:0]          cam_z_u, cam_z_v,
input  wire [31:0]          cam_Pc_X, cam_Pc_Y, cam_Pc_Z, cam_Pc_invZ,
input  wire [31:0]          cam_Rcw [0:2][0:2],

// IMU/Prior使用
input  wire [31:0]          R0 [0:2][0:2],  // Ri (IMU) or Rp (Prior)
input  wire [31:0]          R1 [0:2][0:2],  // Rj (IMU only)
input  wire [31:0]          Rd [0:2][0:2],  // ΔR (IMU only)
input  wire [31:0]          t0 [0:2],       // ti (IMU) or tp (Prior)
input  wire [31:0]          t1 [0:2],       // tj (IMU only)
input  wire [31:0]          td [0:2],       // Δt (IMU only)
```

---

## 使用流程

### 1. 下发一个 Camera 因子
```cpp
// CPU侧准备
Eigen::Matrix3f Rcw = pose_i.rotationMatrix();
Eigen::Vector3f tcw = pose_i.translation();
Eigen::Vector3f lw = landmark.position();
Eigen::Vector3f Pc = Rcw * lw + tcw;
float invZ = 1.0f / Pc.z();
Eigen::Vector2f z_meas = observation.pixel();

// 计算 Huber 权重（CPU侧）
Eigen::Vector2f r = z_meas - project(Pc, K);
Eigen::Vector2f r_hat = r / sigma_pixel; // 简单白化
float rho = r_hat.norm();
float w = (rho <= huber_delta) ? 1.0f : (huber_delta / rho);

// 行系数
float alpha0 = w / sigma_pixel;
float alpha1 = w / sigma_pixel;

// 下发到硬件
axb.factor_valid = 1;
axb.factor_type = 0; // Camera
axb.factor_row_base = current_row;
axb.factor_rows = 2;
axb.col_var0 = pose_col_base;   // 6 cols for pose
axb.col_var1 = landmark_col_base; // 3 cols for landmark
axb.alpha[0] = alpha0;
axb.alpha[1] = alpha1;
axb.cam_fx = K.fx;
axb.cam_fy = K.fy;
axb.cam_cx = K.cx;
axb.cam_cy = K.cy;
axb.cam_z_u = z_meas.x();
axb.cam_z_v = z_meas.y();
axb.cam_Pc_X = Pc.x();
axb.cam_Pc_Y = Pc.y();
axb.cam_Pc_Z = Pc.z();
axb.cam_Pc_invZ = invZ;
for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
        axb.cam_Rcw[i][j] = Rcw(i,j);

// 等待 factor_ready 再下发下一个因子
while (!axb.factor_ready) wait_clock();
```

### 2. 下发一个 IMU 因子
```cpp
// 从GTSAM预积分获取
gtsam::Rot3 deltaR = pim.deltaRij();
gtsam::Vector3 deltaP = pim.deltaPij();
gtsam::Vector3 deltaT = deltaP + v_i * dt + 0.5 * g_i * dt * dt; // 在i帧

// 当前线性化点
Eigen::Matrix3f Ri = pose_i.rotationMatrix();
Eigen::Vector3f ti = pose_i.translation();
Eigen::Matrix3f Rj = pose_j.rotationMatrix();
Eigen::Vector3f tj = pose_j.translation();

// 行系数（IMU不加鲁棒核）
for (int k=0; k<6; k++)
    alpha[k] = 1.0f / sigma_imu[k]; // sigma_imu从预积分协方差映射

// 下发
axb.factor_valid = 1;
axb.factor_type = 1; // IMU
axb.factor_row_base = current_row;
axb.factor_rows = 6;
axb.col_var0 = pose_i_col_base;
axb.col_var1 = pose_j_col_base;
for (int k=0; k<6; k++) axb.alpha[k] = alpha[k];
for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
        axb.R0[i][j] = Ri(i,j);
        axb.R1[i][j] = Rj(i,j);
        axb.Rd[i][j] = deltaR.matrix()(i,j);
    }
    axb.t0[i] = ti[i];
    axb.t1[i] = tj[i];
    axb.td[i] = deltaT[i];
}
```

### 3. 下发一个 Prior 因子
```cpp
// 先验
Eigen::Matrix3f Rp = prior_pose.rotationMatrix();
Eigen::Vector3f tp = prior_pose.translation();

// 行系数（先验强度）
for (int k=0; k<6; k++)
    alpha[k] = 1.0f / sigma_prior[k];

axb.factor_valid = 1;
axb.factor_type = 2; // Prior
axb.factor_row_base = current_row;
axb.factor_rows = 6;
axb.col_var0 = pose_col_base;
for (int k=0; k<6; k++) axb.alpha[k] = alpha[k];
for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++)
        axb.R0[i][j] = Rp(i,j);
    axb.t0[i] = tp[i];
}
```

### 4. 行发射到 QR
```cpp
// 所有因子下发完毕后
axb.cfg_m = total_rows;
axb.cfg_n = total_cols;
axb.emit_start = 1;

// 硬件开始逐行输出
while (!axb.emit_done) {
    if (axb.row_valid && qr.row_ready) {
        // 取一行 A[row_idx][:] 和 b[row_idx]
        for (int j=0; j<N; j++)
            A_row[j] = axb.row_A[j];
        b_val = axb.row_b;
        // 送给QR模块
        qr.input(A_row, b_val);
    }
    wait_clock();
}
```

---

## 资源节省估算

| 项目 | 原三通道架构 | 统一口+共享PE | 节省比例 |
|------|-------------|--------------|---------|
| 乘法器 | ~90个 | 16个 | 82% |
| 加法器 | ~60个 | 12个 | 80% |
| 减法器 | ~40个 | 8个 | 80% |
| sqrt | 3个 | 1个 | 67% |
| atan2 | 3个 | 1个 | 67% |
| 总LUT | ~150K | ~30K | 80% |

**延迟**: 单因子从 2–5拍 → 15–25拍（取决于类型）

---

## 注意事项

1. **同时只处理一个因子**: factor_valid 拉高后需等 factor_ready 再变高才能下发下一个。
2. **列映射由CPU维护**: col_var0/1 是列基址，硬件按偏移写入。
3. **行白化系数 alpha**: 需CPU预先计算 = 鲁棒权 × 1/sigma，硬件在写入时乘到 J 和 r。
4. **坐标系**: Camera 用 Tcw(world→cam)，IMU 用 i 帧坐标，Prior 用 world。
5. **IMU 的 deltaTij**: 必须在CPU/GTSAM预处理里用 v、g、bias 处理好，传入时已是可用的相对平移。

---

## 调试建议

- 使用 `load_*` 端口可直接写任意 A[row][col]/b[row]，便于单元测试。
- 先用简单场景（1个Prior + 1个Camera）验证流水正确性。
- 检查 alpha 是否正确应用：b_buf[row] 应为 alpha[k] * r[k]。

