# 完整系统工作流程

## 系统架构

```
┌──────────────┐    ┌───────────────────┐    ┌─────────────┐
│  CPU (软件)  │───▶│  FPGA (硬件加速)  │───▶│  CPU (求解) │
│              │    │                   │    │             │
│ ORB-SLAM3    │    │ Axb.sv            │    │ 回代求解    │
│ GTSAM预积分  │    │ RowBuffer.sv      │    │ 位姿更新    │
│ 因子调度     │    │ SystolicArray.sv  │    │             │
└──────────────┘    └───────────────────┘    └─────────────┘
```

---

## 阶段1: 数据采集与预处理

### 输入数据
```
相机图像序列: Frame_0, Frame_1, Frame_2, ...
IMU数据流:    {acc[3], gyro[3], t} × 100Hz
GPS (可选):   {lat, lon, alt}
```

### 1.1 视觉特征提取（ORB-SLAM3）
```cpp
for (auto& frame : camera_frames) {
    // 提取ORB特征
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(frame, keypoints, descriptors);
    
    // 特征匹配
    vector<cv::DMatch> matches = matcher->match(desc_prev, desc_current);
    
    // 三角化或跟踪
    for (auto& match : matches) {
        CameraObservation obs;
        obs.landmark_id = landmark_map[match.trainIdx];
        obs.pose_id = current_frame_id;
        obs.pixel << keypoints[match.queryIdx].pt.x,
                     keypoints[match.queryIdx].pt.y;
        obs.sigma_pixel = 1.0;  // 像素噪声
        
        // 相机参数
        obs.fx = camera_K(0,0);
        obs.fy = camera_K(1,1);
        obs.cx = camera_K(0,2);
        obs.cy = camera_K(1,2);
        
        // 当前估计
        obs.Rcw = current_pose.R;
        obs.tcw = current_pose.t;
        obs.lw = landmarks[obs.landmark_id];
        
        camera_observations.push_back(obs);
    }
}
```

### 1.2 IMU预积分（GTSAM）
```cpp
ImuPreprocessorGTSAM imu_proc;

// 收集两帧间的IMU数据
vector<ImuMeasurement> imu_segment;
for (auto& imu : imu_buffer) {
    if (imu.t >= t_i && imu.t < t_j) {
        imu_segment.push_back({
            .acc = imu.acc,
            .gyro = imu.gyro,
            .dt = imu.dt
        });
    }
}

// 预积分
double dt_total = t_j - t_i;
auto imu_output = imu_proc.getRelativePose(
    imu_segment, 
    Ri_estimate,  // 当前旋转估计（用于重力补偿）
    dt_total
);

// 输出: {deltaR, deltaT, sigma[6]}
```

---

## 阶段2: 因子图构建

### 2.1 数据结构
```cpp
struct CameraObservation {
    int pose_id, landmark_id;
    Eigen::Vector2f pixel;        // 观测像素 [u, v]
    Eigen::Matrix3f Rcw;          // 相机姿态
    Eigen::Vector3f tcw, lw;      // 位置，路标
    float fx, fy, cx, cy;         // 内参
    float sigma_pixel;            // 噪声
};

struct ImuConstraint {
    int pose_i_id, pose_j_id;
    Eigen::Matrix3f deltaR;       // Ri^T * Rj
    Eigen::Vector3f deltaT;       // 相对平移（i系）
    Eigen::Matrix<float,6,1> sigma;  // [σ_t, σ_θ]
};

struct PriorConstraint {
    int pose_id;
    Eigen::Matrix3f Rp;           // 先验旋转
    Eigen::Vector3f tp;           // 先验平移
    Eigen::Matrix<float,6,1> sigma;
};
```

### 2.2 变量消元顺序
```
Sliding Window SLAM:
  1. 消除所有老路标 (landmark_0, landmark_1, ...)
  2. 消除老位姿 (pose_0, pose_1, ...)
  3. 保留最近10帧位姿（窗口）
  4. 求解窗口内的稠密系统
```

---

## 阶段3: FPGA硬件加速

### 3.1 消除一个路标（示例）

```cpp
VariableEliminationDriver driver;

// 消除 landmark_5，被 pose_1, pose_3, pose_7 观测
driver.eliminateLandmark(5, {
    Camera(lm5, pose1),
    Camera(lm5, pose3),
    Camera(lm5, pose7)
});

// 内部流程:
// 1. Bundle begin
// 2. 逐个下发 Camera 因子到 Axb
//    - factor_type = 0
//    - local_col_panel = 0 (landmark)
//    - local_col_trail = 3, 9, 15 (各个pose)
// 3. Bundle end
// 4. RowBuffer 拼装成 6行×21列 矩阵
// 5. SystolicArray QR分解
// 6. 返回 R11(3×3), R22(18×18), z1, z2
```

### 3.2 Axb.sv 处理流程

```
输入 Factor1: Camera(landmark_5, pose_1)
  ↓
SC1-SC3:  投影计算
  u_pred = fx * (Xc/Zc) + cx
  v_pred = fy * (Yc/Zc) + cy
  ↓
SC4:      残差
  r_u = z_u - u_pred
  r_v = z_v - v_pred
  ↓
SC5-SC8:  Jacobian计算
  ∂r/∂landmark = -fx/Zc * Rcw * [...]
  ∂r/∂pose = -fx/Zc * [I | -skew(Pc)] * [...]
  ↓
SC9-SC11: 白化 + 输出
  J_white = alpha * J
  r_white = alpha * r
  ↓
输出2行:
  row0: [J_lm[0:2], J_pose[0:5], r_u]
  row1: [J_lm[0:2], J_pose[0:5], r_v]
```

### 3.3 RowBuffer 矩阵拼装

```
收集所有因子的行:
     col: 0  1  2    3  4  5  6  7  8    9 10 11 12 13 14   15 16 17 18 19 20  b
         ┌─────────┬────────────────┬────────────────┬────────────────┬────┐
  r0    │ J_lm    │  J_pose1       │      0         │      0         │ ru │
  r1    │ J_lm    │  J_pose1       │      0         │      0         │ rv │
  r2    │ J_lm    │      0         │  J_pose3       │      0         │ ru │
  r3    │ J_lm    │      0         │  J_pose3       │      0         │ rv │
  r4    │ J_lm    │      0         │      0         │  J_pose7       │ ru │
  r5    │ J_lm    │      0         │      0         │  J_pose7       │ rv │
         └─────────┴────────────────┴────────────────┴────────────────┴────┘
          panel=3      trail0=6         trail1=6         trail2=6

逐行喂给 SystolicArray
```

### 3.4 SystolicArray QR分解

```
输入: A(6×21), b(6×1)
算法: Givens旋转 (CORDIC)
输出:
  R = ┌─────┬──────────────────┐     z = ┌────┐
      │ R11 │      R12         │         │ z1 │  3行
      ├─────┼──────────────────┤         ├────┤
      │  0  │      R22         │         │ z2 │  3行
      └─────┴──────────────────┘         └────┘
       3×3      3×18                      18×1
              18×18

R11: Panel对Panel的Jacobian (landmark对自己)
R12: Panel对Trail的Jacobian (landmark对poses)
R22: Trail对Trail的Jacobian (poses之间)
```

---

## 阶段4: CPU结果处理

### 4.1 回代求解
```cpp
// 求解 landmark_5
// R11 * Δlm = z1
Eigen::Vector3f delta_lm = R11.lu().solve(z1);
landmark_5_new = landmark_5_old + delta_lm;
```

### 4.2 边缘化（Schur补）
```cpp
// R22 和 z2 是边缘化后的新约束
// 连接 pose_1, pose_3, pose_7
NewFactor marginal_factor;
marginal_factor.R = R22;  // 18×18 (3个pose × 6DOF)
marginal_factor.z = z2;   // 18×1
marginal_factor.connected_vars = {pose_1, pose_3, pose_7};

// 下一轮消除pose时，作为新的约束使用
```

---

## 阶段5: 迭代与收敛

### 高斯-牛顿迭代
```cpp
for (int iter=0; iter<MAX_ITER; iter++) {
    // 1. 在当前估计点线性化
    auto [camera_obs, imu_factors, priors] = 
        linearizeAtCurrentEstimate();
    
    // 2. 变量消元
    runVariableElimination(camera_obs, imu_factors, priors);
    
    // 3. 更新估计
    updateEstimates(solution);
    
    // 4. 检查收敛
    if (delta.norm() < threshold) break;
}
```

---

## 完整代码示例

### 示例: 3帧SLAM

```cpp
// 初始化
VariableEliminationDriver driver;
driver.poses_R_[0] = Eigen::Matrix3f::Identity();
driver.poses_t_[0] = Eigen::Vector3f::Zero();
driver.landmarks_[1] = Eigen::Vector3f(1, 2, 5);

// === 迭代1: 消除 landmark_1 ===
std::vector<CameraObservation> lm1_obs = {
    {0, 1, {320, 240}, ...},  // pose0 观测 lm1
    {1, 1, {325, 242}, ...},  // pose1 观测 lm1
    {2, 1, {330, 245}, ...}   // pose2 观测 lm1
};
driver.eliminateLandmark(1, lm1_obs);
// 结果: lm1被消除，R22连接pose0,1,2

// === 迭代2: 消除 pose_0 ===
// IMU: pose0 → pose1
ImuConstraint imu_01 = driver.makeImuConstraint(0, 1, imu_data_01, dt_01);
// Prior: pose0固定
PriorConstraint prior_0 = {0, I, 0, sigma_prior};

driver.eliminatePose(0, {imu_01}, {prior_0});
// 结果: pose0被消除，R22连接pose1,2

// === 迭代3: 求解剩余变量 ===
Eigen::VectorXf final_x = solveDense({pose1, pose2});
auto solution = driver.backsubstitute(final_x);

// === 更新 ===
landmark_1 += solution[1];  // landmark
pose_0 = pose_0 * exp(solution[0]);  // pose
```

---

## 数据流详解

### Camera因子
```
输入:
├─ factor_type = 0
├─ local_col_panel = 0       (landmark在列0-2)
├─ local_col_trail = 9       (pose在列9-14)
├─ alpha[0:1] = {1/σ, 1/σ}
├─ cam_Pc = Rcw*lw + tcw     (CPU预计算)
├─ cam_Rcw, cam_z_u/v
└─ cam_fx/fy/cx/cy

处理 (Axb.sv, SC1-SC11, 15周期):
├─ 投影: u_pred = fx*(Xc/Zc)+cx
├─ 残差: r = z - pred
├─ Jacobian: ∂r/∂lm, ∂r/∂pose
└─ 白化: J*alpha, r*alpha

输出 (2行):
├─ row0: [J_lm, 0, 0, J_pose, 0, 0, r_u]
└─ row1: [J_lm, 0, 0, J_pose, 0, 0, r_v]
         └─3─┘       └──6──┘
```

### IMU因子
```
输入:
├─ factor_type = 1
├─ local_col_panel = 3       (pose_i在列3-8)
├─ local_col_trail = 9       (pose_j在列9-14)
├─ alpha[0:5] = {1/σ_t, 1/σ_θ}
├─ R0=Ri, R1=Rj, Rd=deltaR
└─ t0=ti, t1=tj, td=deltaT

处理 (Axb.sv, SI1-SI34, 80周期):
├─ vi = Ri^T * (tj-ti)
├─ rt = Rd^T * (vi-td)
├─ C = Ri^T * Rj
├─ Re = Rd^T * C
├─ rR = Log(Re)
├─ M = Rd^T * Ri^T
├─ RtK = Kvi * rt
└─ 白化输出

输出 (6行):
├─ row0-2: [-M, 0, M, 0, rt]  平移部分
└─ row3-5: [-rR, 0, rR, 0]    旋转部分
            └6┘   └6┘
```

### Prior因子
```
输入:
├─ factor_type = 2
├─ local_col_panel = 0       (pose在列0-5)
├─ local_col_trail = 0       (无trail)
├─ alpha[0:5] = {1/σ_t, 1/σ_θ}
├─ R0=Rp(先验), R1=Ri(当前)
└─ t0=tp, t1=ti

处理 (Axb.sv, SP1-SP21, 60周期):
├─ tdiff = Rp^T * (ti-tp)
├─ C = Rp^T * Ri
├─ rR = Log(C)
└─ 白化输出

输出 (6行):
├─ row0-2: [-I, 0, tdiff]  平移约束
└─ row3-5: [-I, 0, rR]     旋转约束
            └6┘
```

---

## 阶段3: QR分解与边缘化

### RowBuffer处理
```
输入: 所有因子的行（乱序到达）
处理:
  1. 根据 local_col_* 放到对应列
  2. 拼成致密矩阵
  3. bundle_end → 启动QR

输出:
  逐行喂给 SystolicArray
```

### SystolicArray QR
```
输入: A(m×n), b(m)
算法: Givens旋转 (Systolic CORDIC)
输出:
  R11(p×p), R12(p×l), R22(l×l)
  z1(p), z2(l)
  
  其中 p=panel列数, l=trail列数
```

---

## 阶段4: 回代与更新

### 4.1 回代求解
```cpp
// 从栈顶到栈底依次求解
for (auto& entry : backsub_stack_逆序) {
    // R11 * Δx = z1 - R12 * (已解变量)
    Eigen::VectorXf delta = entry.R11.lu().solve(entry.z1);
    solutions[entry.var_id] = delta;
}
```

### 4.2 变量更新
```cpp
// Landmark更新（加法）
landmark_new = landmark_old + Δlm

// Pose更新（李群右乘）
T_new = T_old * exp([Δt; Δθ])
      = [R_old, t_old] * [exp(Δθ), Δt]
      = [R_old*exp(Δθ), R_old*Δt + t_old]
```

---

## 性能指标

| 阶段 | 延迟 | 吞吐 |
|------|------|------|
| Axb (Camera) | 15周期 | 13M因子/秒 @ 200MHz |
| Axb (IMU) | 80周期 | 2.5M因子/秒 |
| Axb (Prior) | 60周期 | 3.3M因子/秒 |
| RowBuffer | O(1) | 流水线 |
| SystolicArray (32×32) | 63周期 | 3.2M矩阵/秒 |
| **端到端** | ~150周期/landmark | **~1.3M lm/秒** |

---

## 代码文件总览

| 文件 | 行数 | 功能 |
|------|------|------|
| **SW/ImuPreprocessor_GTSAM.cpp** | 100 | GTSAM IMU预积分 |
| **SW/VariableEliminationDriver.cpp** | 574 | 变量消元调度器 |
| **RTL/Axb.sv** | 1036 | 因子处理硬件 |
| **RTL/RowBuffer.sv** | 247 | 行缓冲与拼装 |
| **RTL/baseline.sv** | 449 | Systolic QR阵列 |

---

## 关键技术点

### 1. 硬件复用
```
16个乘法器时分复用:
  Camera用 → IMU用 → Prior用
  节省66%面积
```

### 2. 流水线
```
Axb → RowBuffer → SystolicArray
每个阶段可并行处理不同bundle
```

### 3. 松耦合IMU
```
GTSAM预积分 → 6D相对位姿 → Between约束
vs ORB-SLAM3的紧耦合IMU因子
```

### 4. 变量消元
```
稀疏QR → Schur补 → 边缘化
减少计算量 (O(n²) → O(n))
```

