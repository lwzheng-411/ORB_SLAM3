# Axb 完整实现指南

## 当前状态总结

### 已完成
1. **Camera 因子**: 完整实现投影+雅可比+CORDIC+行白化+行流输出（11个状态）
2. **IMU 因子**: 框架完成，rt 计算路径完成；rR(Log)、严格雅可比（M/Kvi/RtK）需完整扩展
3. **Prior 因子**: 平移残差完成；旋转 Log 需扩展
4. **行流接口**: row_out_* 完整定义，支持 panel/trail 分段输出
5. **共享PE池**: 16mul, 12add, 8sub, sqrt/div/atan/skew 各1个

###完整 IMU 状态流程（详细文档）

**总计约 28 个状态，80+ 时钟周期**

#### 阶段1: 计算 vi = Ri^T (tj - ti)
- ST_IMU_V1: sub tj-ti
- ST_IMU_V2: 9 mul for Ri^T*v
- ST_IMU_V3: 6 add for vi[0..2]

#### 阶段2: 计算 rt = Rd^T*(vi - td)
- ST_IMU_VTILDE: sub vi-td
- ST_IMU_RT1-RT3: 9 mul + 6 add for Rd^T*vtilde

#### 阶段3: 计算 C = Ri^T * Rj (9 elements)
- ST_IMU_C1-C5: 9 mul + 12 add (分批完成9个矩阵元素)
  - C[0][0] = dot(Ri[:,0], Rj[:,0]) = Ri[0][0]*Rj[0][0] + Ri[1][0]*Rj[1][0] + Ri[2][0]*Rj[2][0]
  - ... (共9个)

#### 阶段4: 计算 Re = Rd^T * C (9 elements)
- ST_IMU_RE1-RE4: 27 mul + 18 add
  - Re[i][j] = sum_k Rd[k][i]*C[k][j]

#### 阶段5: 计算 Log(Re) 用 CORDIC
- ST_IMU_LOG_VEE: vee(Re - Re^T)/2 → v[3]
  - v[0] = (Re[2][1] - Re[1][2])/2
  - v[1] = (Re[0][2] - Re[2][0])/2
  - v[2] = (Re[1][0] - Re[0][1])/2
- ST_IMU_LOG_NORM: ||v|| via sqrt
  - s = 0.5 * sqrt(v[0]^2 + v[1]^2 + v[2]^2)
- ST_IMU_LOG_TR: trace(Re) → cos term
  - c = 0.5 * (tr(Re) - 1)
- ST_IMU_LOG_ATAN: atan2(s, c) → theta
- ST_IMU_LOG_SCALE: theta / (2*s) * v → rR[0..2]
  - 需处理 s≈0 的小角情况（theta≈0时，rR≈v）

#### 阶段6: 计算雅可比 M = Rd^T * Ri^T (for translation jac)
- ST_IMU_M1-M4: 27 mul + 18 add

#### 阶段7: 计算 Kvi = skew(vi) 和 RtK = Rd^T * Kvi (for rotation part of trans jac)
- ST_IMU_SKEWVI: skew(vi) → Kvi[3][3]
- ST_IMU_RTK1-RTK3: 27 mul + 18 add for Rd^T * Kvi

#### 阶段8: 行白化与逐行输出
- ST_IMU_WHITEN: 对6行×12列 jac 和 res 乘 alpha (72 mul，需分批)
- ST_IMU_EMIT1-EMIT6: 逐行输出 row_out_*

### 简化版实现策略（当前 Axb_complete.sv）

**为降低复杂度，简化项**:
1. rR 暂填 0（后续可补完整 Log）
2. M 雅可比简化为 ±I（实际应为 ±Rd^T*Ri^T）
3. 旋转对 θi 的项暂为 -I（实际应含 RtK）

**如何扩展为完整版**:
- 在 ST_IMU_C3-C5 完成所有 C 元素
- 在 ST_IMU_RE1-RE4 完成 Re
- 在 ST_IMU_LOG_* 按上述5步完成 CORDIC log
- 在 ST_IMU_M1-M4 完成 M
- 在 ST_IMU_SKEWVI/RTK* 完成 RtK
- 在 ST_IMU_WHITEN 分批对所有元素白化（每批16个mul）
- 在 ST_IMU_EMIT1-6 逐行输出（每状态1行）

---

## RowBuffer 完整逻辑

### 核心功能
1. **列映射表管理**: panel 固定在 local[0:p-1]；trails 按 trail_index 映射到 local[p+6*k : p+6*k+5]
2. **致密行拼装**: 收 row_in_* 片段，按 col_base 放进对应列段，未占用段填0
3. **喂 systolicarray**: bundle_end 后，逐行发 sa_row_*；systolic 只对前 p 列做 QR
4. **R块收集**: 从 sa_out_* 收 R 行，分块保存 R11(p×p), R12(p×(L-p)), R22((L-p)×(L-p)), z1(p), z2(L-p)
5. **新因子生成**: 输出 R12/R22/z2 给 CPU，作为"合并后的新因子"连接 trail 变量

### 列映射详细算法

**输入**: row_in_panel_col_base, row_in_trail_col_base

**方法A（简单）**: CPU 预先告知 trail_index（0,1,2,...），RowBuffer 直接用
- local_col(panel) = 0..p-1
- local_col(trail_k) = p + 6*trail_index + j (j=0..5)

**方法B（自动推断）**: RowBuffer 维护"已见 trail_base"表，首次见某 trail_base 时分配下一个 slot
- slot_map[trail_base] = next_free_slot
- local_col = p + 6*slot_map[trail_base] + j

**推荐**: 方法A，在 Axb 输入时增加 trail_index[3:0]，RowBuffer 直接用

---

## CPU 侧按变量消元调用样例

```cpp
// 示例：消除 landmark y1
// y1 被 f1(x1观测), f2(x2观测), f3(x3观测) 连接

// 1. Bundle begin
axb.bundle_begin = 1;
axb.bundle_panel_cols = 3;       // landmark 3列
axb.bundle_trails_count = 3;     // x1, x2, x3
axb.bundle_local_cols = 3 + 6*3 = 21;
axb.bundle_id = elim_seq_id;
wait_clock();
axb.bundle_begin = 0;

// 2. 下发 f1 (y1-x1)
axb.factor_valid = 1;
axb.factor_type = 0; // Camera
axb.factor_row_base = 0; // 本地行0-1
axb.factor_rows = 2;
axb.col_var1 = 0;    // y1 panel在本地列0-2
axb.col_var0 = 3;    // x1 trail在本地列3-8
axb.alpha[0] = w1_row0 / sigma_pixel;
axb.alpha[1] = w1_row1 / sigma_pixel;
// ... cam_* 字段
wait_until(axb.factor_ready);

// 3. 下发 f2 (y1-x2)
axb.factor_valid = 1;
axb.factor_row_base = 2; // 本地行2-3
axb.col_var1 = 0;        // y1 仍在0-2
axb.col_var0 = 9;        // x2 在列9-14
wait_until(axb.factor_ready);

// 4. 下发 f3 (y1-x3)
axb.factor_valid = 1;
axb.factor_row_base = 4;
axb.col_var1 = 0;
axb.col_var0 = 15;       // x3 在列15-20
axb.bundle_end = 1;      // 最后一条因子
wait_until(axb.factor_ready);
axb.bundle_end = 0;

// 5. RowBuffer 自动拼成 6×21 致密矩阵，喂 systolicarray
// systolicarray 对列 0-2 (y1) 做 QR，更新列 3-20 (x1,x2,x3) 和 b

// 6. 从 RowBuffer 取新因子
wait_until(rowbuf.new_factor_valid);
new_factor_g1 = rowbuf.new_factor_R[3:5][3:20]; // R22块
new_factor_z1 = rowbuf.new_factor_z[3:5];
// g1 连接 (x1,x2,x3)，下一轮可与其他因子合并继续消

// 7. 保留 R11/z1 用于最终回代
R11_y1 = rowbuf.new_factor_R[0:2][0:2];
z1_y1 = rowbuf.new_factor_z[0:2];
backsubst_stack.push({y1_key, R11, z1});
```

---

## 下一步建议

### 选项A：扩展当前简化版
- 在 Axb_complete.sv 补全所有IMU/Prior的LOG与雅可比状态（需增加约100行状态case）
- 测试框架先用简化版验证接口，逐步替换为完整版

### 选项B：分模块设计
- 创建 CameraFactor.sv, ImuFactor.sv, PriorFactor.sv 各自包含完整微程序
- Axb_top 做选择器，根据 factor_type 路由到对应子模块
- 共享PE池作为公共资源，子模块通过仲裁器访问

### 选项C（推荐）：混合策略
- 当前 Axb_complete.sv 作为"功能原型"，Camera 完整、IMU/Prior 简化
- 单独提供 ImuFactorFull.sv 与 PriorFactorFull.sv 模块（含所有状态）
- 用 `generate` 条件编译选择简化版或完整版

---

## 资源与性能估算

### 简化版（当前）
- LUT: ~30K
- 延迟: Camera 15拍, IMU 20拍, Prior 18拍

### 完整版（含Log/严格Jac）
- LUT: ~35K (增加状态机+中间寄存)
- 延迟: Camera 15拍, IMU 60拍, Prior 45拍

### 吞吐
- 若每因子平均30拍，200MHz时钟 → 6.7M 因子/秒
- 实际受限于 RowBuffer/Systolic 的喂料与回收

---

## 测试建议

1. 单元测试：用固定输入验证 Camera/IMU/Prior 各自的 J、r 输出正确性
2. RowBuffer 测试：手动下发2-3个 Camera 因子，检查致密矩阵拼装
3. 端到端测试：用 EuRoC 一帧数据，实际调用 ORB-SLAM3→Axb→RowBuffer→Systolic→回代

---

## 需要你确认的点

1. 是否接受"简化版+文档"的方式，还是必须在代码里逐状态实现（会导致单文件超2000行）？
2. RowBuffer 的列映射用方法A（trail_index输入）还是方法B（自动推断）？
3. 是否需要我提供分模块版本（CameraFactor.sv等独立子模块）？
4. 优先补全哪个：IMU完整Log？RowBuffer完整列映射？CPU侧GTSAM预处理接口？

如果你选择"全部在代码里实现"，我会继续在 Axb_complete.sv 追加所有状态（预计再增加约1000行）。

