# Axb 模块架构说明

## 1. 整体结构

```
Axb.sv (单文件, 1140行)
├── 参数配置
│   ├── NUM_MUL = 16  (乘法器数量)
│   ├── NUM_ADD = 12  (加法器数量)
│   └── NUM_SUB = 8   (减法器数量)
│
├── 共享硬件资源池 (59-143行)
│   ├── 16x DW_fp_mult    : mul_a/b/z[0:15], mul_en/vld[0:15]
│   ├── 12x DW_fp_add     : add_a/b/z[0:11], add_en/vld[0:11]
│   ├── 8x  DW_fp_sub     : sub_a/b/z[0:7],  sub_en/vld[0:7]
│   ├── 1x  DW_fp_sqrt    : sqrt_in, sqrt_out, sqrt_en/vld
│   ├── 1x  DW_fp_div     : div_num/den, div_quot, div_en/vld
│   ├── 1x  Atan2Cordic   : atan_y/x, atan_theta, atan_en/vld
│   └── 1x  Skew          : skew_x/y/z, skew_K[3][3]
│
├── 临时寄存器 (176行)
│   └── reg [31:0] t[0:63]  // 64个寄存器，所有factor共享
│
└── 单体状态机 (188-1138行)
    ├── S0: 空闲，接收 factor (203-211行)
    ├── S_DISPATCH: 引擎分发 (214-221行)
    ├── Camera (SC1-SC11): 226-390行 (165行, ✓已用宏)
    ├── IMU    (SI1-SI34 + SM1-SM3): 393-881行 (488行, ✗未用宏)
    └── Prior  (SP1-SP21): 890-1131行 (241行, ✓已用宏)
```

## 2. 硬件复用机制

### 2.1 时分复用 (Time-Division Multiplexing)

所有三种 factor **顺序执行**，共享同一组物理PE：

```
时间轴:
t=0    : S0 状态，等待 factor_valid
t=1    : S_DISPATCH 根据 factor_type 选择引擎
t=2-16 : Camera 执行 (SC1-SC11), 使用 mul_a[0..15]
t=17   : Camera 完成，返回 S0，硬件资源释放
t=18   : 下一个 factor (IMU) 开始
t=19-98: IMU 执行 (SI1-SI34), 复用相同的 mul_a[0..15]
t=99   : IMU 完成，返回 S0
...
```

### 2.2 复用证据

#### 证据1：共享信号线
所有 factor 都操作同一组信号：
```verilog
// Camera 用 mul_a[0]
`MUL(0, cam_fx, cam_Pc_invZ)  // SC1

// IMU 用同一个 mul_a[0] (不同时刻)
mul_a[0]<=R0[0][0]; mul_b[0]<=t[0]; mul_en[0]<=1'b1;  // SI2

// Prior 用同一个 mul_a[0] (更晚时刻)
`MUL(0,Rp[0][0],t[0])  // SP2
```

#### 证据2：互斥状态
- Camera 状态 (SC1-SC11)
- IMU 状态 (SI1-SI34)
- Prior 状态 (SP1-SP21)

这些状态**永不重叠**，由单体 FSM 保证互斥。

#### 证据3：面积节省
如果为每个 factor 独立配置硬件：
- 方案A (独立): 16×3 = **48个乘法器**
- 方案B (共享): 16×1 = **16个乘法器** ✓当前实现
- **面积节省**: 66% (2/3 的硬件省了)

## 3. 当前问题

### 问题1：代码简化不一致
- ✓ Camera (SC1-SC11): 使用宏 `` `MUL/ADD/SUB``
- ✗ **IMU (SI1-SI34)**: 原始写法 `mul_a[0]<=...; mul_en[0]<=1'b1;` (冗长)
- ✓ Prior (SP1-SP21): 使用宏

**影响**: 
- 可读性差
- 维护困难
- 风格不统一

### 问题2：非真正的层次化子模块
当前结构：
```
单体 FSM
├── eng 寄存器 (标记 ENG_CAM/ENG_IMU/ENG_PRI)
└── 大 case 语句 (所有状态在一个 always 块)
```

**不是**层次化子模块（没有独立的 CameraEngine.sv 等文件）。

真正的层次化子模块应该是：
```
Axb (顶层)
├── CameraEngine (子模块, 独立.sv文件)
├── ImuEngine    (子模块, 独立.sv文件)
└── PriorEngine  (子模块, 独立.sv文件)
```

**当前设计选择单文件**的原因：
1. 用户要求："都写到Axb这个文件里面吧，别分出去"
2. 单文件更容易查看完整逻辑
3. 避免过度抽象

### 问题3：硬件复用不够明显
虽然实际在复用，但代码没有明确体现：
- 没有注释说明 "此处使用 mul_a[0] 与 Camera 共享"
- 没有资源仲裁逻辑 (因为状态互斥，不需要仲裁)

## 4. 改进方案

### 方案A：宏简化（最小改动，推荐）
把 IMU 的 488行原始代码改用宏，减少到约 200行。

**优点**:
- 风格统一
- 可读性提升
- 不改变结构

**缺点**:
- 仍是单体FSM
- 不够"层次化"

### 方案B：真正的层次化子模块
创建 `CameraEngine.sv`, `ImuEngine.sv`, `PriorEngine.sv`，每个子模块有独立接口：

```verilog
module CameraEngine (
    input  clk, rst,
    input  start,
    output done,
    // 共享PE接口
    output [15:0] mul_req,
    input  [15:0] mul_grant,
    output [31:0] mul_a_out[0:15],
    ...
);
```

**优点**:
- 真正的模块化
- 可独立测试
- 清晰的硬件复用接口

**缺点**:
- 需要仲裁器 (arbiter)
- 增加约 500行代码
- 用户不想分文件

### 方案C：内联子FSM + 注释（折中）
保持单文件，但在每个 factor 前加详细注释说明硬件复用：

```verilog
// ============================================================
// IMU Engine (内联子FSM)
// 硬件复用: mul_a[0:15], add_a[0:11], sub_a[0:7]
// 与 Camera/Prior 时分复用，由状态互斥保证
// ============================================================
```

并把 IMU 改用宏简化。

## 5. 推荐行动

1. **立即**: 把 IMU (SI1-SI34) 全部改用宏
2. **文档**: 在 Axb.sv 顶部增加架构注释
3. **验证**: 跑 linter 确保无语法错误

## 6. 硬件资源数量确定方法

参见 Axb.sv 第 8-10 行参数。

确定依据：
1. **静态分析**: 遍历所有状态，找峰值并行度
   - Camera SC10: 14 MUL + 3 ADD (白化两行)
   - IMU SI29: 13 MUL (白化一行)
   - Prior SP16: 13 MUL

2. **取最大值 + 余量**:
   - NUM_MUL = 16 (覆盖14，留余量)
   - NUM_ADD = 12 (够用于归约树)
   - NUM_SUB = 8  (减法不堆叠，8个够)

3. **工程权衡**:
   - 面积: 16 MUL ≈ 30K LUT
   - 性能: 瓶颈在 CORDIC/div，不在乘法器
   - 结论: 16 个是性价比最优点

## 7. 总结

| 项目 | 当前状态 | 应该是 |
|------|---------|--------|
| **代码简化** | Camera/Prior用宏, IMU未用 | 全部用宏 |
| **层次化** | 单体FSM + eng标记 | 单体FSM (用户要求) |
| **硬件复用** | 实际在复用，但不明显 | 增加注释说明 |
| **文件结构** | 单文件 1140行 | 保持单文件 (用户要求) |

**下一步**: 简化 IMU 代码 (488行 → 200行)

