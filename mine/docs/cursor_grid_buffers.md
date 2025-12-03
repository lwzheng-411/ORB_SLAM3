# 理清 grid buffers 处理流程
_Exported on 2025/11/9 at GMT+8 22:38:56 from Cursor (2.0.69)_

---

**User**

@cursor_update_process_for_grid_buffers.md 你越来越混乱了，能不能理一下

---

**Cursor**

好的，我来从头理清楚整个逻辑，用最简单直接的方式说明。

---

## 核心流程梳理

### 你的实际场景

1. **QR 分解每次产出什么**：
   - 每次消元一个变量（如 y1），得到**一行 g**（如 g1）
   - 这一行 g 包含**多个矩阵块**（如：y1 列的 R 块 + x1 列的方阵 + x2 列的方阵）
   - 每个块都有固定尺寸（如 6×6、2×6 等）

2. **需要存储什么**：
   - 把 g1 这一行的所有块存到 Total Buffer
   - 用 Block CSR 格式（节省空间）

3. **可能的并发**：
   - 多个 QR 单元可能同时算完，多个 g 行同时到达

---

## Encoder 的最简方案（适合你当前需求）

### 输入（Global Controller 逐块提交）

每个块提交时给出：
```systemverilog
- block_valid: 1
- block_rows, block_cols: 该块的尺寸（如 6×6）
- factor_id: 该块来自哪个因子（如 f1_id）
- block_col_idx: 该块对应哪个变量列（如 y1/x1/x2 的列索引）
- block_data[6][6]: 稠密矩阵值
- new_block_row: 该块是否是这一行 g 的第一个块（1/0）
- last_block_in_row: 该块是否是这一行 g 的最后一个块（1/0）
```

**就这些字段即可！** 不需要 row_block_id/next_row_block_id/last_block_overall。

### 输出（串行 CSR 流）

对每个块输出序列：
```
如果 new_block_row=1:
  → ROW_START 包（标记：这一行开始了，当前块总数是多少）
  
对每个块都输出：
  → META 包（记录：块尺寸 rows×cols，来自哪个 factor）
  → COL_INDEX 包（记录：该块对应哪个变量列）
  → VALUE 包 × (rows×cols) 个（块的所有数值，行主序）

如果 last_block_in_row=1:
  → ROW_END 包（标记：这一行结束了，总块数变成多少）
```

### 具体例子

**场景**：g1 行有 2 个块（y1 列的 6×6 R 块、x1 列的 6×6 方阵）

**输入**（分两次提交）：
```
第1块：
  block_rows=6, block_cols=6
  factor_id=f1
  block_col_idx=y1（假设 y1 是第 0 列）
  block_data=6×6 的 R 矩阵
  new_block_row=1, last_block_in_row=0

第2块：
  block_rows=6, block_cols=6
  factor_id=f1
  block_col_idx=x1（假设 x1 是第 1 列）
  block_data=6×6 的方阵
  new_block_row=0, last_block_in_row=1
```

**输出**（假设之前已有 5 个块）：
```
1. ROW_START(块计数=5)
2. META(rows=6, cols=6, factor_id=f1)
3. COL_INDEX(列索引=0，即 y1)
4. VALUE × 36 个（R 矩阵的 36 个浮点数）
5. META(rows=6, cols=6, factor_id=f1)
6. COL_INDEX(列索引=1，即 x1)
7. VALUE × 36 个（方阵的 36 个浮点数）
8. ROW_END(块计数=7)  ← 5 + 本行 2 块 = 7
```

存到 Total Buffer 后的效果：
- row_ptr 追加两个值：[..., 5, 7]
- col_idx 追加两个列号：[..., 0, 1]（即 y1, x1）
- values 追加 72 个浮点数

---

## 如果需要并行/乱序，怎么改？

### 问题
多个 QR 同时产出 g1、g2、g3...，它们可能乱序到达（如 g2 先到、g1 后到）

### 最小改动方案
在 Global Controller 给 encoder 的输入里加**一个字段**：
```
- assigned_row_slot: 预分配给该 g 行的"行槽位号"（0,1,2...）
```

**这就是 row_block_id 的作用**：它是一个**稠密的、连续的行槽位编号**，用来告诉 Total Buffer"不管你什么时候到，都往 row_ptr[i] 这个位置写"。

### 为什么不能直接用 factor_id (row_base)？

因为 factor_id 可能是：`f1=5, f3=12, f5=20, f8=35...`（离散的）

如果直接用它做 row_ptr 的索引：
- row_ptr[5], row_ptr[12], row_ptr[20], row_ptr[35]...
- 中间的 row_ptr[6~11]、[13~19] 等都是空洞，浪费存储
- Decoder 解码时不知道下一个有效的 row_ptr 索引是多少

**解决办法**：
- Global Controller 维护一个映射：
  - g1（来自 f1, factor_id=5）→ assigned_row_slot=0
  - g2（来自 f3, factor_id=12）→ assigned_row_slot=1
  - g3（来自 f5, factor_id=20）→ assigned_row_slot=2
- 发给 encoder 时带上这个 `assigned_row_slot`（这就是 row_block_id）
- Total Buffer 用这个稠密索引写 row_ptr[0], row_ptr[1], row_ptr[2]...

---

## 建议的最终接口（支持并行/乱序）

### Encoder 输入
```systemverilog
- block_rows, block_cols: 尺寸
- factor_id: 来自哪个因子（用于追溯）
- assigned_row_slot: 该行在 Total Buffer 的行槽位号（0,1,2...）
- block_col_idx: 该块对应的变量列
- block_data: 稠密值
- new_block_row, last_block_in_row: 行首/行尾标记
```

### Encoder 输出
```
ROW_START(row_slot=0, block_count=当前总块数)
→ META → COL_INDEX → VALUE×n
→ META → COL_INDEX → VALUE×m
→ ...
ROW_END(row_slot=1, block_count=新的总块数)
```

Total Buffer 收到后：
- 看到 `row_slot=0` → 写 `row_ptr[0] = block_count`
- 看到 `row_slot=1` → 写 `row_ptr[1] = block_count`
- 乱序到达也不怕，因为每个 g 行都有自己的 `assigned_row_slot`

---

## 总结

**最核心的一句话**：
- `factor_id`（row_base）是"物理标识"（来自哪个因子），用于追溯/调试
- `assigned_row_slot`（row_block_id）是"存储地址"（写到 row_ptr 的哪个位置），用于构建紧凑的 CSR 数组

**如果你只有一路 QR、顺序归档**：不需要 assigned_row_slot，用顺序追加即可。

**如果你有多路 QR、乱序到达**：需要 assigned_row_slot，让 Total Buffer 能随机写而不需排序。

**现在需要确认的是**：你的架构是单路还是多路？如果是多路，我们保留 row_block_id；如果是单路，我们删掉它，简化接口。

```systemverilog
- block_valid: 1
- block_rows, block_cols: 该块的尺寸（如 6×6）
- factor_id: 该块来自哪个因子（如 f1_id）
- block_col_idx: 该块对应哪个变量列（如 y1/x1/x2 的列索引）
- block_data[6][6]: 稠密矩阵值
- new_block_row: 该块是否是这一行 g 的第一个块（1/0）
- last_block_in_row: 该块是否是这一行 g 的最后一个块（1/0）
```

```plaintext
如果 new_block_row=1:
  → ROW_START 包（标记：这一行开始了，当前块总数是多少）
  
对每个块都输出：
  → META 包（记录：块尺寸 rows×cols，来自哪个 factor）
  → COL_INDEX 包（记录：该块对应哪个变量列）
  → VALUE 包 × (rows×cols) 个（块的所有数值，行主序）

如果 last_block_in_row=1:
  → ROW_END 包（标记：这一行结束了，总块数变成多少）
```

```plaintext
第1块：
  block_rows=6, block_cols=6
  factor_id=f1
  block_col_idx=y1（假设 y1 是第 0 列）
  block_data=6×6 的 R 矩阵
  new_block_row=1, last_block_in_row=0

第2块：
  block_rows=6, block_cols=6
  factor_id=f1
  block_col_idx=x1（假设 x1 是第 1 列）
  block_data=6×6 的方阵
  new_block_row=0, last_block_in_row=1
```

```plaintext
1. ROW_START(块计数=5)
2. META(rows=6, cols=6, factor_id=f1)
3. COL_INDEX(列索引=0，即 y1)
4. VALUE × 36 个（R 矩阵的 36 个浮点数）
5. META(rows=6, cols=6, factor_id=f1)
6. COL_INDEX(列索引=1，即 x1)
7. VALUE × 36 个（方阵的 36 个浮点数）
8. ROW_END(块计数=7)  ← 5 + 本行 2 块 = 7
```

```plaintext
- assigned_row_slot: 预分配给该 g 行的"行槽位号"（0,1,2...）
```

```systemverilog
- block_rows, block_cols: 尺寸
- factor_id: 来自哪个因子（用于追溯）
- assigned_row_slot: 该行在 Total Buffer 的行槽位号（0,1,2...）
- block_col_idx: 该块对应的变量列
- block_data: 稠密值
- new_block_row, last_block_in_row: 行首/行尾标记
```

```plaintext
ROW_START(row_slot=0, block_count=当前总块数)
→ META → COL_INDEX → VALUE×n
→ META → COL_INDEX → VALUE×m
→ ...
ROW_END(row_slot=1, block_count=新的总块数)
```

---

**User**

存到 Total Buffer 不存Meta吗？

---

**Cursor**

好问题！这确实需要澄清。让我说清楚 **Block CSR 的核心存储内容**。

---

## 标准 Block CSR 格式的核心三件套

真正的 Block CSR 格式只需要存储三个数组：

1. **`row_ptr[]`**：行指针数组
2. **`col_idx[]`**：列索引数组（每个非零块一个）
3. **`values[]`**：数值数组（所有块的数据拼接）

**`META` 包含的信息（块尺寸、factor_id）在标准 Block CSR 中通常不存储！**

---

## 为什么会有 META？两种方案对比

### 方案 A：不存 META（标准紧凑 Block CSR）

**Total Buffer 只存**：
```
row_ptr[] = [0, 3, 5, 10, ...]
col_idx[] = [y1, x1, x2, y2, x3, ...]
values[]  = [R块的36个数, x1块的36个数, x2块的36个数, ...]
```

**前提条件**：
- 所有块尺寸相同（如全部 6×6）**或者**能通过变量类型推导出块尺寸
- 不需要追溯"哪个块来自哪个 factor"

**Encoder 输出流**：
- ROW_START → COL_INDEX → VALUE×n → COL_INDEX → VALUE×m → ROW_END
- **不需要 META 包**

---

### 方案 B：存 META（扩展 Block CSR，支持变块尺寸）

**Total Buffer 存**：
```
row_ptr[] = [0, 3, 5, ...]
col_idx[] = [y1, x1, x2, y2, x3, ...]
block_info[] = [(rows=6,cols=6,factor=f1), (6,6,f1), (6,6,f1), (2,6,f3), (6,6,f3), ...]
values[]  = [块数据...]
```

**适用场景**：
- 块尺寸可变（6×6、2×6、2×3 混合）
- 需要追溯每个块来自哪个 factor（调试/统计/后处理）

**Encoder 输出流**：
- ROW_START → **META** → COL_INDEX → VALUE×n → **META** → COL_INDEX → VALUE×m → ROW_END

---

## 你应该选哪个？

```plaintext
row_ptr[] = [0, 3, 5, 10, ...]
col_idx[] = [y1, x1, x2, y2, x3, ...]
values[]  = [R块的36个数, x1块的36个数, x2块的36个数, ...]
```

```plaintext
row_ptr[] = [0, 3, 5, ...]
col_idx[] = [y1, x1, x2, y2, x3, ...]
block_info[] = [(rows=6,cols=6,factor=f1), (6,6,f1), (6,6,f1), (2,6,f3), (6,6,f3), ...]
values[]  = [块数据...]
```

**推荐方案 B**，因为：
1. 你的块尺寸是**可变的**（6×6、2×6、2×3）
2. 你需要知道"哪个块来自哪个 factor"（factor_id/row_base）

---

## 简化后的数据格式（最终版）

### Encoder 输入
```systemverilog
- block_rows, block_cols
- factor_id            // 来自哪个因子
- block_col_idx        // 对应哪个变量列
- block_data[][]
- new_block_row        // 行首块标记
- last_block_in_row    // 行尾块标记
```

### Encoder 输出到 Total Buffer
```
ROW_START(block_count_before)
  ↓
对每个块：
  META(rows, cols, factor_id)
  COL_INDEX(block_col_idx)
  VALUE × (rows×cols)
  ↓
ROW_END(block_count_after)
```

### Total Buffer 存储的内容
```
row_ptr[]     ← 从 ROW_START/ROW_END 构建
col_idx[]     ← 从 COL_INDEX 提取
block_info[]  ← 从 META 提取（rows, cols, factor_id）
values[]      ← 从 VALUE 提取
```

---

## 总结

**要不要存 META？** → **要存**，因为你的块尺寸可变 + 需要追溯来源。

**存哪些内容？** → 4 个数组：`row_ptr[]`, `col_idx[]`, `block_info[]`, `values[]`

**能不能更简化？** → 可以！如果你能保证所有块都是 6×6，且不需要 factor_id，就可以删掉 META，只存核心三件套。

现在的 encoder.sv 已经支持这个流程了，你觉得是否需要调整？

```systemverilog
- block_rows, block_cols
- factor_id            // 来自哪个因子
- block_col_idx        // 对应哪个变量列
- block_data[][]
- new_block_row        // 行首块标记
- last_block_in_row    // 行尾块标记
```

```plaintext
ROW_START(block_count_before)
  ↓
对每个块：
  META(rows, cols, factor_id)
  COL_INDEX(block_col_idx)
  VALUE × (rows×cols)
  ↓
ROW_END(block_count_after)
```

```systemverilog
- block_rows, block_cols
- factor_id            // 来自哪个因子
- block_col_idx        // 对应哪个变量列
- block_data[][]
- new_block_row        // 行首块标记
- last_block_in_row    // 行尾块标记
```

```plaintext
ROW_START(block_count_before)
  ↓
对每个块：
  META(rows, cols, factor_id)
  COL_INDEX(block_col_idx)
  VALUE × (rows×cols)
  ↓
ROW_END(block_count_after)
```

```plaintext
row_ptr[]     ← 从 ROW_START/ROW_END 构建
col_idx[]     ← 从 COL_INDEX 提取
block_info[]  ← 从 META 提取（rows, cols, factor_id）
values[]      ← 从 VALUE 提取
```

---

**User**

B，为什么需要记录因子在info里面？这三组数据都意味着什么？第一组是输入，第三组是输出，中间的是中间数据？

---

**Cursor**

完全正确！你的理解非常清晰：

## 三组数据的含义

### 第一组（左）：Encoder 的**输入**
- Global Controller **逐块**提交给 Encoder 的并行数据
- 一次提交描述一个矩阵块的完整信息
- 通过 `block_valid/ready` 握手

### 第二组（中）：Encoder 的**输出**（中间流数据）
- Encoder 将输入的并行块数据**串行化**
- 输出成带类型标记的 `csr_word_t` 流
- 通过 `csr_valid/ready` 握手传给 Total Buffer
- **这是临时的传输格式，不是最终存储格式**

### 第三组（右）：Total Buffer 的**存储**（最终数据）
- Total Buffer 收到流后，**解析并重组**成标准 Block CSR 格式
- 分成 4 个独立数组存储
- 这是**最终持久化**的数据，供后续回代使用

---

## 为什么 `block_info[]` 里要记录 `factor_id`？

### 两个关键原因

#### 1. **支持变块尺寸**
- 不同因子可能产生不同尺寸的块（6×6、2×6、2×3）
- Decoder 读取时需要知道每个块的尺寸，才能正确从 `values[]` 中切分数据
- 例如：`values[0:35]` 是 6×6=36 个数，`values[36:47]` 是 2×6=12 个数

#### 2. **追溯和调试**（可选但推荐）
- 回代完成后，你得到解向量 `x`
- 你可能想检查某个测量（如某个 IMU factor）的残差
- 通过 `factor_id` 可以定位"第 i 个块行对应的是哪个原始测量"

---

## 能不能不存 factor_id？

**可以，但有条件：**

### 如果所有块都是固定尺寸（如全部 6×6）
```
Total Buffer 只存：
- row_ptr[]
- col_idx[]  
- values[]
```

**Decoder 解码时**：
- 看到 row_ptr[i+1] - row_ptr[i] = 3，知道第 i 行有 3 个块
- 每个块固定 6×6，所以从 values 中每次取 36 个数即可
- **不需要 block_info[]**

### 如果块尺寸可变（你的情况）
```
Total Buffer 必须存：
- row_ptr[]
- col_idx[]
- block_info[] = [(rows, cols), (rows, cols), ...]  ← 至少要记录尺寸
- values[]
```

**可以不存 factor_id，只存 (rows, cols)**：
```
block_info[] = [(6,6), (6,6), (6,6), (2,6), (6,6), ...]
```

---

## 推荐做法（平衡方案）

```systemverilog
// 精简版 block_info（只存必需的）
struct {
    logic [2:0] rows;
    logic [2:0] cols;
    // factor_id 可选：调试时有用，纯计算不需要
    // logic [IDX_WIDTH-1:0] factor_id;  
} block_info_t;
```

**Total Buffer 存储**：
```
row_ptr[]     : 行指针
col_idx[]     : 每个块的列索引
block_info[]  : 每个块的尺寸 (rows, cols)
values[]      : 所有块的浮点数据拼接
```

**Decoder 用法**：
1. 读 row_ptr[i] ~ row_ptr[i+1] → 知道第 i 行有 k 个块
2. 对这 k 个块，依次：
   - 读 col_idx[j] → 知道第 j 个块在哪一列
   - 读 block_info[j] → 知道该块是 (rows, cols)
   - 读 values[offset : offset+rows×cols-1] → 取出该块的数据

---

## 总结

✅ **要存 META（至少存 rows, cols）**，因为你的块尺寸可变

❓ **factor_id 可选**：
- 存 → 便于追溯调试
- 不存 → 节省存储，纯计算够用

**建议**：先保留 factor_id，后期优化时如果存储紧张再删。

你图里的方案是对的！三组数据：输入（并行块） → 中间流（串行传输） → 输出（解析存储）。

```plaintext
Total Buffer 只存：
- row_ptr[]
- col_idx[]  
- values[]
```

```plaintext
Total Buffer 必须存：
- row_ptr[]
- col_idx[]
- block_info[] = [(rows, cols), (rows, cols), ...]  ← 至少要记录尺寸
- values[]
```

```plaintext
block_info[] = [(6,6), (6,6), (6,6), (2,6), (6,6), ...]
```

```systemverilog
// 精简版 block_info（只存必需的）
struct {
    logic [2:0] rows;
    logic [2:0] cols;
    // factor_id 可选：调试时有用，纯计算不需要
    // logic [IDX_WIDTH-1:0] factor_id;  
} block_info_t;
```

```plaintext
row_ptr[]     : 行指针
col_idx[]     : 每个块的列索引
block_info[]  : 每个块的尺寸 (rows, cols)
values[]      : 所有块的浮点数据拼接
```

---

**User**

为什么需要中间的变量那一组？

---

**Cursor**

非常好的问题！这触及了硬件设计的核心。让我解释为什么不能直接从左边跳到右边。

---

## 为什么需要中间流？核心原因是**硬件接口限制**

### 问题：如果没有中间流会怎样？

假设 Encoder 直接把并行输入写到 Total Buffer：

```systemverilog
// 不现实的"理想接口"
module encoder (
    input [31:0] block_data[6][6],  // 36个32位数 = 1152位！
    output logic [31:0] values[],   // 可变长数组？硬件做不到！
    output logic [IDX_WIDTH-1:0] col_idx[],
    ...
);
```

**硬件无法实现的原因**：

1. **端口位宽爆炸**
   - 一个 6×6 块 = 36×32 = **1152 位**并行数据
   - 如果一行有 3 个块 = **3456 位**！
   - 这样的总线在硬件里不现实（布线、时序、功耗都无法接受）

2. **存储器接口限制**
   - RAM/BRAM 通常只有**固定宽度**的读写端口（如 32/64/128 位）
   - 不可能一次写入可变长度的数组
   - 必须**分多次、逐个写入**

3. **时序约束**
   - 即使强行做超宽端口，组合逻辑延迟会非常大
   - 无法满足时序要求（setup/hold time）

---

## 中间流的作用：**硬件可实现的串行化接口**

### 它解决了什么问题？

```
并行的、结构复杂的输入
         ↓
    【Encoder】
         ↓
串行的、固定宽度的流 ← 这个可以高效地写入存储器
         ↓
   【Total Buffer】
         ↓
解析后存入多个数组
```

### 具体来说

**Encoder 的工作**：
- 接收 1152 位的并行块数据（一次性采样到内部寄存器）
- 用状态机**逐拍、逐个**把数据发出去
- 每拍只发一个 `csr_word_t`（固定宽度，如 64~128 位）

**Total Buffer 的工作**：
- 每拍接收一个固定宽度的 `csr_word_t`
- 根据 `kind` 字段判断该写到哪个数组
- 用简单的**计数器控制写地址**，逐个写入

---

## 类比：快递打包

### 没有中间流（不可行）
- 你想把一整个房间的家具（block_data）**瞬间传送**到仓库
- 需要一个"和房间一样大的传送门" ← 硬件做不到

### 有中间流（可行）

```systemverilog
// 不现实的"理想接口"
module encoder (
    input [31:0] block_data[6][6],  // 36个32位数 = 1152位！
    output logic [31:0] values[],   // 可变长数组？硬件做不到！
    output logic [IDX_WIDTH-1:0] col_idx[],
    ...
);
```

```plaintext
并行的、结构复杂的输入
         ↓
    【Encoder】
         ↓
串行的、固定宽度的流 ← 这个可以高效地写入存储器
         ↓
   【Total Buffer】
         ↓
解析后存入多个数组
```

- 你把家具（block_data）**打包成标准箱子**（csr_word_t）
- 通过**标准门**逐个搬运（每拍一个包）
- 到仓库后**拆箱归类**存放

---

## 三个核心原因

### 1. **硬件端口宽度限制**
- Encoder 输入：`block_data[6][6]` = 36×32 = **1152 位**并行
- Total Buffer 如果直接接收 1152 位 → 布线/时序灾难
- **必须串行化**成固定宽度（如 64 位的 `csr_word_t`），每拍传一个

### 2. **存储器写入必须逐个进行**
- RAM/BRAM 每次只能写**一个地址的数据**
- 你要写 36 个数到 `values[]`，就得写 36 次
- Encoder → Total Buffer 的流就是这"36 次写"的数据源

### 3. **模块解耦与流控**
- 如果 Total Buffer 正忙（处理其他数据），`csr_ready=0`
- Encoder 会**等待**，不会丢数据
- 这种 `valid/ready` 握手是硬件模块间通信的标准方式

---

## 为什么要记录 factor_id？

### 必须记录的：`rows, cols`
- 因为块尺寸可变（6×6、2×6、2×3）
- Decoder 读 `values[]` 时必须知道"下一个块有多少个数"

### 可选记录的：`factor_id`
**不存的后果**：
- 你只知道"第 5 个块是 6×6、在 y1 列"
- 但不知道"它来自哪个测量/因子"
- 调试时无法追溯

**存了的好处**：
- 回代完成后，你可以说"factor f1 的残差是多少"
- 可以分析"哪些测量贡献了误差"
- 方便统计、可视化

**我的建议**：先保留 `factor_id`，如果后期存储紧张再删。

---

## 最终答案

**为什么需要中间流？**
→ **硬件必需**。不可能把 1152 位并行数据一次性写入存储器，必须串行化成固定宽度的包，逐拍传输。

**为什么要存 block_info (含 factor_id)？**
→ `rows, cols` **必须存**（块尺寸可变）；`factor_id` **推荐存**（便于追溯调试）。

你的图完全正确：**输入（并行）→ 中间流（串行传输）→ 输出（解析存储）**，这是硬件实现的标准三段式。

---

**User**

还是不太理解为什么要中间数据

---

**Cursor**

你纠结的“中间数据”不是多存一份数据，而是“传输协议”。它的存在只为把“很宽、结构复杂的输入”变成“固定宽度、可逐拍写入存储器的字流”。不落地、不占额外存储，被接收后立刻写进最终的 3/4 个数组。

### 必要性（一句话版）
- 输入是并行块（最多上千位宽）→ 存储器只支持固定小宽度逐拍写入 → 必须先“串行化 + 带类型标记”，否则无法稳定写 RAM。

### 它具体解决的4件事
- 串行化宽数据：例如 6×6×32=1152 bit 的 `block_data`，被拆成若干“包”（META/COL_INDEX/VALUE），每拍一个固定宽度字。
- 指示写入目的地：通过“包类型”告诉 Total Buffer 该写 `row_ptr/col_idx/block_info/values` 的哪个地址。
- 流控与解耦：`valid/ready` 让生产/消费两侧不同速率也不会丢数据或卡死。
- 多源易仲裁：多路 Encoder 时，按包仲裁合并很容易；否则要做超宽总线/多端口跨写，代价高且时序难收敛.

### 没有中间流能不能做？
- 可以把 Encoder 和 Total Buffer 合成一个模块，内部自己做“串行化 + 分发 + 写内存”的状态机。实质上你仍然实现了同样的中间“包”，只是接口藏在模块内部。拆成中间流只是把职责清晰化、接口可复用、易调试。

### 你可以把它当成
- “日志流”：META/INDEX/VALUE 是写入日志；Total Buffer 读一条，写一条，立刻消费，不缓存。
- “快递打包”：先装箱（统一规格的包），到库房再按箱上的标签入库。

简短结论：
- 中间那组不是“中间变量存储”，而是“按包定义的传输协议”，为硬件可实现性（位宽、时序、存储器写入、流控、仲裁）服务；最终仍只落到 `row_ptr/col_idx/(block_info)/values`。

---

**User**

所以中间那个是把输入打包？为什么还要多一步打包？

---

**Cursor**

对！中间那组就是“打包”，而“多一步打包”是**硬件实现所必需的**，原因很简单：

**输入端（并行）和存储端（串行）的“数据宽度”和“访问方式”完全不匹配。**

---

### 一个简单的比喻：把大象装进冰箱

1.  **输入（左边）**：一只完整的大象（`block_data` + 所有元数据）。
    -   这是**并行**的，所有信息在同一时刻都准备好了。
    -   数据宽度非常大（比如 1000+ 位）。

2.  **存储（右边）**：一个标准冰箱（`Total Buffer` 里的 RAM）。
    -   冰箱门很小（RAM 的写端口，比如只有 64 位宽）。
    -   你一次只能往里放一小块东西（一次写一个 64 位的数据）。

**问题**：你怎么把整只大象（1000+ 位）塞进只有一个小门（64 位）的冰箱？

**答案**：你不能直接塞。你必须先把大象**切成标准大小的肉块**，然后一块一块地放进去。

---

### “打包”就是“切块”和“贴标签”的过程

-   **中间数据流（中间那组）**：就是那些被切好的、标准大小的肉块，每一块都贴着标签。
    -   **`ROW_START`, `INFO`, `COL_INDEX`, `VALUE`** 就是这些“标签”。
    -   **`csr_word_t`** 就是“标准大小的盒子”，每个盒子装一块带标签的肉。

-   **Encoder 的工作**：就是那个负责“切块”和“贴标签”的屠夫。它接收整只大象，然后有条不紊地输出一盒一盒的标准肉块。

-   **Total Buffer 的工作**：就是那个在冰箱前整理的人。他拿到一盒肉，看到标签上写着“腿肉”（`COL_INDEX`），就把它放进冷冻室的“腿肉区”（`col_idx[]` 数组）；看到标签是“里脊”（`VALUE`），就放进“里脊区”（`values[]` 数组）。

---

### 总结：为什么必须多一步打包？

1.  **解决硬件端口宽度不匹配的问题**：
    -   **输入端**：超宽总线（1000+ 位），并行。
    -   **存储端**：标准窄总线（如 64 位），串行。
    -   **打包**：就是把“宽并行”数据转换成“窄串行”流，这是硬件能处理的唯一方式。

2.  **解决存储器写入方式不匹配的问题**：
    -   **输入端**：数据是结构化的（`row_base`, `col_idx`, `block_data` 等）。
    -   **存储端**：数据需要被分门别类存到不同的数组（`row_ptr[]`, `col_idx[]`, `values[]`）。
    -   **打包（贴标签）**：`kind` 字段告诉 `Total Buffer` 当前这个数据包应该写入哪个数组的哪个位置。没有这个标签，`Total Buffer` 就是一团乱麻。

所以，中间这一步**不是多余的存储，而是一个必需的、用于物理传输的“协议转换”**。它将一个逻辑上完整的、并行的“块”对象，转换成了一系列硬件上可行的、串行的写操作。

---

**User**

我觉得这里打包的作用是因为，我的QR的R和update不一定同步算完，打包是为了等都算完，这样理解会不会好一点？

---

**Cursor**

这个理解抓到了一个非常关键的点，但我们需要把它放在正确的位置上。你说对了一半！

确实需要一个“等待”的步骤，但**“等待”和“打包”是两个独立的环节**，由不同的模块完成。

---

### 1. **“等待”环节：发生在 Encoder 之前**

-   **谁负责等待？** → **Global Controller**
-   **等待什么？** → 等待 `R.sv` 算完 `R` 块，**并且**所有相关的 `update.sv` 模块都算完各自的方阵块。
-   **为什么需要等待？** → 因为 `g1` 这一整行的数据必须是**完整**的，你才能开始处理它。

这个过程就像**“集齐一套”**。`Global Controller` 必须等到 `g1` 行的所有块（`y1` 的R块、`x1` 的方阵、`x2` 的方阵）都计算完毕并准备好，才能进行下一步。

### 2. **“打包”环节：发生在 Encoder 内部**

-   **谁负责打包？** → **Encoder**
-   **打包什么？** → 打包**已经集齐了的**、单个的矩阵块。
-   **为什么需要打包？** → 为了解决**硬件接口**问题（把上千位的并行数据，变成几十位的串行流）。

这个过程是**“序列化”**。在 `Global Controller` 确认 `g1` 的所有块都 ready 之后，它会**逐个**把这些块喂给 `Encoder`。`Encoder` 每拿到一个块，就把它“打包”成 `csr_word_t` 流发出去。

---

### 用你的图来解释这两个步骤

![你的图](https://cursor.blob.core.windows.net/user-uploads/2025-11-07/image-1730957971701.png)

1.  **等待/集齐 (图中未画出)**:
    -   `Global Controller` 触发 `R.sv` 和 `update.sv` 开始计算。
    -   `Global Controller` 等待 `R.sv` 和所有 `update.sv` 的 `done` 信号。
    -   **此时，`g1` 行的所有块在逻辑上都已准备就绪。**

2.  **Encoder 输入 (图左侧)**:
    -   `Global Controller` **从集齐的数据中**取出**第一个块**（`y1` 的 R 块）及其元数据。
    -   将这个块的所有信息（1000+ 位）**并行**地提供给 `Encoder`。

3.  **打包/串行化 (图中部)**:
    -   `Encoder` 接收到这一个块。
    -   在接下来的几十个周期里，`Encoder` 将这个块的信息**逐个**打包成 `csr_word_t` 包（`ROW_START`, `INFO`, `COL_INDEX`, `VALUE`...）发送出去。

4.  **循环**:
    -   `Encoder` 完成第一个块的打包后，通过 `block_ready` 信号告诉 `Global Controller`：“我空了”。
    -   `Global Controller` 再取出**第二个块**（`x1` 的方阵），重复第 2、3 步。
    -   ...直到 `g1` 行的所有块都被 `Encoder` 处理完毕。

---

### 总结

你的理解**“打包是为了等都算完”**，其实是指出了**需要一个同步点**。这个同步点是存在的，但它在 `Encoder` 的上游，由 `Global Controller` 负责。

-   **等待/集齐**：是**逻辑上**的需要，确保数据完整性。
-   **打包/串行化**：是**物理上**的需要，确保数据能通过硬件接口传输和存储。

所以，更准确的说法是：
**`Global Controller` 先“等待”所有块算完并“集齐”它们，然后“逐个”交给 `Encoder` 去“打包”成串行流。**

---

**User**

硬件接口到底什么问题！

---

**Cursor**

好的，我们彻底说清楚硬件接口的问题，不打比方，只讲硬件实现。

---

### 问题核心：`Encoder` 和 `Total Buffer` 之间的物理连接

`Total Buffer` 的核心是存储器（RAM/BRAM）。在FPGA或ASIC中，一块存储器具有**固定**的物理特性：

1.  **固定的数据宽度（Data Width）**：比如 32-bit、64-bit 或 128-bit。这意味着你**一次只能写入**这么多位的数据。
2.  **固定的地址线（Address Width）**：决定了存储器的深度。
3.  **一个写使能信号（Write Enable）**：拉高时，在时钟上升沿将数据写入指定地址。

现在，我们来看 `Encoder` 的输入。

### `Encoder` 的输入有多宽？

假设一个块是 `6x6`，数据类型是 32-bit 浮点数。
-   `block_data`: 6 * 6 * 32-bit = **1152 位**
-   `block_rows`, `block_cols`: 3 + 3 = **6 位**
-   `row_base`: 16-bit
-   `block_col_idx`: 16-bit
-   `new_block_row`, `last_block_in_row`: 1 + 1 = **2 位**
-   ... 还有其他元数据

加起来，`Encoder` 在一个时钟周期内需要接收的**并行数据总宽度轻松超过 1200 位**。

---

### 如果没有中间流，直接连接会怎么样？

**设想一：`Total Buffer` 的 RAM 也有 1200+ 位的写端口**

-   **物理上不可能**：
    -   FPGA/ASIC 中的标准存储器模块（BRAM/SRAM）根本没有这么宽的端口。标准块的端口宽度通常是 32/64/72 位。
    -   即使你自己用触发器去拼一个超宽的存储器，从 `Encoder` 到 `Total Buffer` 需要**超过 1200 根物理连线**来传输数据。这在芯片布局布线（Place & Route）阶段会是**一场灾难**，导致大面积的布线拥塞和无法满足的时序。

**设想二：`Total Buffer` 内部自己处理 1200+ 位的输入**

-   `Total Buffer` 接收 1200+ 位的数据。
-   它内部需要一个巨大的多路选择器（Mux）网络，根据 `kind` 字段（如果我们在 1200 位里也打包一个 `kind` 的话）来决定这一大坨数据里的哪一部分（比如 `col_idx` 部分）应该被写入 `col_idx_ram`，哪一部分（比如 `values`）应该被写入 `values_ram`。
-   更糟糕的是，`values` 部分（1152位）本身也需要被拆分，因为 `values_ram` 的端口也只有 32 位。所以 `Total Buffer` 内部需要一个 1152-to-32 的巨大 Mux，以及一个计数器，在 36 个周期内依次把这 36 个浮点数写进去。

**这实际上就是把 `Encoder` 的打包逻辑搬到了 `Total Buffer` 内部，问题并没有消失，只是转移了。** 而且这样做违反了模块化设计的原则，让 `Total Buffer` 变得异常复杂和臃肿。

---

### 中间流的解决方案：**将时空进行转换**

中间流（`csr_word_t`）的本质，是**用时间换空间**。

-   **空间**：我们不使用 1200+ 根物理连线。我们只用几十根线（例如 64 或 128 根，取决于 `csr_word_t` 的宽度）。
-   **时间**：我们不用一个时钟周期传完所有信息。我们用几十个时钟周期，每周期传一部分。

**`Encoder` 做的事情**：
-   在一个周期内，锁存（Latch）所有并行的输入数据到内部寄存器。
-   然后启动一个状态机，在接下来的几十个周期里，**逐个**把 `row_ptr` 信息、`col_idx` 信息、以及 36 个 `value` 数据，打包成固定宽度的 `csr_word_t`，通过那几十根线发送出去。

**`Total Buffer` 做的事情**：
-   接口非常简单：只有一个几十位宽的数据输入端口。
-   每周期接收一个 `csr_word_t`。
-   根据 `kind` 字段，简单地决定把这个包写入哪个 RAM 的当前地址指针位置。

---

### 总结：硬件接口的刚性约束

1.  **物理连线成本**：上千根并行数据线在物理上不可行。
2.  **存储器端口限制**：RAM/BRAM 只有固定且较窄的读写端口。
3.  **时序收敛**：超宽的组合逻辑路径会导致时序违例（timing violation）。

**中间的打包步骤，就是为了解决这三个核心的物理约束。** 它是一个标准的、几乎在所有高性能硬件设计中都会用到的技术：**将并行数据域和串行（或窄并行）存储/传输域进行转换的接口协议。**

---

**User**

所以这里是为了扁平化，可以这样理解吧

---

**Cursor**

对！**“扁平化” (Flattening) 这个词用得非常精准**。

你可以从两个维度来理解这个“扁平化”：

---

### 1. **结构上的扁平化**

-   **之前 (输入端)**：数据是**结构化、有层次**的。
    -   它不是一堆无差别的比特流，而是被清晰地分组成 `row_base`, `col_idx`, `block_data` 等多个字段。
    -   `block_data` 本身还是一个二维数组 `[row][col]`。

-   **之后 (中间流)**：数据被**扁平化成一维的、无差别的数据包流**。
    -   `Encoder` 就像一个序列化工具，把所有这些结构（二维数组、多个元数据字段）都拆解开，转换成一长串 `csr_word_t`。
    -   每个 `csr_word_t` 包的结构都是一样的，只是 `kind` 字段不同，表明了它承载的是原始结构中的哪一部分。

这就好比你把一个复杂的 JSON 对象：
```json
{
  "row_base": 7,
  "col_idx": 10,
  "data": [[1.0, 2.0], [3.0, 4.0]]
}
```
扁平化成一个事件流：
```
EVENT: META, row_base=7
EVENT: COL_INDEX, col_idx=10
EVENT: VALUE, value=1.0
EVENT: VALUE, value=2.0
EVENT: VALUE, value=3.0
EVENT: VALUE, value=4.0
```
下游模块只需要按顺序处理这个简单的事件流，而不需要理解复杂的嵌套结构。

---

### 2. **物理上的扁平化**

-   **之前 (输入端)**：数据在**空间维度**上是巨大的。
    -   需要上千根并行的物理连线来同时承载所有信息。

-   **之后 (中间流)**：数据在**时间维度**上被展开了。
    -   物理连线数量大大减少（例如 64 根）。
    -   作为代价，传输时间从 1 个周期延长到了几十个周期。

这是一种典型的**时空权衡 (Space-Time Tradeoff)**。我们用更多的时间（串行化传输），换取了更少的物理空间（更少的连线），从而使设计在物理上成为可能。

---

### 总结

是的，**“扁平化”** 是一个非常贴切的描述。中间这一步的核心作用就是：

-   **在逻辑上**，将复杂的、结构化的数据对象，**扁平化**成一个简单的、自描述的事件/数据包流。
-   **在物理上**，将空间上超宽的并行总线，**扁平化**成时间上更长的、但物理上可行的窄总线传输。

```json
{
  "row_base": 7,
  "col_idx": 10,
  "data": [[1.0, 2.0], [3.0, 4.0]]
}
```

```plaintext
EVENT: META, row_base=7
EVENT: COL_INDEX, col_idx=10
EVENT: VALUE, value=1.0
EVENT: VALUE, value=2.0
EVENT: VALUE, value=3.0
EVENT: VALUE, value=4.0
```

---

**User**

按照我的需求，修改@encoder.sv @decoder.sv 

---

**Cursor**

```systemverilog
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/RTL/encoder.sv
@@
-module encoder #(
+module encoder #(
     parameter int MAX_ROWS       = 6,
     parameter int MAX_COLS       = 6,
     parameter int DATA_WIDTH     = 32,
     parameter int IDX_WIDTH      = 16,   // Width for global column indices
-    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters / block-row id
+    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
 ) (
     input  logic clk,
     input  logic rst,
 
     // Block input handshake --------------------------------------------------
@@
-    input  logic [IDX_WIDTH-1:0] row_base,            // Top-left row index in global matrix
-    input  logic [ROWPTR_WIDTH-1:0] row_block_id,     // Sequential block-row identifier
-    input  logic [ROWPTR_WIDTH-1:0] next_row_block_id,// Identifier for row_ptr entry after this row
+    input  logic [IDX_WIDTH-1:0]   block_factor_id,   // Identifier for the originating factor
+    input  logic [IDX_WIDTH-1:0]   block_col_idx,     // Block-column index
     input  logic                 new_block_row,       // 1 when this block is the first in its block-row
     input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
     input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream
 
-    // Column indices for each column inside the block (length MAX_COLS, unused entries ignored)
-    input  logic [IDX_WIDTH-1:0] col_indices [MAX_COLS-1:0],
-
     // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
     input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],
@@
-        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / row base (depends on kind)
-        logic [IDX_WIDTH-1:0]       idx_field;  // Column index or block-row id (depends on kind)
+        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
+        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
         logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
     } csr_word_t;
@@
-    logic [IDX_WIDTH-1:0] row_base_q;
-    logic [ROWPTR_WIDTH-1:0] row_block_id_q;
-    logic [ROWPTR_WIDTH-1:0] next_row_block_id_q;
+    logic [IDX_WIDTH-1:0] block_factor_id_q;
+    logic [IDX_WIDTH-1:0] block_col_idx_q;
     logic                 new_row_q;
     logic                 last_in_row_q;
     logic                 last_overall_q;
 
-    logic [IDX_WIDTH-1:0] col_indices_q [MAX_COLS-1:0];
     logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];
 
-    // Counters for streaming columns/values
-    logic [2:0] col_idx_q, col_idx_d;
+    // Counters for streaming values
     logic [2:0] val_row_q, val_row_d;
     logic [2:0] val_col_q, val_col_d;
 
     // Running number of blocks already emitted (nnzb in block units)
     logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;
 
+    // Row index tracker for emitted row_ptr entries
+    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;
+
     // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
     logic final_end_pending_q, final_end_pending_d;
@@
-            row_base_q         <= '0;
-            row_block_id_q     <= '0;
-            next_row_block_id_q<= '0;
+            block_factor_id_q  <= '0;
+            block_col_idx_q    <= '0;
             new_row_q          <= 1'b0;
             last_in_row_q      <= 1'b0;
             last_overall_q     <= 1'b0;
             for (r = 0; r < MAX_ROWS; r++) begin
                 for (c = 0; c < MAX_COLS; c++) begin
                     block_data_q[r][c] <= '0;
                 end
             end
-            for (c = 0; c < MAX_COLS; c++) begin
-                col_indices_q[c] <= '0;
-            end
-            col_idx_q           <= '0;
             val_row_q           <= '0;
             val_col_q           <= '0;
             nnzb_count_q        <= '0;
+            row_index_q         <= '0;
             final_end_pending_q <= 1'b0;
         end else begin
             state_q <= state_d;
 
             if (accept_block) begin
                 rows_q             <= block_rows;
                 cols_q             <= block_cols;
-                row_base_q         <= row_base;
-                row_block_id_q     <= row_block_id;
-                next_row_block_id_q<= next_row_block_id;
+                block_factor_id_q  <= block_factor_id;
+                block_col_idx_q    <= block_col_idx;
                 new_row_q          <= new_block_row;
                 last_in_row_q      <= last_block_in_row;
                 last_overall_q     <= last_block_overall;
-
-                for (c = 0; c < MAX_COLS; c++) begin
-                    col_indices_q[c] <= col_indices[c];
-                end
                 for (r = 0; r < MAX_ROWS; r++) begin
                     for (c = 0; c < MAX_COLS; c++) begin
                         block_data_q[r][c] <= block_data[r][c];
                     end
                 end
 
-                col_idx_q <= '0;
                 val_row_q <= '0;
                 val_col_q <= '0;
             end else begin
-                col_idx_q <= col_idx_d;
                 val_row_q <= val_row_d;
                 val_col_q <= val_col_d;
             end
 
             nnzb_count_q        <= nnzb_count_d;
+            row_index_q         <= row_index_d;
             final_end_pending_q <= final_end_pending_d;
         end
     end
@@
-        col_idx_d           = col_idx_q;
         val_row_d           = val_row_q;
         val_col_d           = val_col_q;
 
         nnzb_count_d        = nnzb_count_q;
+        row_index_d         = row_index_q;
         final_end_pending_d = final_end_pending_q;
 
         unique case (state_q)
@@
             S_SEND_ROW_PTR_START: begin
                 csr_word_d.kind      = CSR_KIND_ROW_PTR;
                 csr_word_d.row_field = nnzb_count_q;
-                csr_word_d.idx_field = row_block_id_q;
+                csr_word_d.idx_field = row_index_q;
                 csr_valid_d          = 1'b1;
                 if (csr_ready) begin
                     state_d = S_SEND_META;
                 end
             end
 
             // Emit block metadata (header)
             S_SEND_META: begin
                 csr_word_d.kind      = CSR_KIND_META;
                 csr_word_d.rows      = rows_q;
                 csr_word_d.cols      = cols_q;
-                csr_word_d.row_field = row_base_q;
-                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
+                csr_word_d.row_field = nnzb_count_q; // Prefix count snapshot (optional)
+                csr_word_d.idx_field = block_factor_id_q;
                 csr_valid_d          = 1'b1;
                 if (csr_ready) begin
-                    if (cols_q != 0) begin
-                        state_d   = S_SEND_COLS;
-                        col_idx_d = '0;
-                    end else begin
-                        // No columns -> skip directly to value phase (will emit nothing)
-                        state_d = S_SEND_VALUES;
-                    end
+                    state_d = S_SEND_COLS;
                 end
             end
 
             // Emit column indices for this block
             S_SEND_COLS: begin
-                if (col_idx_q < cols_q) {
-                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
-                    csr_word_d.idx_field = col_indices_q[col_idx_q];
-                    csr_valid_d          = 1'b1;
-                    if (csr_ready) begin
-                        col_idx_d = col_idx_q + 1'b1;
-                        if (col_idx_q + 1'b1 == cols_q) begin
-                            state_d = S_SEND_VALUES;
-                            val_row_d = '0;
-                            val_col_d = '0;
-                        end
-                    end
-                end else begin
-                    state_d = S_SEND_VALUES;
-                end
+                csr_word_d.kind      = CSR_KIND_COL_INDEX;
+                csr_word_d.idx_field = block_col_idx_q;
+                csr_valid_d          = 1'b1;
+                if (csr_ready) begin
+                    state_d   = S_SEND_VALUES;
+                    val_row_d = '0;
+                    val_col_d = '0;
+                end
             end
 
             // Emit dense values (row-major order)
@@
                             nnzb_count_d = nnzb_count_q + 1'b1;
                             if (last_in_row_q || last_overall_q) begin
                                 state_d = S_SEND_ROW_PTR_END;
                             end else begin
                                 state_d = S_IDLE;
@@
             S_SEND_ROW_PTR_END: begin
                 csr_word_d.kind      = CSR_KIND_ROW_PTR;
                 csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
-                csr_word_d.idx_field = next_row_block_id_q;
+                csr_word_d.idx_field = row_index_q + 1'b1;
                 csr_valid_d          = 1'b1;
                 if (csr_ready) begin
+                    row_index_d = row_index_q + 1'b1;
                     if (last_overall_q) begin
                         final_end_pending_d = 1'b1;
                         state_d = S_SEND_STREAM_END;
                     end else begin
                         state_d = S_IDLE;
@@
             csr_word_d.kind      = CSR_KIND_STREAM_END;
             csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
             csr_valid_d          = 1'b1;
             csr_last_d           = 1'b1;
             if (csr_ready) begin
+                row_index_d         = row_index_q; // hold
                 final_end_pending_d = 1'b0;
                 state_d = S_IDLE;
             end
         end
 
*** End Patch

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / row base (depends on kind)
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index or block-row id (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] row_base_q;
    logic [ROWPTR_WIDTH-1:0] row_block_id_q;
    logic [ROWPTR_WIDTH-1:0] next_row_block_id_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [IDX_WIDTH-1:0] col_indices_q [MAX_COLS-1:0];
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming columns/values
    logic [2:0] col_idx_q, col_idx_d;
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            row_base_q         <= '0;
            row_block_id_q     <= '0;
            next_row_block_id_q<= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            col_idx_q           <= '0;
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                row_base_q         <= row_base;
                row_block_id_q     <= row_block_id;
                next_row_block_id_q<= next_row_block_id;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (c = 0; c < MAX_COLS; c++) begin
                    col_indices_q[c] <= col_indices[c];
                end
                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                col_idx_q <= '0;
                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                col_idx_q <= col_idx_d;
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        col_idx_d           = col_idx_q;
        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = row_base_q;
                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (cols_q != 0) begin
                        state_d   = S_SEND_COLS;
                        col_idx_d = '0;
                    end else begin
                        // No columns -> skip directly to value phase (will emit nothing)
                        state_d = S_SEND_VALUES;
                    end
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                if (col_idx_q < cols_q) begin
                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
                    csr_word_d.idx_field = col_indices_q[col_idx_q];
                    csr_valid_d          = 1'b1;
                    if (csr_ready) begin
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d = S_SEND_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end else begin
                    state_d = S_SEND_VALUES;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = next_row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] row_base_q;
    logic [ROWPTR_WIDTH-1:0] row_block_id_q;
    logic [ROWPTR_WIDTH-1:0] next_row_block_id_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [IDX_WIDTH-1:0] col_indices_q [MAX_COLS-1:0];
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming columns/values
    logic [2:0] col_idx_q, col_idx_d;
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            row_base_q         <= '0;
            row_block_id_q     <= '0;
            next_row_block_id_q<= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            col_idx_q           <= '0;
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                row_base_q         <= row_base;
                row_block_id_q     <= row_block_id;
                next_row_block_id_q<= next_row_block_id;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (c = 0; c < MAX_COLS; c++) begin
                    col_indices_q[c] <= col_indices[c];
                end
                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                col_idx_q <= '0;
                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                col_idx_q <= col_idx_d;
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        col_idx_d           = col_idx_q;
        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = row_base_q;
                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (cols_q != 0) begin
                        state_d   = S_SEND_COLS;
                        col_idx_d = '0;
                    end else begin
                        // No columns -> skip directly to value phase (will emit nothing)
                        state_d = S_SEND_VALUES;
                    end
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                if (col_idx_q < cols_q) begin
                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
                    csr_word_d.idx_field = col_indices_q[col_idx_q];
                    csr_valid_d          = 1'b1;
                    if (csr_ready) begin
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d = S_SEND_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end else begin
                    state_d = S_SEND_VALUES;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = next_row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            row_base_q         <= '0;
            row_block_id_q     <= '0;
            next_row_block_id_q<= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            col_idx_q           <= '0;
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                row_base_q         <= row_base;
                row_block_id_q     <= row_block_id;
                next_row_block_id_q<= next_row_block_id;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (c = 0; c < MAX_COLS; c++) begin
                    col_indices_q[c] <= col_indices[c];
                end
                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                col_idx_q <= '0;
                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                col_idx_q <= col_idx_d;
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        col_idx_d           = col_idx_q;
        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = row_base_q;
                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (cols_q != 0) begin
                        state_d   = S_SEND_COLS;
                        col_idx_d = '0;
                    end else begin
                        // No columns -> skip directly to value phase (will emit nothing)
                        state_d = S_SEND_VALUES;
                    end
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                if (col_idx_q < cols_q) begin
                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
                    csr_word_d.idx_field = col_indices_q[col_idx_q];
                    csr_valid_d          = 1'b1;
                    if (csr_ready) begin
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d = S_SEND_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end else begin
                    state_d = S_SEND_VALUES;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = next_row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        col_idx_d           = col_idx_q;
        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = row_base_q;
                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (cols_q != 0) begin
                        state_d   = S_SEND_COLS;
                        col_idx_d = '0;
                    end else begin
                        // No columns -> skip directly to value phase (will emit nothing)
                        state_d = S_SEND_VALUES;
                    end
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                if (col_idx_q < cols_q) begin
                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
                    csr_word_d.idx_field = col_indices_q[col_idx_q];
                    csr_valid_d          = 1'b1;
                    if (csr_ready) begin
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d = S_SEND_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end else begin
                    state_d = S_SEND_VALUES;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = next_row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        row_index_d         = row_index_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = row_base_q;
                csr_word_d.idx_field = nnzb_count_q; // snapshot for reference
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (cols_q != 0) begin
                        state_d   = S_SEND_COLS;
                        col_idx_d = '0;
                    end else begin
                        // No columns -> skip directly to value phase (will emit nothing)
                        state_d = S_SEND_VALUES;
                    end
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                if (col_idx_q < cols_q) begin
                    csr_word_d.kind      = CSR_KIND_COL_INDEX;
                    csr_word_d.idx_field = col_indices_q[col_idx_q];
                    csr_valid_d          = 1'b1;
                    if (csr_ready) begin
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d = S_SEND_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end else begin
                    state_d = S_SEND_VALUES;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = next_row_block_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row_block_id, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        row_index_d         = row_index_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_index_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = nnzb_count_q; // snapshot for reference (optional)
                csr_word_d.idx_field = block_factor_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_COLS;
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                csr_word_d.kind      = CSR_KIND_COL_INDEX;
                csr_word_d.idx_field = block_col_idx_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d   = S_SEND_VALUES;
                    val_row_d = '0;
                    val_col_d = '0;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = row_index_q + 1'b1;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    row_index_d = row_index_q + 1'b1;
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row index, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // Tracks whether we still owe a STREAM_END word after emitting final row_ptr
    logic final_end_pending_q, final_end_pending_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
            final_end_pending_q <= 1'b0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
            final_end_pending_q <= final_end_pending_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        row_index_d         = row_index_q;
        final_end_pending_d = final_end_pending_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_index_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = nnzb_count_q; // snapshot for reference (optional)
                csr_word_d.idx_field = block_factor_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_COLS;
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                csr_word_d.kind      = CSR_KIND_COL_INDEX;
                csr_word_d.idx_field = block_col_idx_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d   = S_SEND_VALUES;
                    val_row_d = '0;
                    val_col_d = '0;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = row_index_q + 1'b1;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    row_index_d = row_index_q + 1'b1;
                    if (last_overall_q) begin
                        final_end_pending_d = 1'b1;
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    final_end_pending_d = 1'b0;
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row index, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 new_row_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            new_row_q          <= 1'b0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                new_row_q          <= new_block_row;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        row_index_d         = row_index_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_index_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = nnzb_count_q; // snapshot for reference (optional)
                csr_word_d.idx_field = block_factor_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_COLS;
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                csr_word_d.kind      = CSR_KIND_COL_INDEX;
                csr_word_d.idx_field = block_col_idx_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d   = S_SEND_VALUES;
                    val_row_d = '0;
                    val_col_d = '0;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = row_index_q + 1'b1;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    row_index_d = row_index_q + 1'b1;
                    if (last_overall_q) begin
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [IDX_WIDTH-1:0] row_base_q, row_base_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;

    logic [IDX_WIDTH-1:0] col_indices_q [MAX_COLS-1:0];
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] col_idx_q, col_idx_d;
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        row_base_d      = row_base_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        col_idx_d       = col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d        = csr_word.rows;
                                cols_d        = csr_word.cols;
                                row_base_d    = csr_word.row_field;
                                nnzb_prefix_d = csr_word.idx_field;
                                col_idx_d     = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES; // values phase may still have 0 words
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        col_indices_q[col_idx_q] = csr_word.idx_field;
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d   = S_RECV_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        row_base_d      = row_base_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        col_idx_d       = col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d        = csr_word.rows;
                                cols_d        = csr_word.cols;
                                row_base_d    = csr_word.row_field;
                                nnzb_prefix_d = csr_word.idx_field;
                                col_idx_d     = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES; // values phase may still have 0 words
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        col_indices_q[col_idx_q] = csr_word.idx_field;
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d   = S_RECV_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d        = csr_word.rows;
                                cols_d        = csr_word.cols;
                                row_base_d    = csr_word.row_field;
                                nnzb_prefix_d = csr_word.idx_field;
                                col_idx_d     = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES; // values phase may still have 0 words
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        col_indices_q[col_idx_q] = csr_word.idx_field;
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d   = S_RECV_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                state_d       = S_RECV_COLS;
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        col_indices_q[col_idx_q] = csr_word.idx_field;
                        col_idx_d = col_idx_q + 1'b1;
                        if (col_idx_q + 1'b1 == cols_q) begin
                            state_d   = S_RECV_VALUES;
                            val_row_d = '0;
                            val_col_d = '0;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                state_d       = S_RECV_COLS;
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_col_idx_d = csr_word.idx_field;
                        state_d   = S_RECV_VALUES;
                        val_row_d = '0;
                        val_col_d = '0;
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES;
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_col_idx_d = csr_word.idx_field;
                        state_d   = S_RECV_VALUES;
                        val_row_d = '0;
                        val_col_d = '0;
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            row_base_q      <= '0;
            nnzb_prefix_q   <= '0;
            col_idx_q       <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (c = 0; c < MAX_COLS; c++) begin
                col_indices_q[c] <= '0;
            end
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            row_base_q      <= row_base_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            col_idx_q       <= col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES;
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_col_idx_d = csr_word.idx_field;
                        state_d   = S_RECV_VALUES;
                        val_row_d = '0;
                        val_col_d = '0;
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            nnzb_prefix_q   <= '0;
            block_factor_id_q <= '0;
            block_col_idx_q   <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            block_factor_id_q <= block_factor_id_d;
            block_col_idx_q   <= block_col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign row_base          = row_base_q;
    assign nnzb_prefix       = nnzb_prefix_q;

    // Provide column indices & dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_COLS; gi++) begin : GEN_COL_OUT
            assign col_indices[gi] = col_indices_q[gi];
        end
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Accept conditions -------------------------------------------------
    logic accept_word;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        accept_word     = 1'b0;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                accept_word     = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && state_q == S_WAIT_WORD && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES;
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                accept_word   = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_col_idx_d = csr_word.idx_field;
                        state_d   = S_RECV_VALUES;
                        val_row_d = '0;
                        val_col_d = '0;
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            nnzb_prefix_q   <= '0;
            block_factor_id_q <= '0;
            block_col_idx_q   <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            block_factor_id_q <= block_factor_id_d;
            block_col_idx_q   <= block_col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign nnzb_prefix       = nnzb_prefix_q;
    assign block_factor_id   = block_factor_id_q;
    assign block_col_idx     = block_col_idx_q;

    // Provide dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

```systemverilog
// -----------------------------------------------------------------------------
// encoder.sv
// -----------------------------------------------------------------------------
// Convert a dense QR/update block into Block-CSR stream words that can be stored
// inside the Total Buffer. Each block is accompanied by metadata describing its
// global position and shape. Optionally, row_ptr entries are emitted whenever a
// new block-row begins or ends so that the consumer can rebuild the global
// Block-CSR row pointer array.
// -----------------------------------------------------------------------------

module encoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,   // Width for global column indices / factor ids
    parameter int ROWPTR_WIDTH   = 16    // Width for row_ptr counters and row indices
) (
    input  logic clk,
    input  logic rst,

    // Block input handshake --------------------------------------------------
    input  logic                 block_valid,
    output logic                 block_ready,

    // Block metadata
    input  logic [2:0]           block_rows,          // <= MAX_ROWS
    input  logic [2:0]           block_cols,          // <= MAX_COLS
    input  logic [IDX_WIDTH-1:0] block_factor_id,     // Identifier for the originating factor
    input  logic [IDX_WIDTH-1:0] block_col_idx,       // Block-column index
    input  logic                 new_block_row,       // 1 when this block is the first in its block-row
    input  logic                 last_block_in_row,   // 1 when this block is the final block in its block-row
    input  logic                 last_block_overall,  // 1 when this is the final block of the entire stream

    // Dense block payload (row-major). Unused rows/cols must be set to 0 by caller.
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Output stream ----------------------------------------------------------
    output logic                 csr_valid,
    input  logic                 csr_ready,
    output logic                 csr_last,
    output csr_word_t            csr_word
);

    // ---------------------------------------------------------------------
    // Stream word definition
    // ---------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;       // CSR_KIND_* (see localparams)
        logic [2:0]                 rows;       // Block rows (only valid in meta word)
        logic [2:0]                 cols;       // Block cols (only valid in meta word)
        logic [ROWPTR_WIDTH-1:0]    row_field;  // Row pointer value / nnzb prefix
        logic [IDX_WIDTH-1:0]       idx_field;  // Column index / factor id / row index (depends on kind)
        logic [DATA_WIDTH-1:0]      value_field;// Dense value (only valid for VALUE words)
    } csr_word_t;

    localparam csr_word_t CSR_WORD_RESET = '{default:'0};

    localparam logic [7:0] CSR_KIND_META        = 8'h01; // Header describing the block
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02; // Column index entry (idx_field)
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03; // Dense value (value_field)
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04; // Row pointer entry: idx_field=row index, row_field=nnzb prefix
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF; // Marks end of entire stream

    // ---------------------------------------------------------------------
    // Internal storage for the accepted block
    // ---------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_SEND_ROW_PTR_START,
        S_SEND_META,
        S_SEND_COLS,
        S_SEND_VALUES,
        S_SEND_ROW_PTR_END,
        S_SEND_STREAM_END
    } state_e;

    state_e state_q, state_d;

    logic [2:0]           rows_q;
    logic [2:0]           cols_q;
    logic [IDX_WIDTH-1:0] block_factor_id_q;
    logic [IDX_WIDTH-1:0] block_col_idx_q;
    logic                 last_in_row_q;
    logic                 last_overall_q;

    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    // Counters for streaming values
    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Running number of blocks already emitted (nnzb in block units)
    logic [ROWPTR_WIDTH-1:0] nnzb_count_q, nnzb_count_d;

    // Row index tracker for row_ptr output (assumes sequential block-rows)
    logic [ROWPTR_WIDTH-1:0] row_index_q, row_index_d;

    // ------------------------------------------------------------------
    // Accept block when idle
    // ------------------------------------------------------------------
    assign block_ready = (state_q == S_IDLE);

    logic accept_block = block_ready && block_valid;

    // ------------------------------------------------------------------
    // Sequential storage of metadata and block payload
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q            <= S_IDLE;
            rows_q             <= '0;
            cols_q             <= '0;
            block_factor_id_q  <= '0;
            block_col_idx_q    <= '0;
            last_in_row_q      <= 1'b0;
            last_overall_q     <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
            val_row_q           <= '0;
            val_col_q           <= '0;
            nnzb_count_q        <= '0;
            row_index_q         <= '0;
        end else begin
            state_q <= state_d;

            if (accept_block) begin
                rows_q             <= block_rows;
                cols_q             <= block_cols;
                block_factor_id_q  <= block_factor_id;
                block_col_idx_q    <= block_col_idx;
                last_in_row_q      <= last_block_in_row;
                last_overall_q     <= last_block_overall;

                for (r = 0; r < MAX_ROWS; r++) begin
                    for (c = 0; c < MAX_COLS; c++) begin
                        block_data_q[r][c] <= block_data[r][c];
                    end
                end

                val_row_q <= '0;
                val_col_q <= '0;
            end else begin
                val_row_q <= val_row_d;
                val_col_q <= val_col_d;
            end

            nnzb_count_q        <= nnzb_count_d;
            row_index_q         <= row_index_d;
        end
    end

    // ------------------------------------------------------------------
    // Combinational FSM & output generator
    // ------------------------------------------------------------------
    csr_word_t csr_word_d;
    logic      csr_valid_d;
    logic      csr_last_d;

    // default assignments
    always_comb begin
        state_d             = state_q;
        csr_word_d          = CSR_WORD_RESET;
        csr_valid_d         = 1'b0;
        csr_last_d          = 1'b0;

        val_row_d           = val_row_q;
        val_col_d           = val_col_q;

        nnzb_count_d        = nnzb_count_q;
        row_index_d         = row_index_q;

        unique case (state_q)
            // ------------------------------------------------------------------
            S_IDLE: begin
                if (accept_block) begin
                    if (new_block_row) begin
                        state_d = S_SEND_ROW_PTR_START;
                    end else begin
                        state_d = S_SEND_META;
                    end
                end
            end

            // Emit row_ptr entry for the beginning of a block-row
            S_SEND_ROW_PTR_START: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;
                csr_word_d.idx_field = row_index_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_META;
                end
            end

            // Emit block metadata (header)
            S_SEND_META: begin
                csr_word_d.kind      = CSR_KIND_META;
                csr_word_d.rows      = rows_q;
                csr_word_d.cols      = cols_q;
                csr_word_d.row_field = nnzb_count_q; // snapshot for reference (optional)
                csr_word_d.idx_field = block_factor_id_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d = S_SEND_COLS;
                end
            end

            // Emit column indices for this block
            S_SEND_COLS: begin
                csr_word_d.kind      = CSR_KIND_COL_INDEX;
                csr_word_d.idx_field = block_col_idx_q;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    state_d   = S_SEND_VALUES;
                    val_row_d = '0;
                    val_col_d = '0;
                end
            end

            // Emit dense values (row-major order)
            S_SEND_VALUES: begin
                if ((rows_q == 0) || (cols_q == 0)) begin
                    // Nothing to emit, treat as empty block
                    nnzb_count_d = nnzb_count_q + 1'b1;
                    if (last_in_row_q || last_overall_q) begin
                        state_d = S_SEND_ROW_PTR_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end else begin
                    csr_word_d.kind       = CSR_KIND_VALUE;
                    csr_word_d.value_field= block_data_q[val_row_q][val_col_q];
                    csr_valid_d           = 1'b1;

                    if (csr_ready) begin
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            // Last value emitted -> advance
                            nnzb_count_d = nnzb_count_q + 1'b1;
                            if (last_in_row_q || last_overall_q) begin
                                state_d = S_SEND_ROW_PTR_END;
                            end else begin
                                state_d = S_IDLE;
                            end
                        end
                    end
                end
            end

            // Emit row_ptr entry for the end of the block-row
            S_SEND_ROW_PTR_END: begin
                csr_word_d.kind      = CSR_KIND_ROW_PTR;
                csr_word_d.row_field = nnzb_count_q;      // number of blocks emitted so far
                csr_word_d.idx_field = row_index_q + 1'b1;
                csr_valid_d          = 1'b1;
                if (csr_ready) begin
                    row_index_d = row_index_q + 1'b1;
                    if (last_overall_q) begin
                        state_d = S_SEND_STREAM_END;
                    end else begin
                        state_d = S_IDLE;
                    end
                end
            end

            // Emit terminal word (stream end)
            S_SEND_STREAM_END: begin
                csr_word_d.kind      = CSR_KIND_STREAM_END;
                csr_word_d.row_field = nnzb_count_q; // final nnzb count snapshot
                csr_valid_d          = 1'b1;
                csr_last_d           = 1'b1;
                if (csr_ready) begin
                    state_d = S_IDLE;
                end
            end

            default: begin
                state_d = S_IDLE;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Output registers
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            csr_word  <= CSR_WORD_RESET;
        end else begin
            // Valid/last are driven by combinational logic; respect csr_ready back-pressure
            if (csr_ready || !csr_valid) begin
                csr_valid <= csr_valid_d;
                csr_last  <= csr_last_d;
                csr_word  <= csr_word_d;
            end else begin
                // Hold previous word until consumer accepts
                csr_valid <= csr_valid;
                csr_last  <= csr_last;
                csr_word  <= csr_word;
            end
        end
    end

endmodule

```

---

**User**

我发现有多余的代码，能优化的地方帮我优化

---

**Cursor**

```systemverilog
// -----------------------------------------------------------------------------
// decoder.sv
// -----------------------------------------------------------------------------
// Reverse operation of encoder.sv. Consumes the Block-CSR stream words and
// reconstructs dense blocks alongside row_ptr entries. The module exposes two
// independent handshakes:
//   * row_ptr_valid/ready  : forwards row pointer entries as they appear.
//   * block_valid/ready    : delivers block metadata and dense payload once the
//                            corresponding META/COL/VALUE words have been fully
//                            collected.
// -----------------------------------------------------------------------------

module decoder #(
    parameter int MAX_ROWS       = 6,
    parameter int MAX_COLS       = 6,
    parameter int DATA_WIDTH     = 32,
    parameter int IDX_WIDTH      = 16,
    parameter int ROWPTR_WIDTH   = 16
) (
    input  logic clk,
    input  logic rst,

    // CSR input stream ------------------------------------------------------
    input  logic         csr_valid,
    output logic         csr_ready,
    input  logic         csr_last,
    input  csr_word_t    csr_word,

    // Row pointer output ----------------------------------------------------
    output logic                     row_ptr_valid,
    input  logic                     row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id,

    // Dense block output ----------------------------------------------------
    output logic                     block_valid,
    input  logic                     block_ready,
    output logic [2:0]               block_rows,
    output logic [2:0]               block_cols,
    output logic [ROWPTR_WIDTH-1:0]  nnzb_prefix,
    output logic [IDX_WIDTH-1:0]     block_factor_id,
    output logic [IDX_WIDTH-1:0]     block_col_idx,
    output logic [DATA_WIDTH-1:0]    block_data [MAX_ROWS-1:0][MAX_COLS-1:0],

    // Stream termination pulse ---------------------------------------------
    output logic                     stream_done
);

    // ------------------------------------------------------------------
    // Stream word definition (must match encoder.sv)
    // ------------------------------------------------------------------
    typedef struct packed {
        logic [7:0]                 kind;
        logic [2:0]                 rows;
        logic [2:0]                 cols;
        logic [ROWPTR_WIDTH-1:0]    row_field;
        logic [IDX_WIDTH-1:0]       idx_field;
        logic [DATA_WIDTH-1:0]      value_field;
    } csr_word_t;

    localparam logic [7:0] CSR_KIND_META        = 8'h01;
    localparam logic [7:0] CSR_KIND_COL_INDEX   = 8'h02;
    localparam logic [7:0] CSR_KIND_VALUE       = 8'h03;
    localparam logic [7:0] CSR_KIND_ROW_PTR     = 8'h04;
    localparam logic [7:0] CSR_KIND_STREAM_END  = 8'hFF;

    typedef enum logic [1:0] {
        S_WAIT_WORD,
        S_RECV_COLS,
        S_RECV_VALUES,
        S_BLOCK_READY
    } state_e;

    state_e state_q, state_d;

    logic [2:0] rows_q, rows_d;
    logic [2:0] cols_q, cols_d;
    logic [ROWPTR_WIDTH-1:0] nnzb_prefix_q, nnzb_prefix_d;
    logic [IDX_WIDTH-1:0] block_factor_id_q, block_factor_id_d;
    logic [IDX_WIDTH-1:0] block_col_idx_q, block_col_idx_d;
    logic [DATA_WIDTH-1:0] block_data_q [MAX_ROWS-1:0][MAX_COLS-1:0];

    logic [2:0] val_row_q, val_row_d;
    logic [2:0] val_col_q, val_col_d;

    // Row pointer FIFO (single entry) -----------------------------------
    logic row_ptr_valid_q, row_ptr_valid_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_value_q, row_ptr_value_d;
    logic [ROWPTR_WIDTH-1:0] row_ptr_row_id_q, row_ptr_row_id_d;

    // Stream done pulse register ----------------------------------------
    logic stream_done_q, stream_done_d;

    // Default assignments
    always_comb begin
        csr_ready       = 1'b0;
        state_d         = state_q;
        rows_d          = rows_q;
        cols_d          = cols_q;
        nnzb_prefix_d   = nnzb_prefix_q;
        block_factor_id_d = block_factor_id_q;
        block_col_idx_d = block_col_idx_q;
        val_row_d       = val_row_q;
        val_col_d       = val_col_q;
        row_ptr_valid_d = row_ptr_valid_q;
        row_ptr_value_d = row_ptr_value_q;
        row_ptr_row_id_d= row_ptr_row_id_q;
        stream_done_d   = 1'b0; // pulse-only

        unique case (state_q)
            // ------------------------------------------------------------------
            S_WAIT_WORD: begin
                if (csr_valid) begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            if (!row_ptr_valid_q) begin
                                csr_ready       = 1'b1;
                                row_ptr_valid_d = 1'b1;
                                row_ptr_value_d = csr_word.row_field;
                                row_ptr_row_id_d= csr_word.idx_field;
                            end
                        end

                        CSR_KIND_META: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                rows_d           = csr_word.rows;
                                cols_d           = csr_word.cols;
                                nnzb_prefix_d    = csr_word.row_field;
                                block_factor_id_d= csr_word.idx_field;
                                block_col_idx_d  = '0;
                                val_row_d     = '0;
                                val_col_d     = '0;
                                if (csr_word.cols == 0) begin
                                    state_d = S_RECV_VALUES;
                                end else begin
                                    state_d = S_RECV_COLS;
                                end
                            end
                        end

                        CSR_KIND_STREAM_END: begin
                            if (!row_ptr_valid_q && !block_valid) begin
                                csr_ready     = 1'b1;
                                stream_done_d = 1'b1;
                            end
                        end

                        default: begin
                            // Unexpected word - hold until downstream frees up
                        end
                    endcase
                end
            end

            // ------------------------------------------------------------------
            S_RECV_COLS: begin
                if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_col_idx_d = csr_word.idx_field;
                        state_d   = S_RECV_VALUES;
                        val_row_d = '0;
                        val_col_d = '0;
                    end
                end
            end

            // ------------------------------------------------------------------
            S_RECV_VALUES: begin
                if (rows_q == 0 || cols_q == 0) begin
                    state_d = S_BLOCK_READY;
                end else if (csr_valid) begin
                    csr_ready = 1'b1;
                    if (csr_ready) begin
                        block_data_q[val_row_q][val_col_q] = csr_word.value_field;
                        if (val_col_q + 1'b1 == cols_q) begin
                            val_col_d = '0;
                            val_row_d = val_row_q + 1'b1;
                        end else begin
                            val_col_d = val_col_q + 1'b1;
                        end

                        if ((val_col_q + 1'b1 == cols_q) && (val_row_q + 1'b1 == rows_q)) begin
                            state_d = S_BLOCK_READY;
                        end
                    end
                end
            end

            // ------------------------------------------------------------------
            S_BLOCK_READY: begin
                if (block_ready) begin
                    state_d = S_WAIT_WORD;
                end
            end
        endcase

        // Maintain row_ptr backpressure when output not yet consumed
        if (row_ptr_valid_q && !row_ptr_ready) begin
            csr_ready = 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // Sequential logic
    // ------------------------------------------------------------------
    integer r, c;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q         <= S_WAIT_WORD;
            rows_q          <= '0;
            cols_q          <= '0;
            nnzb_prefix_q   <= '0;
            block_factor_id_q <= '0;
            block_col_idx_q   <= '0;
            val_row_q       <= '0;
            val_col_q       <= '0;
            row_ptr_valid_q <= 1'b0;
            row_ptr_value_q <= '0;
            row_ptr_row_id_q<= '0;
            stream_done_q   <= 1'b0;
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data_q[r][c] <= '0;
                end
            end
        end else begin
            state_q         <= state_d;
            rows_q          <= rows_d;
            cols_q          <= cols_d;
            nnzb_prefix_q   <= nnzb_prefix_d;
            block_factor_id_q <= block_factor_id_d;
            block_col_idx_q   <= block_col_idx_d;
            val_row_q       <= val_row_d;
            val_col_q       <= val_col_d;

            // Row pointer queue handling
            if (row_ptr_valid_q && row_ptr_ready) begin
                row_ptr_valid_q <= 1'b0;
            end else begin
                row_ptr_valid_q <= row_ptr_valid_d;
            end
            if (row_ptr_valid_d && (!row_ptr_valid_q || (row_ptr_valid_q && row_ptr_ready))) begin
                row_ptr_value_q  <= row_ptr_value_d;
                row_ptr_row_id_q <= row_ptr_row_id_d;
            end

            // Stream done pulse
            stream_done_q <= stream_done_d;
        end
    end

    // Block output registers -------------------------------------------------
    logic block_valid_q;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            block_valid_q <= 1'b0;
        end else begin
            if (state_q == S_BLOCK_READY) begin
                block_valid_q <= 1'b1;
            end else if (block_ready) begin
                block_valid_q <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Output assignments
    // ------------------------------------------------------------------
    assign row_ptr_valid     = row_ptr_valid_q;
    assign row_ptr_value     = row_ptr_value_q;
    assign row_ptr_row_id    = row_ptr_row_id_q;

    assign block_valid       = block_valid_q;
    assign block_rows        = rows_q;
    assign block_cols        = cols_q;
    assign nnzb_prefix       = nnzb_prefix_q;
    assign block_factor_id   = block_factor_id_q;
    assign block_col_idx     = block_col_idx_q;

    // Provide dense data
    generate
        genvar gi, gj;
        for (gi = 0; gi < MAX_ROWS; gi++) begin : GEN_ROW_OUT
            for (gj = 0; gj < MAX_COLS; gj++) begin : GEN_COL_IN_ROW_OUT
                assign block_data[gi][gj] = block_data_q[gi][gj];
            end
        end
    endgenerate

    assign stream_done = stream_done_q;

endmodule

```

- 优化 `decoder.sv`：删除未被使用的 `accept_word` 变量及其相关赋值逻辑，同时去掉在 `S_WAIT_WORD` 状态中冗余的 `state_q == S_WAIT_WORD` 判断，保持相同功能但减少无效组合电路与布线负担。 ` ```94:145:QR/RTL/decoder.sv
// ... existing code ...
``` `
- 检查 `encoder.sv`：当前实现未发现多余逻辑，保持不变。

已运行 `read_lints` 确认无新的静态检查问题。建议后续若方便，可跑一次仿真或现有测试，确认流控行为仍符合期望。

---

**User**

现在@encoder.sv 和@decoder.sv 和我预期的逻辑一样了吗？是什么逻辑

---

**Cursor**

好的，我们来最终确认一下 `encoder.sv` 和 `decoder.sv` 当前的代码逻辑，以及它是否符合你的预期。

---

### 是的，现在的逻辑已经和你最终确认的方案完全一致。

简单来说，这套代码实现了一个**将单个矩阵块及其元数据，打包成串行 Block CSR 流**的功能。它遵循“顺序追加”模式，非常适合你“一次处理并归档一行 `g`”的单通道工作流程。

---

### `encoder.sv` 的逻辑

**它的作用**：接收一个矩阵块，把它转换成一连串带有类型标记的数据包。

**输入接口的逻辑**：
-   `Global Controller` 通过 `block_valid/ready` **一次提交一个块**。
-   提交时必须附带：
    -   `block_rows`, `block_cols`: 这个块的**精确尺寸**（例如 `2, 3`）。
    -   `block_factor_id`: 这个块源自哪个**因子**（例如 `f1` 的 ID）。
    -   `block_col_idx`: 这个块对应哪个**变量列**（例如 `y1` 的列索引）。
    -   `block_data`: 块的**稠密数值**。
    -   `new_block_row`, `last_block_in_row`: 用来告诉 `Encoder` **何时**应该生成行指针（`ROW_PTR`）包。

**内部状态机逻辑**：
1.  **等待块 (S_IDLE)**: 等待 `Global Controller` 提交新块。
2.  **发送行指针起点 (S_SEND_ROW_PTR_START)**: 如果是行首块 (`new_block_row=1`)，就先发送一个 `ROW_PTR` 包，内容是“到目前为止的块总数”。
3.  **发送元数据 (S_SEND_META)**: 发送一个 `META` 包，包含块的尺寸和 `factor_id`。
4.  **发送列索引 (S_SEND_COLS)**: 发送一个 `COL_INDEX` 包，包含 `block_col_idx`。
5.  **发送数值 (S_SEND_VALUES)**: 连续发送 `rows * cols` 个 `VALUE` 包，内容是块的稠密数据。
6.  **发送行指针终点 (S_SEND_ROW_PTR_END)**: 如果是行尾块 (`last_block_in_row=1`)，就再发送一个 `ROW_PTR` 包，内容是“包含这个块之后的新块总数”。
7.  回到等待状态，准备接收下一个块。

**输出**：一长串串行的 `csr_word_t` 数据包。

---

### `decoder.sv` 的逻辑

**它的作用**：接收 `Encoder` 生成的串行数据包，**反向**还原出原始的矩阵块和它的元数据。

**输入**：`csr_word_t` 数据流。

**内部状态机逻辑**：
1.  **等待数据包 (S_WAIT_WORD)**:
    -   如果收到 `ROW_PTR` 包，就立即通过 `row_ptr_valid/ready` 接口把它转发出去。下游模块（如 `Total Buffer` 的控制器）可以用它来构建 `row_ptr` 数组。
    -   如果收到 `META` 包，就开始准备接收一个新块，记录下尺寸和 `factor_id`，并进入 `S_RECV_COLS` 状态。
2.  **接收列索引 (S_RECV_COLS)**: 等待并接收一个 `COL_INDEX` 包，存下 `block_col_idx`。
3.  **接收数值 (S_RECV_VALUES)**: 进入循环，等待并接收 `rows * cols` 个 `VALUE` 包，将它们重新组装成一个二维的矩阵。
4.  **块准备就绪 (S_BLOCK_READY)**:
    -   当一个块的所有信息（META, COL_INDEX, VALUEs）都收集完毕后，进入此状态。
    -   拉高 `block_valid` 信号，告诉下游模块：“一个完整的块已经还原好了，可以来取了！”
    -   下游模块取走数据后，`Decoder` 返回等待状态。

**输出接口的逻辑**：
-   它有两个独立的输出通道：
    1.  **`row_ptr` 通道**: 实时转发行指针信息。
    2.  **`block` 通道**: **缓冲并聚合**，直到一个完整的块被还原后，才一次性输出。

---

### 总结

这套 `Encoder/Decoder` 的逻辑完美匹配你的**单通道、顺序处理**流程：

-   **解耦了计算和存储**：`Global Controller` 只负责调度和提交数据，不需要关心 `Total Buffer` 的具体写入细节。
-   **解决了硬件接口问题**：通过“打包”（扁平化/串行化），将上千位的并行数据转换成几十位的串行流，使得数据能够通过物理上可行的总线宽度写入标准存储器。
-   **格式清晰**：`kind` 字段使得数据流是自描述的，`Decoder` 可以独立工作，无需与 `Encoder` 有额外的同步信号。
-   **支持你的核心需求**：能够处理可变尺寸的块，并保留了追溯来源的 `factor_id` 和块位置的 `block_col_idx`。

---

**User**

帮我在QR/RTL/py里面写这个encoder和decoder的py格式，我要先做验证

---

**Cursor**

```python
"""Reference Python models for the Block-CSR encoder/decoder.

These models mirror the behaviour of the SystemVerilog modules in
`QR/RTL/encoder.sv` and `QR/RTL/decoder.sv`.  They are intentionally small and
dependency-free so they can be used directly in unit tests or quick scripts to
validate traces produced by simulations.
"""

from __future__ import print_function

import struct


def f32(value):
    """Round a Python float to IEEE-754 single precision and back."""

    return struct.unpack('>f', struct.pack('>f', float(value)))[0]


class CSRKind(object):
    META = 0x01
    COL_INDEX = 0x02
    VALUE = 0x03
    ROW_PTR = 0x04
    STREAM_END = 0xFF


def make_word(kind, rows=0, cols=0, row_field=0, idx_field=0, value_field=0.0):
    """Create a CSR stream word represented as a python dict."""

    return {
        'kind': int(kind),
        'rows': int(rows) & 0x7,
        'cols': int(cols) & 0x7,
        'row_field': int(row_field),
        'idx_field': int(idx_field),
        'value_field': f32(value_field),
    }


class EncoderModel(object):
    """Cycle-accurate model of encoder.sv (without handshake timing)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._nnzb_count = 0
        self._row_index = 0

    def encode_block(
        self,
        block_rows,
        block_cols,
        block_factor_id,
        block_col_idx,
        block_data,
        new_block_row,
        last_block_in_row,
        last_block_overall,
    ):
        """Encode a single dense block into CSR words.

        Args mirror the SystemVerilog inputs.  `block_data` must be a list of
        lists with dimensions >= block_rows x block_cols; unused entries are
        ignored exactly like in the RTL.
        """

        words = []

        if new_block_row:
            words.append(
                make_word(
                    CSRKind.ROW_PTR,
                    row_field=self._nnzb_count,
                    idx_field=self._row_index,
                )
            )

        words.append(
            make_word(
                CSRKind.META,
                rows=block_rows,
                cols=block_cols,
                row_field=self._nnzb_count,
                idx_field=block_factor_id,
            )
        )

        words.append(
            make_word(
                CSRKind.COL_INDEX,
                idx_field=block_col_idx,
            )
        )

        if block_rows and block_cols:
            for rr in range(block_rows):
                for cc in range(block_cols):
                    words.append(
                        make_word(
                            CSRKind.VALUE,
                            value_field=block_data[rr][cc],
                        )
                    )
        # Empty blocks still advance the nnzb counter
        self._nnzb_count += 1

        if last_block_in_row or last_block_overall:
            words.append(
                make_word(
                    CSRKind.ROW_PTR,
                    row_field=self._nnzb_count,
                    idx_field=self._row_index + 1,
                )
            )
            self._row_index += 1

        if last_block_overall:
            words.append(
                make_word(
                    CSRKind.STREAM_END,
                    row_field=self._nnzb_count,
                )
            )

        return words

    def encode_blocks(self, blocks):
        """Encode a sequence of blocks.

        `blocks` should be an iterable of dictionaries with the same keys as
        `encode_block`.  Returns a flat list of CSR words.
        """

        output = []
        for blk in blocks:
            output.extend(self.encode_block(**blk))
        return output


class DecoderModel(object):
    """Cycle-accurate model of decoder.sv (without handshake timing)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.row_ptrs = []  # list of (row_index, nnzb_prefix)
        self.blocks = []  # list of decoded block dicts
        self.stream_done = False

        self._pending_rows = 0
        self._pending_cols = 0
        self._pending_nnzb_prefix = 0
        self._pending_factor_id = 0
        self._pending_col_idx = 0
        self._pending_values = []
        self._value_cursor = 0
        self._expecting_values = False

    def consume_word(self, word):
        kind = word['kind']

        if kind == CSRKind.ROW_PTR:
            self.row_ptrs.append((word['idx_field'], word['row_field']))
            return

        if kind == CSRKind.META:
            self._pending_rows = word['rows']
            self._pending_cols = word['cols']
            self._pending_nnzb_prefix = word['row_field']
            self._pending_factor_id = word['idx_field']
            self._pending_col_idx = None
            self._value_cursor = 0
            total_values = self._pending_rows * self._pending_cols
            self._pending_values = [
                [f32(0.0) for _ in range(self._pending_cols)]
                for _ in range(self._pending_rows)
            ]
            self._expecting_values = total_values != 0
            return

        if kind == CSRKind.COL_INDEX:
            self._pending_col_idx = word['idx_field']
            if not self._expecting_values:
                self._commit_block()
            return

        if kind == CSRKind.VALUE:
            if not self._expecting_values:
                raise ValueError('Unexpected VALUE word without pending block')

            rows = self._pending_rows
            cols = self._pending_cols
            r = self._value_cursor // cols
            c = self._value_cursor % cols
            self._pending_values[r][c] = f32(word['value_field'])
            self._value_cursor += 1

            if self._value_cursor == rows * cols:
                self._commit_block()
            return

        if kind == CSRKind.STREAM_END:
            self.stream_done = True
            return

        raise ValueError('Unknown CSR word kind: %r' % (kind,))

    def consume_words(self, words):
        for word in words:
            self.consume_word(word)

    def _commit_block(self):
        block = {
            'rows': self._pending_rows,
            'cols': self._pending_cols,
            'nnzb_prefix': self._pending_nnzb_prefix,
            'factor_id': self._pending_factor_id,
            'col_idx': self._pending_col_idx,
            'values': [row[:] for row in self._pending_values],
        }
        self.blocks.append(block)
        self._pending_rows = 0
        self._pending_cols = 0
        self._pending_nnzb_prefix = 0
        self._pending_factor_id = 0
        self._pending_col_idx = 0
        self._pending_values = []
        self._value_cursor = 0
        self._expecting_values = False


def demo():
    encoder = EncoderModel()
    decoder = DecoderModel()

    blocks = [
        {
            'block_rows': 2,
            'block_cols': 3,
            'block_factor_id': 7,
            'block_col_idx': 4,
            'block_data': [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            'new_block_row': True,
            'last_block_in_row': False,
            'last_block_overall': False,
        },
        {
            'block_rows': 2,
            'block_cols': 2,
            'block_factor_id': 8,
            'block_col_idx': 9,
            'block_data': [
                [7.0, 8.0],
                [9.0, 10.0],
            ],
            'new_block_row': False,
            'last_block_in_row': True,
            'last_block_overall': True,
        },
    ]

    stream = encoder.encode_blocks(blocks)
    decoder.consume_words(stream)

    print('CSR stream words (%d):' % len(stream))
    for idx, word in enumerate(stream):
        print(idx, word)

    print('\nRow pointers:')
    for row_idx, prefix in decoder.row_ptrs:
        print('  row %d -> %d' % (row_idx, prefix))

    print('\nBlocks:')
    for blk in decoder.blocks:
        print(blk)


if __name__ == '__main__':
    demo()

```

新增了 `QR/RTL/py/csr_codec.py`，里面实现了纯 Python 版的 `EncoderModel` 和 `DecoderModel`，对应 RTL 里的 `encoder.sv` / `decoder.sv`。核心点：

- 复用 `CSRKind` 常量和 `make_word()` 辅助函数，统一生成 CSR 流数据结构。
- `EncoderModel.encode_block()` 精确复现硬件输出顺序，包括 `ROW_PTR`、`META`、`COL_INDEX`、`VALUE`、`STREAM_END` 逻辑，支持空块场景，自动保持 `nnzb_count` 和 `row_index`。
- `DecoderModel.consume_word()` 能把字流还原成行指针和块内容，输出为易于校验的字典结构，并在接收结束时标记 `stream_done`。
- 附带 `demo()` 示例，展示两块数据的编码、解码结果；已运行 `python3 csr_codec.py` 验证通过。

接下来可在测试脚本里直接调用 `EncoderModel` 生成期望流，再与仿真结果比对，或用 `DecoderModel` 对仿真输出做逆解析。

---

**User**

我要怎么使用这个python

---

**Cursor**

你可以按以下方式使用 `QR/RTL/py/csr_codec.py`：

1. **直接跑自带 demo**  
   在终端进入 `QR/RTL/py` 目录后执行：
   ```
   python3 csr_codec.py
   ```
   会生成一个示例的编码流，并用同一个脚本立刻解码，控制台会打印：
   - 序列化后的 CSR 字流（每个 word 的 `kind/row_field/idx_field/value_field`）
   - 解码得到的 `row_ptr` 列表
   - 解码得到的每个块（包含尺寸、所属因子、列索引、数值）

2. **在自己的验证脚本中 import 使用**  
   ```python
   from csr_codec import EncoderModel, DecoderModel, CSRKind

   # Example blocks to encode
   blocks = [
       {
           "block_rows": 6,
           "block_cols": 6,
           "block_factor_id": 5,
           "block_col_idx": 0,
           "block_data": [[...]],  # 6x6 list of floats
           "new_block_row": True,
           "last_block_in_row": False,
           "last_block_overall": False,
       },
       {
           "block_rows": 6,
           "block_cols": 6,
           "block_factor_id": 5,
           "block_col_idx": 1,
           "block_data": [[...]],
           "new_block_row": False,
           "last_block_in_row": True,
           "last_block_overall": True,
       },
   ]

   encoder = EncoderModel()
   stream = encoder.encode_blocks(blocks)

   decoder = DecoderModel()
   decoder.consume_words(stream)

   print(decoder.row_ptrs)   # [(row_idx, nnzb_prefix), ...]
   print(decoder.blocks)     # 每个块的明细
   ```

   - `encode_blocks()` 输入是字典列表，字段名与 RTL 端口一一对应。
   - `DecoderModel.blocks` 会输出一个数组，其中每个元素都是：
     ```python
     {
         "rows": ...,
         "cols": ...,
         "nnzb_prefix": ...,
         "factor_id": ...,
         "col_idx": ...,
         "values": [[...], ...]
     }
     ```
   - 这样可以把仿真得到的 CSR 字流与 Python 期望值直接做对比，或把仿真输出喂给 `DecoderModel` 还原成稠密块后再继续处理。

3. **手动逐 word 驱动**（如果你要复现握手或做波形对比）  
   ```python
   encoder.reset()
   words = encoder.encode_block(...single block args...)
   decoder.reset()
   for word in words:
       decoder.consume_word(word)
   ```

4. **结合仿真数据验证**  
   - 让 RTL 仿真把 `csr_word.kind/row_field/idx_field/value_field` 打印或写文件。
   - 在 Python 里读取这些 word，同样调用 `decoder.consume_words(...)`。
   - 对比 `decoder.row_ptrs`、`decoder.blocks` 与 RTL 输出的其他参考值，验证 RTL 与 Python 模型是否一致。

使用前记得保证 `csr_codec.py` 在 `PYTHONPATH` 或当前目录下。若你需要自定义块数据，只要按照 `EncoderModel.encode_block()` 的参数格式准备好数据，就能生成与硬件一致的 CSR 序列。

```plaintext
   python3 csr_codec.py
```

```python
   from csr_codec import EncoderModel, DecoderModel, CSRKind

   # Example blocks to encode
   blocks = [
       {
           "block_rows": 6,
           "block_cols": 6,
           "block_factor_id": 5,
           "block_col_idx": 0,
           "block_data": [[...]],  # 6x6 list of floats
           "new_block_row": True,
           "last_block_in_row": False,
           "last_block_overall": False,
       },
       {
           "block_rows": 6,
           "block_cols": 6,
           "block_factor_id": 5,
           "block_col_idx": 1,
           "block_data": [[...]],
           "new_block_row": False,
           "last_block_in_row": True,
           "last_block_overall": True,
       },
   ]

   encoder = EncoderModel()
   stream = encoder.encode_blocks(blocks)

   decoder = DecoderModel()
   decoder.consume_words(stream)

   print(decoder.row_ptrs)   # [(row_idx, nnzb_prefix), ...]
   print(decoder.blocks)     # 每个块的明细
```

```python
     {
         "rows": ...,
         "cols": ...,
         "nnzb_prefix": ...,
         "factor_id": ...,
         "col_idx": ...,
         "values": [[...], ...]
     }
```

```python
   encoder.reset()
   words = encoder.encode_block(...single block args...)
   decoder.reset()
   for word in words:
       decoder.consume_word(word)
```

---

**User**

我要python和我的sv的逻辑一模一样，分成两个文件，一个encoder一个decoder

---

**Cursor**

```python
"""Cycle-accurate Python model of `QR/RTL/encoder.sv`.

The goal of this model is to mirror the SystemVerilog implementation one clock
cycle at a time.  The public API exposes a single `step()` method that mimics a
posedge clock, with explicit handshake inputs (`block_valid`, `csr_ready`) and
registered outputs (`block_ready`, `csr_valid`, `csr_last`, `csr_word`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CSRWord:
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


class EncoderModel:
    CSR_KIND_META = 0x01
    CSR_KIND_COL_INDEX = 0x02
    CSR_KIND_VALUE = 0x03
    CSR_KIND_ROW_PTR = 0x04
    CSR_KIND_STREAM_END = 0xFF

    S_IDLE = 0
    S_SEND_ROW_PTR_START = 1
    S_SEND_META = 2
    S_SEND_COLS = 3
    S_SEND_VALUES = 4
    S_SEND_ROW_PTR_END = 5
    S_SEND_STREAM_END = 6

    def __init__(self, max_rows: int = 6, max_cols: int = 6):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = self.S_IDLE
        self.rows_q = 0
        self.cols_q = 0
        self.block_factor_id_q = 0
        self.block_col_idx_q = 0
        self.last_in_row_q = False
        self.last_overall_q = False
        self.block_data_q = [
            [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
        ]
        self.val_row_q = 0
        self.val_col_q = 0
        self.nnzb_count_q = 0
        self.row_index_q = 0

        self.csr_valid = False
        self.csr_last = False
        self.csr_word = CSRWord()

    def step(
        self,
        *,
        block_valid: bool,
        block_rows: int = 0,
        block_cols: int = 0,
        block_factor_id: int = 0,
        block_col_idx: int = 0,
        block_data: Optional[List[List[float]]] = None,
        new_block_row: bool = False,
        last_block_in_row: bool = False,
        last_block_overall: bool = False,
        csr_ready: bool,
    ) -> dict:
        """Advance the model by one cycle.

        Inputs mirror the RTL ports.  The return value contains the registered
        outputs for the *next* cycle (after this clock edge):

            {
                "block_ready": bool,
                "csr_valid": bool,
                "csr_last": bool,
                "csr_word": CSRWord,
            }
        """

        if block_data is None:
            block_data = [[0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)]

        block_ready = self.state == self.S_IDLE
        accept_block = block_ready and block_valid

        state_d = self.state
        rows_d = self.rows_q
        cols_d = self.cols_q
        block_factor_id_d = self.block_factor_id_q
        block_col_idx_d = self.block_col_idx_q
        last_in_row_d = self.last_in_row_q
        last_overall_d = self.last_overall_q
        block_data_d = [row[:] for row in self.block_data_q]
        val_row_d = self.val_row_q
        val_col_d = self.val_col_q
        nnzb_count_d = self.nnzb_count_q
        row_index_d = self.row_index_q

        csr_word_d = CSRWord()
        csr_valid_d = False
        csr_last_d = False

        if accept_block:
            rows_d = block_rows & 0x7
            cols_d = block_cols & 0x7
            block_factor_id_d = block_factor_id
            block_col_idx_d = block_col_idx
            last_in_row_d = last_block_in_row
            last_overall_d = last_block_overall

            block_data_d = [
                [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
            ]
            for r in range(self.max_rows):
                for c in range(self.max_cols):
                    if r < len(block_data) and c < len(block_data[r]):
                        block_data_d[r][c] = float(block_data[r][c])
            val_row_d = 0
            val_col_d = 0

        # ------------------------------------------------------------------
        # Combinational FSM (mirrors SystemVerilog)
        # ------------------------------------------------------------------
        if self.state == self.S_IDLE:
            if accept_block:
                if new_block_row:
                    state_d = self.S_SEND_ROW_PTR_START
                else:
                    state_d = self.S_SEND_META

        elif self.state == self.S_SEND_ROW_PTR_START:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_ROW_PTR,
                row_field=self.nnzb_count_q,
                idx_field=self.row_index_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_META

        elif self.state == self.S_SEND_META:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_META,
                rows=self.rows_q,
                cols=self.cols_q,
                row_field=self.nnzb_count_q,
                idx_field=self.block_factor_id_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_COLS

        elif self.state == self.S_SEND_COLS:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_COL_INDEX,
                idx_field=self.block_col_idx_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_VALUES
                val_row_d = 0
                val_col_d = 0

        elif self.state == self.S_SEND_VALUES:
            if self.rows_q == 0 or self.cols_q == 0:
                nnzb_count_d = self.nnzb_count_q + 1
                if self.last_in_row_q or self.last_overall_q:
                    state_d = self.S_SEND_ROW_PTR_END
                else:
                    state_d = self.S_IDLE
            else:
                csr_word_d = CSRWord(
                    kind=self.CSR_KIND_VALUE,
                    value_field=self.block_data_q[self.val_row_q][self.val_col_q],
                )
                csr_valid_d = True
                if csr_ready:
                    next_col = (self.val_col_q + 1) % max(self.cols_q, 1)
                    next_row = self.val_row_q + 1 if next_col == 0 else self.val_row_q
                    if next_col == 0:
                        val_col_d = 0
                        if next_row < self.rows_q:
                            val_row_d = next_row
                    else:
                        val_col_d = self.val_col_q + 1

                    if (
                        (self.val_col_q + 1 == self.cols_q)
                        and (self.val_row_q + 1 == self.rows_q)
                    ):
                        nnzb_count_d = self.nnzb_count_q + 1
                        if self.last_in_row_q or self.last_overall_q:
                            state_d = self.S_SEND_ROW_PTR_END
                        else:
                            state_d = self.S_IDLE

        elif self.state == self.S_SEND_ROW_PTR_END:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_ROW_PTR,
                row_field=self.nnzb_count_q,
                idx_field=self.row_index_q + 1,
            )
            csr_valid_d = True
            if csr_ready:
                row_index_d = self.row_index_q + 1
                if self.last_overall_q:
                    state_d = self.S_SEND_STREAM_END
                else:
                    state_d = self.S_IDLE

        elif self.state == self.S_SEND_STREAM_END:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_STREAM_END,
                row_field=self.nnzb_count_q,
            )
            csr_valid_d = True
            csr_last_d = True
            if csr_ready:
                state_d = self.S_IDLE

        # ------------------------------------------------------------------
        # Sequential update (posedge clk)
        # ------------------------------------------------------------------
        self.state = state_d
        self.rows_q = rows_d
        self.cols_q = cols_d
        self.block_factor_id_q = block_factor_id_d
        self.block_col_idx_q = block_col_idx_d
        self.last_in_row_q = last_in_row_d
        self.last_overall_q = last_overall_d
        self.block_data_q = block_data_d
        self.val_row_q = val_row_d
        self.val_col_q = val_col_d
        self.nnzb_count_q = nnzb_count_d
        self.row_index_q = row_index_d

        if csr_ready or not self.csr_valid:
            self.csr_valid = csr_valid_d
            self.csr_last = csr_last_d
            self.csr_word = csr_word_d

        return {
            "block_ready": block_ready,
            "csr_valid": self.csr_valid,
            "csr_last": self.csr_last,
            "csr_word": self.csr_word,
        }


def demo():
    enc = EncoderModel()
    enc.reset()

    # Example sequence of blocks.
    blocks = [
        {
            "block_valid": True,
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_valid": True,
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    stream = []
    csr_ready = True
    for blk in blocks:
        # Present block until consumed.
        consumed = False
        while not consumed:
            outputs = enc.step(csr_ready=csr_ready, **blk)
            stream.append(outputs["csr_word"] if outputs["csr_valid"] else None)
            consumed = outputs["block_ready"] and blk["block_valid"]
            blk["block_valid"] = False  # Only valid for one cycle in this demo

        # Drain remaining words for this block.
        draining = True
        while draining:
            outputs = enc.step(
                block_valid=False,
                csr_ready=csr_ready,
            )
            if outputs["csr_valid"]:
                stream.append(outputs["csr_word"])
            draining = enc.state != EncoderModel.S_IDLE or enc.csr_valid

    print("Generated %d CSR words" % len([w for w in stream if w is not None]))


if __name__ == "__main__":
    demo()

```

```python
"""Cycle-accurate Python model of `QR/RTL/encoder.sv`.

The goal of this model is to mirror the SystemVerilog implementation one clock
cycle at a time.  The public API exposes a single `step()` method that mimics a
posedge clock, with explicit handshake inputs (`block_valid`, `csr_ready`) and
registered outputs (`block_ready`, `csr_valid`, `csr_last`, `csr_word`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CSRWord:
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


class EncoderModel:
    CSR_KIND_META = 0x01
    CSR_KIND_COL_INDEX = 0x02
    CSR_KIND_VALUE = 0x03
    CSR_KIND_ROW_PTR = 0x04
    CSR_KIND_STREAM_END = 0xFF

    S_IDLE = 0
    S_SEND_ROW_PTR_START = 1
    S_SEND_META = 2
    S_SEND_COLS = 3
    S_SEND_VALUES = 4
    S_SEND_ROW_PTR_END = 5
    S_SEND_STREAM_END = 6

    def __init__(self, max_rows: int = 6, max_cols: int = 6):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = self.S_IDLE
        self.rows_q = 0
        self.cols_q = 0
        self.block_factor_id_q = 0
        self.block_col_idx_q = 0
        self.last_in_row_q = False
        self.last_overall_q = False
        self.block_data_q = [
            [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
        ]
        self.val_row_q = 0
        self.val_col_q = 0
        self.nnzb_count_q = 0
        self.row_index_q = 0

        self.csr_valid = False
        self.csr_last = False
        self.csr_word = CSRWord()

    def step(
        self,
        *,
        block_valid: bool,
        block_rows: int = 0,
        block_cols: int = 0,
        block_factor_id: int = 0,
        block_col_idx: int = 0,
        block_data: Optional[List[List[float]]] = None,
        new_block_row: bool = False,
        last_block_in_row: bool = False,
        last_block_overall: bool = False,
        csr_ready: bool,
    ) -> dict:
        """Advance the model by one cycle.

        Inputs mirror the RTL ports.  The return value contains the registered
        outputs for the *next* cycle (after this clock edge):

            {
                "block_ready": bool,
                "csr_valid": bool,
                "csr_last": bool,
                "csr_word": CSRWord,
            }
        """

        if block_data is None:
            block_data = [[0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)]

        block_ready = self.state == self.S_IDLE
        accept_block = block_ready and block_valid

        state_d = self.state
        rows_d = self.rows_q
        cols_d = self.cols_q
        block_factor_id_d = self.block_factor_id_q
        block_col_idx_d = self.block_col_idx_q
        last_in_row_d = self.last_in_row_q
        last_overall_d = self.last_overall_q
        block_data_d = [row[:] for row in self.block_data_q]
        val_row_d = self.val_row_q
        val_col_d = self.val_col_q
        nnzb_count_d = self.nnzb_count_q
        row_index_d = self.row_index_q

        csr_word_d = CSRWord()
        csr_valid_d = False
        csr_last_d = False

        if accept_block:
            rows_d = block_rows & 0x7
            cols_d = block_cols & 0x7
            block_factor_id_d = block_factor_id
            block_col_idx_d = block_col_idx
            last_in_row_d = last_block_in_row
            last_overall_d = last_block_overall

            block_data_d = [
                [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
            ]
            for r in range(self.max_rows):
                for c in range(self.max_cols):
                    if r < len(block_data) and c < len(block_data[r]):
                        block_data_d[r][c] = float(block_data[r][c])
            val_row_d = 0
            val_col_d = 0

        # ------------------------------------------------------------------
        # Combinational FSM (mirrors SystemVerilog)
        # ------------------------------------------------------------------
        if self.state == self.S_IDLE:
            if accept_block:
                if new_block_row:
                    state_d = self.S_SEND_ROW_PTR_START
                else:
                    state_d = self.S_SEND_META

        elif self.state == self.S_SEND_ROW_PTR_START:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_ROW_PTR,
                row_field=self.nnzb_count_q,
                idx_field=self.row_index_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_META

        elif self.state == self.S_SEND_META:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_META,
                rows=self.rows_q,
                cols=self.cols_q,
                row_field=self.nnzb_count_q,
                idx_field=self.block_factor_id_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_COLS

        elif self.state == self.S_SEND_COLS:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_COL_INDEX,
                idx_field=self.block_col_idx_q,
            )
            csr_valid_d = True
            if csr_ready:
                state_d = self.S_SEND_VALUES
                val_row_d = 0
                val_col_d = 0

        elif self.state == self.S_SEND_VALUES:
            if self.rows_q == 0 or self.cols_q == 0:
                nnzb_count_d = self.nnzb_count_q + 1
                if self.last_in_row_q or self.last_overall_q:
                    state_d = self.S_SEND_ROW_PTR_END
                else:
                    state_d = self.S_IDLE
            else:
                csr_word_d = CSRWord(
                    kind=self.CSR_KIND_VALUE,
                    value_field=self.block_data_q[self.val_row_q][self.val_col_q],
                )
                csr_valid_d = True
                if csr_ready:
                    if self.val_col_q + 1 == self.cols_q:
                        val_col_d = 0
                        val_row_d = self.val_row_q + 1
                    else:
                        val_col_d = self.val_col_q + 1

                    if (
                        (self.val_col_q + 1 == self.cols_q)
                        and (self.val_row_q + 1 == self.rows_q)
                    ):
                        nnzb_count_d = self.nnzb_count_q + 1
                        if self.last_in_row_q or self.last_overall_q:
                            state_d = self.S_SEND_ROW_PTR_END
                        else:
                            state_d = self.S_IDLE

        elif self.state == self.S_SEND_ROW_PTR_END:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_ROW_PTR,
                row_field=self.nnzb_count_q,
                idx_field=self.row_index_q + 1,
            )
            csr_valid_d = True
            if csr_ready:
                row_index_d = self.row_index_q + 1
                if self.last_overall_q:
                    state_d = self.S_SEND_STREAM_END
                else:
                    state_d = self.S_IDLE

        elif self.state == self.S_SEND_STREAM_END:
            csr_word_d = CSRWord(
                kind=self.CSR_KIND_STREAM_END,
                row_field=self.nnzb_count_q,
            )
            csr_valid_d = True
            csr_last_d = True
            if csr_ready:
                state_d = self.S_IDLE

        # ------------------------------------------------------------------
        # Sequential update (posedge clk)
        # ------------------------------------------------------------------
        self.state = state_d
        self.rows_q = rows_d
        self.cols_q = cols_d
        self.block_factor_id_q = block_factor_id_d
        self.block_col_idx_q = block_col_idx_d
        self.last_in_row_q = last_in_row_d
        self.last_overall_q = last_overall_d
        self.block_data_q = block_data_d
        self.val_row_q = val_row_d
        self.val_col_q = val_col_d
        self.nnzb_count_q = nnzb_count_d
        self.row_index_q = row_index_d

        if csr_ready or not self.csr_valid:
            self.csr_valid = csr_valid_d
            self.csr_last = csr_last_d
            self.csr_word = csr_word_d

        return {
            "block_ready": block_ready,
            "csr_valid": self.csr_valid,
            "csr_last": self.csr_last,
            "csr_word": self.csr_word,
        }


def demo():
    enc = EncoderModel()
    enc.reset()

    # Example sequence of blocks.
    blocks = [
        {
            "block_valid": True,
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_valid": True,
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    stream = []
    csr_ready = True
    for blk in blocks:
        # Present block until consumed.
        consumed = False
        while not consumed:
            outputs = enc.step(csr_ready=csr_ready, **blk)
            stream.append(outputs["csr_word"] if outputs["csr_valid"] else None)
            consumed = outputs["block_ready"] and blk["block_valid"]
            blk["block_valid"] = False  # Only valid for one cycle in this demo

        # Drain remaining words for this block.
        draining = True
        while draining:
            outputs = enc.step(
                block_valid=False,
                csr_ready=csr_ready,
            )
            if outputs["csr_valid"]:
                stream.append(outputs["csr_word"])
            draining = enc.state != EncoderModel.S_IDLE or enc.csr_valid

    print("Generated %d CSR words" % len([w for w in stream if w is not None]))


if __name__ == "__main__":
    demo()


```

```python
"""Cycle-accurate Python model of `QR/RTL/decoder.sv`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


class DecoderModel:
    CSR_KIND_META = 0x01
    CSR_KIND_COL_INDEX = 0x02
    CSR_KIND_VALUE = 0x03
    CSR_KIND_ROW_PTR = 0x04
    CSR_KIND_STREAM_END = 0xFF

    S_WAIT_WORD = 0
    S_RECV_COLS = 1
    S_RECV_VALUES = 2
    S_BLOCK_READY = 3

    def __init__(self, max_rows: int = 6, max_cols: int = 6):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.reset()

    def reset(self) -> None:
        self.state = self.S_WAIT_WORD
        self.rows_q = 0
        self.cols_q = 0
        self.nnzb_prefix_q = 0
        self.block_factor_id_q = 0
        self.block_col_idx_q = 0
        self.block_data_q = [
            [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
        ]
        self.val_row_q = 0
        self.val_col_q = 0

        self.row_ptr_valid_q = False
        self.row_ptr_value_q = 0
        self.row_ptr_row_id_q = 0
        self.stream_done_q = False

        self.block_valid_q = False

    def step(
        self,
        *,
        csr_valid: bool,
        csr_word: CSRWord,
        block_ready: bool,
        row_ptr_ready: bool,
    ) -> dict:
        """Advance one cycle with handshake inputs.

        Returns a dictionary with the registered outputs after the edge:

            {
                "csr_ready": bool,
                "row_ptr_valid": bool,
                "row_ptr_value": int,
                "row_ptr_row_id": int,
                "block_valid": bool,
                "block_rows": int,
                "block_cols": int,
                "nnzb_prefix": int,
                "block_factor_id": int,
                "block_col_idx": int,
                "block_data": List[List[float]],
                "stream_done": bool,
            }
        """

        csr_ready = False
        state_d = self.state
        rows_d = self.rows_q
        cols_d = self.cols_q
        nnzb_prefix_d = self.nnzb_prefix_q
        block_factor_id_d = self.block_factor_id_q
        block_col_idx_d = self.block_col_idx_q
        block_data_d = [row[:] for row in self.block_data_q]
        val_row_d = self.val_row_q
        val_col_d = self.val_col_q
        row_ptr_valid_d = self.row_ptr_valid_q
        row_ptr_value_d = self.row_ptr_value_q
        row_ptr_row_id_d = self.row_ptr_row_id_q
        stream_done_d = False  # pulse only

        if self.state == self.S_WAIT_WORD:
            if csr_valid:
                if csr_word.kind == self.CSR_KIND_ROW_PTR:
                    if not self.row_ptr_valid_q:
                        csr_ready = True
                        row_ptr_valid_d = True
                        row_ptr_value_d = csr_word.row_field
                        row_ptr_row_id_d = csr_word.idx_field

                elif csr_word.kind == self.CSR_KIND_META:
                    if not self.row_ptr_valid_q and not self.block_valid_q:
                        csr_ready = True
                        rows_d = csr_word.rows
                        cols_d = csr_word.cols
                        nnzb_prefix_d = csr_word.row_field
                        block_factor_id_d = csr_word.idx_field
                        block_col_idx_d = 0
                        val_row_d = 0
                        val_col_d = 0
                        block_data_d = [
                            [0.0 for _ in range(self.max_cols)]
                            for _ in range(self.max_rows)
                        ]
                        if csr_word.cols == 0 or csr_word.rows == 0:
                            state_d = self.S_RECV_VALUES
                        else:
                            state_d = self.S_RECV_COLS

                elif csr_word.kind == self.CSR_KIND_STREAM_END:
                    if not self.row_ptr_valid_q and not self.block_valid_q:
                        csr_ready = True
                        stream_done_d = True

        elif self.state == self.S_RECV_COLS:
            if csr_valid:
                csr_ready = True
                block_col_idx_d = csr_word.idx_field
                state_d = self.S_RECV_VALUES
                val_row_d = 0
                val_col_d = 0

        elif self.state == self.S_RECV_VALUES:
            if self.rows_q == 0 or self.cols_q == 0:
                state_d = self.S_BLOCK_READY
            elif csr_valid:
                csr_ready = True
                block_data_d[self.val_row_q][self.val_col_q] = csr_word.value_field
                if self.val_col_q + 1 == self.cols_q:
                    val_col_d = 0
                    val_row_d = self.val_row_q + 1
                else:
                    val_col_d = self.val_col_q + 1

                if (
                    (self.val_col_q + 1 == self.cols_q)
                    and (self.val_row_q + 1 == self.rows_q)
                ):
                    state_d = self.S_BLOCK_READY

        elif self.state == self.S_BLOCK_READY:
            if block_ready:
                state_d = self.S_WAIT_WORD

        # ------------------------------------------------------------------
        # Sequential update
        # ------------------------------------------------------------------
        self.state = state_d
        self.rows_q = rows_d
        self.cols_q = cols_d
        self.nnzb_prefix_q = nnzb_prefix_d
        self.block_factor_id_q = block_factor_id_d
        self.block_col_idx_q = block_col_idx_d
        self.block_data_q = block_data_d
        self.val_row_q = val_row_d
        self.val_col_q = val_col_d

        if self.row_ptr_valid_q and row_ptr_ready:
            self.row_ptr_valid_q = False
        else:
            self.row_ptr_valid_q = row_ptr_valid_d

        if row_ptr_valid_d and (
            not self.row_ptr_valid_q or (self.row_ptr_valid_q and row_ptr_ready)
        ):
            self.row_ptr_value_q = row_ptr_value_d
            self.row_ptr_row_id_q = row_ptr_row_id_d

        self.stream_done_q = stream_done_d

        if self.state == self.S_BLOCK_READY:
            self.block_valid_q = True
        elif block_ready:
            self.block_valid_q = False

        return {
            "csr_ready": csr_ready,
            "row_ptr_valid": self.row_ptr_valid_q,
            "row_ptr_value": self.row_ptr_value_q,
            "row_ptr_row_id": self.row_ptr_row_id_q,
            "block_valid": self.block_valid_q,
            "block_rows": self.rows_q,
            "block_cols": self.cols_q,
            "nnzb_prefix": self.nnzb_prefix_q,
            "block_factor_id": self.block_factor_id_q,
            "block_col_idx": self.block_col_idx_q,
            "block_data": [row[:] for row in self.block_data_q],
            "stream_done": self.stream_done_q,
        }


def demo():
    dec = DecoderModel()
    dec.reset()

    words = [
        CSRWord(kind=DecoderModel.CSR_KIND_ROW_PTR, row_field=0, idx_field=0),
        CSRWord(
            kind=DecoderModel.CSR_KIND_META,
            rows=2,
            cols=2,
            row_field=0,
            idx_field=7,
        ),
        CSRWord(kind=DecoderModel.CSR_KIND_COL_INDEX, idx_field=3),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=1.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=2.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=3.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=4.0),
        CSRWord(kind=DecoderModel.CSR_KIND_ROW_PTR, row_field=1, idx_field=1),
        CSRWord(kind=DecoderModel.CSR_KIND_STREAM_END, row_field=1),
    ]

    csr_index = 0
    block_ready = True
    row_ptr_ready = True

    while csr_index < len(words) or dec.block_valid_q or dec.row_ptr_valid_q:
        csr_valid = csr_index < len(words)
        csr_word = words[csr_index] if csr_valid else CSRWord()
        outputs = dec.step(
            csr_valid=csr_valid,
            csr_word=csr_word,
            block_ready=block_ready,
            row_ptr_ready=row_ptr_ready,
        )
        if outputs["csr_ready"] and csr_valid:
            csr_index += 1

        if outputs["block_valid"]:
            print("Decoded block", outputs["block_data"])
            block_ready = True

    print("Row ptrs:", dec.row_ptr_value_q, dec.row_ptr_row_id_q)


if __name__ == "__main__":
    demo()

```

```python
"""Cycle-accurate Python model of `QR/RTL/decoder.sv`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


class DecoderModel:
    CSR_KIND_META = 0x01
    CSR_KIND_COL_INDEX = 0x02
    CSR_KIND_VALUE = 0x03
    CSR_KIND_ROW_PTR = 0x04
    CSR_KIND_STREAM_END = 0xFF

    S_WAIT_WORD = 0
    S_RECV_COLS = 1
    S_RECV_VALUES = 2
    S_BLOCK_READY = 3

    def __init__(self, max_rows: int = 6, max_cols: int = 6):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.reset()

    def reset(self) -> None:
        self.state = self.S_WAIT_WORD
        self.rows_q = 0
        self.cols_q = 0
        self.nnzb_prefix_q = 0
        self.block_factor_id_q = 0
        self.block_col_idx_q = 0
        self.block_data_q = [
            [0.0 for _ in range(self.max_cols)] for _ in range(self.max_rows)
        ]
        self.val_row_q = 0
        self.val_col_q = 0

        self.row_ptr_valid_q = False
        self.row_ptr_value_q = 0
        self.row_ptr_row_id_q = 0
        self.stream_done_q = False

        self.block_valid_q = False

    def step(
        self,
        *,
        csr_valid: bool,
        csr_word: CSRWord,
        block_ready: bool,
        row_ptr_ready: bool,
    ) -> dict:
        """Advance one cycle with handshake inputs.

        Returns a dictionary with the registered outputs after the edge:

            {
                "csr_ready": bool,
                "row_ptr_valid": bool,
                "row_ptr_value": int,
                "row_ptr_row_id": int,
                "block_valid": bool,
                "block_rows": int,
                "block_cols": int,
                "nnzb_prefix": int,
                "block_factor_id": int,
                "block_col_idx": int,
                "block_data": List[List[float]],
                "stream_done": bool,
            }
        """

        csr_ready = False
        state_d = self.state
        rows_d = self.rows_q
        cols_d = self.cols_q
        nnzb_prefix_d = self.nnzb_prefix_q
        block_factor_id_d = self.block_factor_id_q
        block_col_idx_d = self.block_col_idx_q
        block_data_d = [row[:] for row in self.block_data_q]
        val_row_d = self.val_row_q
        val_col_d = self.val_col_q
        row_ptr_valid_d = self.row_ptr_valid_q
        row_ptr_value_d = self.row_ptr_value_q
        row_ptr_row_id_d = self.row_ptr_row_id_q
        stream_done_d = False  # pulse only

        if self.state == self.S_WAIT_WORD:
            if csr_valid:
                if csr_word.kind == self.CSR_KIND_ROW_PTR:
                    if not self.row_ptr_valid_q:
                        csr_ready = True
                        row_ptr_valid_d = True
                        row_ptr_value_d = csr_word.row_field
                        row_ptr_row_id_d = csr_word.idx_field

                elif csr_word.kind == self.CSR_KIND_META:
                    if not self.row_ptr_valid_q and not self.block_valid_q:
                        csr_ready = True
                        rows_d = csr_word.rows
                        cols_d = csr_word.cols
                        nnzb_prefix_d = csr_word.row_field
                        block_factor_id_d = csr_word.idx_field
                        block_col_idx_d = 0
                        val_row_d = 0
                        val_col_d = 0
                        block_data_d = [
                            [0.0 for _ in range(self.max_cols)]
                            for _ in range(self.max_rows)
                        ]
                        if csr_word.cols == 0 or csr_word.rows == 0:
                            state_d = self.S_RECV_VALUES
                        else:
                            state_d = self.S_RECV_COLS

                elif csr_word.kind == self.CSR_KIND_STREAM_END:
                    if not self.row_ptr_valid_q and not self.block_valid_q:
                        csr_ready = True
                        stream_done_d = True

        elif self.state == self.S_RECV_COLS:
            if csr_valid:
                csr_ready = True
                block_col_idx_d = csr_word.idx_field
                state_d = self.S_RECV_VALUES
                val_row_d = 0
                val_col_d = 0

        elif self.state == self.S_RECV_VALUES:
            if self.rows_q == 0 or self.cols_q == 0:
                state_d = self.S_BLOCK_READY
            elif csr_valid:
                csr_ready = True
                block_data_d[self.val_row_q][self.val_col_q] = csr_word.value_field
                if self.val_col_q + 1 == self.cols_q:
                    val_col_d = 0
                    val_row_d = self.val_row_q + 1
                else:
                    val_col_d = self.val_col_q + 1

                if (
                    (self.val_col_q + 1 == self.cols_q)
                    and (self.val_row_q + 1 == self.rows_q)
                ):
                    state_d = self.S_BLOCK_READY

        elif self.state == self.S_BLOCK_READY:
            if block_ready:
                state_d = self.S_WAIT_WORD

        # ------------------------------------------------------------------
        # Sequential update
        # ------------------------------------------------------------------
        prev_state = self.state
        prev_row_ptr_valid = self.row_ptr_valid_q

        self.state = state_d
        self.rows_q = rows_d
        self.cols_q = cols_d
        self.nnzb_prefix_q = nnzb_prefix_d
        self.block_factor_id_q = block_factor_id_d
        self.block_col_idx_q = block_col_idx_d
        self.block_data_q = block_data_d
        self.val_row_q = val_row_d
        self.val_col_q = val_col_d

        if prev_row_ptr_valid and row_ptr_ready:
            self.row_ptr_valid_q = False
        else:
            self.row_ptr_valid_q = row_ptr_valid_d

        if row_ptr_valid_d and (
            not prev_row_ptr_valid or (prev_row_ptr_valid and row_ptr_ready)
        ):
            self.row_ptr_value_q = row_ptr_value_d
            self.row_ptr_row_id_q = row_ptr_row_id_d

        self.stream_done_q = stream_done_d

        if prev_state == self.S_BLOCK_READY:
            self.block_valid_q = True
        elif block_ready:
            self.block_valid_q = False

        return {
            "csr_ready": csr_ready,
            "row_ptr_valid": self.row_ptr_valid_q,
            "row_ptr_value": self.row_ptr_value_q,
            "row_ptr_row_id": self.row_ptr_row_id_q,
            "block_valid": self.block_valid_q,
            "block_rows": self.rows_q,
            "block_cols": self.cols_q,
            "nnzb_prefix": self.nnzb_prefix_q,
            "block_factor_id": self.block_factor_id_q,
            "block_col_idx": self.block_col_idx_q,
            "block_data": [row[:] for row in self.block_data_q],
            "stream_done": self.stream_done_q,
        }


def demo():
    dec = DecoderModel()
    dec.reset()

    words = [
        CSRWord(kind=DecoderModel.CSR_KIND_ROW_PTR, row_field=0, idx_field=0),
        CSRWord(
            kind=DecoderModel.CSR_KIND_META,
            rows=2,
            cols=2,
            row_field=0,
            idx_field=7,
        ),
        CSRWord(kind=DecoderModel.CSR_KIND_COL_INDEX, idx_field=3),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=1.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=2.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=3.0),
        CSRWord(kind=DecoderModel.CSR_KIND_VALUE, value_field=4.0),
        CSRWord(kind=DecoderModel.CSR_KIND_ROW_PTR, row_field=1, idx_field=1),
        CSRWord(kind=DecoderModel.CSR_KIND_STREAM_END, row_field=1),
    ]

    csr_index = 0
    block_ready = True
    row_ptr_ready = True

    while csr_index < len(words) or dec.block_valid_q or dec.row_ptr_valid_q:
        csr_valid = csr_index < len(words)
        csr_word = words[csr_index] if csr_valid else CSRWord()
        outputs = dec.step(
            csr_valid=csr_valid,
            csr_word=csr_word,
            block_ready=block_ready,
            row_ptr_ready=row_ptr_ready,
        )
        if outputs["csr_ready"] and csr_valid:
            csr_index += 1

        if outputs["block_valid"]:
            print("Decoded block", outputs["block_data"])
            block_ready = True

    print("Row ptrs:", dec.row_ptr_value_q, dec.row_ptr_row_id_q)


if __name__ == "__main__":
    demo()


```

- 新增 `QR/RTL/py/encoder_model.py`，一比一还原 `encoder.sv` 的状态机与握手逻辑，提供 `EncoderModel.step()`（同 RTL 端口：`block_valid/.../csr_ready`）和 `CSRWord` 结构。  
- 新增 `QR/RTL/py/decoder_model.py`，一比一还原 `decoder.sv` 的握手流程与输出寄存器，提供 `DecoderModel.step()`、`CSRWord`。删除旧的 `csr_codec.py`。  
- 两个文件都附带 `demo()` 可直接 `python3 encoder_model.py` / `python3 decoder_model.py` 验证；实际验证时在循环里调用 `step()`，传入当前时钟的输入，读取返回的 `csr_valid/csr_word` 或 `block_valid/row_ptr_valid` 等输出，与 RTL 仿真逐拍比对即可。

---

**User**

和我详细讲解一下encoder.py的逻辑

---

**Cursor**


查看 `encoder_model.py` 的完整代码，然后详细解释其逻辑：



下面详细解释 `encoder_model.py` 的逻辑。

## 整体架构

`EncoderModel` 是一个周期精确的 Python 模型，模拟 `encoder.sv` 的行为。核心是状态机，将并行输入的矩阵块转换为串行的 CSR 数据包流。

---

## 核心数据结构

### `CSRWord` 类（第 8-15 行）

```python
@dataclass
class CSRWord:
    kind: int = 0          # 包类型（META/COL_INDEX/VALUE/ROW_PTR/STREAM_END）
    rows: int = 0          # 块的行数（仅在 META 包中有效）
    cols: int = 0          # 块的列数（仅在 META 包中有效）
    row_field: int = 0     # 行指针值或 nnzb 计数
    idx_field: int = 0     # 列索引、因子ID或行索引（取决于 kind）
    value_field: float = 0.0  # 浮点数值（仅在 VALUE 包中有效）
```

这是 CSR 流中每个数据包的格式，与 RTL 中的 `csr_word_t` 结构体对应。

---

## 状态机设计（第 25-31 行）

状态机有 7 个状态：

```python
S_IDLE = 0                    # 空闲，等待新块
S_SEND_ROW_PTR_START = 1      # 发送行指针（行首）
S_SEND_META = 2               # 发送块元数据
S_SEND_COLS = 3               # 发送列索引
S_SEND_VALUES = 4             # 发送数值（可能多个周期）
S_SEND_ROW_PTR_END = 5        # 发送行指针（行尾）
S_SEND_STREAM_END = 6         # 发送流结束标记
```

---

## 内部寄存器（第 41-59 行）

`reset()` 初始化所有寄存器，对应 RTL 中的 `_q` 寄存器：

- `state`: 当前状态
- `rows_q`, `cols_q`: 当前块的尺寸
- `block_factor_id_q`, `block_col_idx_q`: 因子ID和列索引
- `block_data_q`: 块的稠密数据（二维数组）
- `val_row_q`, `val_col_q`: 发送 VALUE 时的行列计数器
- `nnzb_count_q`: 已发送的块总数（用于构建 row_ptr）
- `row_index_q`: 当前行索引（用于 row_ptr 的 idx_field）
- `csr_valid`, `csr_last`, `csr_word`: 输出寄存器

---

## `step()` 方法详解（第 61-252 行）

`step()` 模拟一个时钟周期，包含组合逻辑和时序更新。

### 第一步：计算握手信号（第 91-92 行）

```python
block_ready = self.state == self.S_IDLE  # 只有空闲时才接受新块
accept_block = block_ready and block_valid  # 握手成功
```

### 第二步：准备下一周期的寄存器值（第 94-105 行）

所有 `_d` 变量是下一周期的值，默认保持当前值：

```python
state_d = self.state  # 默认保持状态
rows_d = self.rows_q  # 默认保持当前值
# ... 其他寄存器类似
```

### 第三步：如果接受新块，更新内部寄存器（第 111-127 行）

```python
if accept_block:
    rows_d = block_rows & 0x7  # 限制在 0-7
    cols_d = block_cols & 0x7
    block_factor_id_d = block_factor_id
    block_col_idx_d = block_col_idx
    last_in_row_d = last_block_in_row
    last_overall_d = last_block_overall
    # 复制块数据到内部寄存器
    block_data_d = [...]
    val_row_d = 0  # 重置计数器
    val_col_d = 0
```

### 第四步：状态机组合逻辑（第 132-224 行）

根据当前状态决定下一状态和输出：

#### 状态 0: `S_IDLE`（第 132-137 行）

```python
if self.state == self.S_IDLE:
    if accept_block:
        if new_block_row:
            state_d = self.S_SEND_ROW_PTR_START  # 新行，先发 ROW_PTR
        else:
            state_d = self.S_SEND_META  # 同一行，直接发 META
```

#### 状态 1: `S_SEND_ROW_PTR_START`（第 139-147 行）

```python
elif self.state == self.S_SEND_ROW_PTR_START:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_ROW_PTR,
        row_field=self.nnzb_count_q,  # 当前块总数
        idx_field=self.row_index_q,   # 当前行索引
    )
    csr_valid_d = True
    if csr_ready:  # 下游接受后，进入下一状态
        state_d = self.S_SEND_META
```

#### 状态 2: `S_SEND_META`（第 149-159 行）

```python
elif self.state == self.S_SEND_META:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_META,
        rows=self.rows_q,              # 块尺寸
        cols=self.cols_q,
        row_field=self.nnzb_count_q,   # nnzb 快照（可选）
        idx_field=self.block_factor_id_q,  # 因子ID
    )
    csr_valid_d = True
    if csr_ready:
        state_d = self.S_SEND_COLS
```

#### 状态 3: `S_SEND_COLS`（第 161-170 行）

```python
elif self.state == self.S_SEND_COLS:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_COL_INDEX,
        idx_field=self.block_col_idx_q,  # 列索引
    )
    csr_valid_d = True
    if csr_ready:
        state_d = self.S_SEND_VALUES
        val_row_d = 0  # 重置计数器
        val_col_d = 0
```

#### 状态 4: `S_SEND_VALUES`（第 172-200 行）

最复杂的状态，需要发送 `rows × cols` 个 VALUE 包：

```python
elif self.state == self.S_SEND_VALUES:
    if self.rows_q == 0 or self.cols_q == 0:
        # 空块：直接跳过，但计数+1
        nnzb_count_d = self.nnzb_count_q + 1
        if self.last_in_row_q or self.last_overall_q:
            state_d = self.S_SEND_ROW_PTR_END
        else:
            state_d = self.S_IDLE
    else:
        # 正常块：发送当前值
        csr_word_d = CSRWord(
            kind=self.CSR_KIND_VALUE,
            value_field=self.block_data_q[self.val_row_q][self.val_col_q],
        )
        csr_valid_d = True
        if csr_ready:
            # 更新行列计数器（行主序）
            if self.val_col_q + 1 == self.cols_q:
                val_col_d = 0
                val_row_d = self.val_row_q + 1
            else:
                val_col_d = self.val_col_q + 1
            
            # 检查是否发送完所有值
            if (self.val_col_q + 1 == self.cols_q) and \
               (self.val_row_q + 1 == self.rows_q):
                nnzb_count_d = self.nnzb_count_q + 1  # 块计数+1
                if self.last_in_row_q or self.last_overall_q:
                    state_d = self.S_SEND_ROW_PTR_END
                else:
                    state_d = self.S_IDLE
```

要点：
- 行主序遍历：先列后行
- 每个 VALUE 包发送一个浮点数
- 发送完所有值后，`nnzb_count` 加 1

#### 状态 5: `S_SEND_ROW_PTR_END`（第 202-214 行）

```python
elif self.state == self.S_SEND_ROW_PTR_END:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_ROW_PTR,
        row_field=self.nnzb_count_q,      # 包含当前块后的总数
        idx_field=self.row_index_q + 1,   # 下一行的索引
    )
    csr_valid_d = True
    if csr_ready:
        row_index_d = self.row_index_q + 1  # 行索引+1
        if self.last_overall_q:
            state_d = self.S_SEND_STREAM_END
        else:
            state_d = self.S_IDLE
```

#### 状态 6: `S_SEND_STREAM_END`（第 216-224 行）

```python
elif self.state == self.S_SEND_STREAM_END:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_STREAM_END,
        row_field=self.nnzb_count_q,  # 最终块总数
    )
    csr_valid_d = True
    csr_last_d = True  # 标记流结束
    if csr_ready:
        state_d = self.S_IDLE
```

### 第五步：时序更新（第 229-245 行）

模拟时钟上升沿，更新所有寄存器：

```python
# 更新状态和所有内部寄存器
self.state = state_d
self.rows_q = rows_d
# ... 其他寄存器

# 输出寄存器的特殊处理（支持 back-pressure）
if csr_ready or not self.csr_valid:
    # 下游接受 OR 当前无效 → 更新输出
    self.csr_valid = csr_valid_d
    self.csr_last = csr_last_d
    self.csr_word = csr_word_d
# 否则：保持当前输出（back-pressure）
```

---

## 完整工作流程示例

假设输入一个 2×2 的块，`new_block_row=True`，`last_block_in_row=True`：

```
周期 0 (S_IDLE):
  输入: block_valid=1, new_block_row=1, block_data=[[1,2],[3,4]]
  输出: block_ready=1 (接受块)
  下一状态: S_SEND_ROW_PTR_START

周期 1 (S_SEND_ROW_PTR_START):
  输出: csr_valid=1, csr_word={kind=ROW_PTR, row_field=0, idx_field=0}
  如果 csr_ready=1: 下一状态 = S_SEND_META

周期 2 (S_SEND_META):
  输出: csr_valid=1, csr_word={kind=META, rows=2, cols=2, idx_field=factor_id}
  如果 csr_ready=1: 下一状态 = S_SEND_COLS

周期 3 (S_SEND_COLS):
  输出: csr_valid=1, csr_word={kind=COL_INDEX, idx_field=col_idx}
  如果 csr_ready=1: 下一状态 = S_SEND_VALUES, val_row=0, val_col=0

周期 4 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=1.0}  # [0][0]
  如果 csr_ready=1: val_col=1

周期 5 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=2.0}  # [0][1]
  如果 csr_ready=1: val_col=0, val_row=1

周期 6 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=3.0}  # [1][0]
  如果 csr_ready=1: val_col=1

周期 7 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=4.0}  # [1][1]
  如果 csr_ready=1: nnzb_count=1, 下一状态 = S_SEND_ROW_PTR_END

周期 8 (S_SEND_ROW_PTR_END):
  输出: csr_valid=1, csr_word={kind=ROW_PTR, row_field=1, idx_field=1}
  如果 csr_ready=1: row_index=1, 下一状态 = S_IDLE
```

---

## 关键设计要点

1. 周期精确：每个 `step()` 对应一个时钟周期
2. 握手协议：`block_valid/ready` 和 `csr_valid/ready` 实现流控
3. Back-pressure：下游不 ready 时保持输出不变
4. 状态机驱动：状态决定输出和下一状态
5. 计数器管理：`nnzb_count` 和 `row_index` 自动维护

这个模型与 RTL 行为一致，可用于验证和调试。

```python
@dataclass
class CSRWord:
    kind: int = 0          # 包类型（META/COL_INDEX/VALUE/ROW_PTR/STREAM_END）
    rows: int = 0          # 块的行数（仅在 META 包中有效）
    cols: int = 0          # 块的列数（仅在 META 包中有效）
    row_field: int = 0     # 行指针值或 nnzb 计数
    idx_field: int = 0     # 列索引、因子ID或行索引（取决于 kind）
    value_field: float = 0.0  # 浮点数值（仅在 VALUE 包中有效）
```

```python
S_IDLE = 0                    # 空闲，等待新块
S_SEND_ROW_PTR_START = 1      # 发送行指针（行首）
S_SEND_META = 2               # 发送块元数据
S_SEND_COLS = 3               # 发送列索引
S_SEND_VALUES = 4             # 发送数值（可能多个周期）
S_SEND_ROW_PTR_END = 5        # 发送行指针（行尾）
S_SEND_STREAM_END = 6         # 发送流结束标记
```

```python
block_ready = self.state == self.S_IDLE  # 只有空闲时才接受新块
accept_block = block_ready and block_valid  # 握手成功
```

```python
state_d = self.state  # 默认保持状态
rows_d = self.rows_q  # 默认保持当前值
# ... 其他寄存器类似
```

```python
if accept_block:
    rows_d = block_rows & 0x7  # 限制在 0-7
    cols_d = block_cols & 0x7
    block_factor_id_d = block_factor_id
    block_col_idx_d = block_col_idx
    last_in_row_d = last_block_in_row
    last_overall_d = last_block_overall
    # 复制块数据到内部寄存器
    block_data_d = [...]
    val_row_d = 0  # 重置计数器
    val_col_d = 0
```

```python
if self.state == self.S_IDLE:
    if accept_block:
        if new_block_row:
            state_d = self.S_SEND_ROW_PTR_START  # 新行，先发 ROW_PTR
        else:
            state_d = self.S_SEND_META  # 同一行，直接发 META
```

```python
elif self.state == self.S_SEND_ROW_PTR_START:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_ROW_PTR,
        row_field=self.nnzb_count_q,  # 当前块总数
        idx_field=self.row_index_q,   # 当前行索引
    )
    csr_valid_d = True
    if csr_ready:  # 下游接受后，进入下一状态
        state_d = self.S_SEND_META
```

```python
elif self.state == self.S_SEND_META:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_META,
        rows=self.rows_q,              # 块尺寸
        cols=self.cols_q,
        row_field=self.nnzb_count_q,   # nnzb 快照（可选）
        idx_field=self.block_factor_id_q,  # 因子ID
    )
    csr_valid_d = True
    if csr_ready:
        state_d = self.S_SEND_COLS
```

```python
elif self.state == self.S_SEND_COLS:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_COL_INDEX,
        idx_field=self.block_col_idx_q,  # 列索引
    )
    csr_valid_d = True
    if csr_ready:
        state_d = self.S_SEND_VALUES
        val_row_d = 0  # 重置计数器
        val_col_d = 0
```

```python
elif self.state == self.S_SEND_VALUES:
    if self.rows_q == 0 or self.cols_q == 0:
        # 空块：直接跳过，但计数+1
        nnzb_count_d = self.nnzb_count_q + 1
        if self.last_in_row_q or self.last_overall_q:
            state_d = self.S_SEND_ROW_PTR_END
        else:
            state_d = self.S_IDLE
    else:
        # 正常块：发送当前值
        csr_word_d = CSRWord(
            kind=self.CSR_KIND_VALUE,
            value_field=self.block_data_q[self.val_row_q][self.val_col_q],
        )
        csr_valid_d = True
        if csr_ready:
            # 更新行列计数器（行主序）
            if self.val_col_q + 1 == self.cols_q:
                val_col_d = 0
                val_row_d = self.val_row_q + 1
            else:
                val_col_d = self.val_col_q + 1
            
            # 检查是否发送完所有值
            if (self.val_col_q + 1 == self.cols_q) and \
               (self.val_row_q + 1 == self.rows_q):
                nnzb_count_d = self.nnzb_count_q + 1  # 块计数+1
                if self.last_in_row_q or self.last_overall_q:
                    state_d = self.S_SEND_ROW_PTR_END
                else:
                    state_d = self.S_IDLE
```

```python
elif self.state == self.S_SEND_ROW_PTR_END:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_ROW_PTR,
        row_field=self.nnzb_count_q,      # 包含当前块后的总数
        idx_field=self.row_index_q + 1,   # 下一行的索引
    )
    csr_valid_d = True
    if csr_ready:
        row_index_d = self.row_index_q + 1  # 行索引+1
        if self.last_overall_q:
            state_d = self.S_SEND_STREAM_END
        else:
            state_d = self.S_IDLE
```

```python
elif self.state == self.S_SEND_STREAM_END:
    csr_word_d = CSRWord(
        kind=self.CSR_KIND_STREAM_END,
        row_field=self.nnzb_count_q,  # 最终块总数
    )
    csr_valid_d = True
    csr_last_d = True  # 标记流结束
    if csr_ready:
        state_d = self.S_IDLE
```

```python
# 更新状态和所有内部寄存器
self.state = state_d
self.rows_q = rows_d
# ... 其他寄存器

# 输出寄存器的特殊处理（支持 back-pressure）
if csr_ready or not self.csr_valid:
    # 下游接受 OR 当前无效 → 更新输出
    self.csr_valid = csr_valid_d
    self.csr_last = csr_last_d
    self.csr_word = csr_word_d
# 否则：保持当前输出（back-pressure）
```

```plaintext
周期 0 (S_IDLE):
  输入: block_valid=1, new_block_row=1, block_data=[[1,2],[3,4]]
  输出: block_ready=1 (接受块)
  下一状态: S_SEND_ROW_PTR_START

周期 1 (S_SEND_ROW_PTR_START):
  输出: csr_valid=1, csr_word={kind=ROW_PTR, row_field=0, idx_field=0}
  如果 csr_ready=1: 下一状态 = S_SEND_META

周期 2 (S_SEND_META):
  输出: csr_valid=1, csr_word={kind=META, rows=2, cols=2, idx_field=factor_id}
  如果 csr_ready=1: 下一状态 = S_SEND_COLS

周期 3 (S_SEND_COLS):
  输出: csr_valid=1, csr_word={kind=COL_INDEX, idx_field=col_idx}
  如果 csr_ready=1: 下一状态 = S_SEND_VALUES, val_row=0, val_col=0

周期 4 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=1.0}  # [0][0]
  如果 csr_ready=1: val_col=1

周期 5 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=2.0}  # [0][1]
  如果 csr_ready=1: val_col=0, val_row=1

周期 6 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=3.0}  # [1][0]
  如果 csr_ready=1: val_col=1

周期 7 (S_SEND_VALUES):
  输出: csr_valid=1, csr_word={kind=VALUE, value_field=4.0}  # [1][1]
  如果 csr_ready=1: nnzb_count=1, 下一状态 = S_SEND_ROW_PTR_END

周期 8 (S_SEND_ROW_PTR_END):
  输出: csr_valid=1, csr_word={kind=ROW_PTR, row_field=1, idx_field=1}
  如果 csr_ready=1: row_index=1, 下一状态 = S_IDLE
```

---

**User**

不是，你这样不是和RTL一样了吗？我不需要这么复杂的状态机，能正确表示encode前后的格式就好了

---

**Cursor**


改为更简单的函数式接口，只做格式转换，不模拟硬件行为。



重写为简单的函数式接口，只做格式转换：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    last_block_overall: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row
        last_block_overall: True if this is the last block in the entire stream
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row or last_block_overall:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )

    # Emit STREAM_END if last block overall
    if last_block_overall:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row, last_block_overall

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            last_block_overall=blk.get("last_block_overall", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```


简化 `decoder_model.py`：


```python
"""Simple functional model for Block-CSR decoding.

This module provides a simple function to convert CSR stream words back into
dense matrix blocks, matching the format consumed by `QR/RTL/decoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


@dataclass
class DecodedBlock:
    """A decoded block with all its metadata."""
    rows: int
    cols: int
    nnzb_prefix: int
    factor_id: int
    col_idx: int
    values: List[List[float]]


@dataclass
class DecodeResult:
    """Result of decoding a CSR stream."""
    row_ptrs: List[tuple[int, int]]  # List of (row_index, nnzb_prefix) tuples
    blocks: List[DecodedBlock]
    stream_done: bool


def decode_stream(words: List[CSRWord], max_rows: int = 6, max_cols: int = 6) -> DecodeResult:
    """Decode a CSR word stream into blocks and row pointers.

    Args:
        words: List of CSRWord objects from the encoder
        max_rows: Maximum rows in a block (for allocating storage)
        max_cols: Maximum cols in a block (for allocating storage)

    Returns:
        DecodeResult containing row_ptrs, blocks, and stream_done flag.
    """
    row_ptrs = []
    blocks = []
    stream_done = False

    i = 0
    while i < len(words):
        word = words[i]

        if word.kind == CSR_KIND_ROW_PTR:
            row_ptrs.append((word.idx_field, word.row_field))
            i += 1

        elif word.kind == CSR_KIND_META:
            # Start of a new block
            rows = word.rows
            cols = word.cols
            nnzb_prefix = word.row_field
            factor_id = word.idx_field

            # Read COL_INDEX
            if i + 1 < len(words) and words[i + 1].kind == CSR_KIND_COL_INDEX:
                col_idx = words[i + 1].idx_field
                i += 2
            else:
                col_idx = 0
                i += 1

            # Read VALUE words
            values = []
            if rows > 0 and cols > 0:
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        if i < len(words) and words[i].kind == CSR_KIND_VALUE:
                            row.append(words[i].value_field)
                            i += 1
                        else:
                            row.append(0.0)
                    values.append(row)
            else:
                # Empty block
                values = [[0.0 for _ in range(max_cols)] for _ in range(max_rows)]

            blocks.append(
                DecodedBlock(
                    rows=rows,
                    cols=cols,
                    nnzb_prefix=nnzb_prefix,
                    factor_id=factor_id,
                    col_idx=col_idx,
                    values=values,
                )
            )

        elif word.kind == CSR_KIND_STREAM_END:
            stream_done = True
            i += 1

        else:
            # Skip unknown words
            i += 1

    return DecodeResult(
        row_ptrs=row_ptrs,
        blocks=blocks,
        stream_done=stream_done,
    )


def demo():
    """Example usage."""
    from encoder_model import encode_blocks

    # Encode some blocks
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 8,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Encoded {len(words)} words")

    # Decode them back
    result = decode_stream(words)
    print(f"\nDecoded {len(result.blocks)} blocks:")
    for i, blk in enumerate(result.blocks):
        print(f"  Block {i}: rows={blk.rows}, cols={blk.cols}, "
              f"factor_id={blk.factor_id}, col_idx={blk.col_idx}")
        print(f"    Values: {blk.values}")

    print(f"\nRow pointers: {result.row_ptrs}")
    print(f"Stream done: {result.stream_done}")


if __name__ == "__main__":
    demo()

```


测试这两个简化后的文件：



已简化两个文件，改为简单的函数式接口，不再模拟硬件状态机。

## 简化后的架构

### `encoder_model.py`
- **`encode_block()`**: 输入一个块的所有信息，直接返回该块对应的 CSR 字列表
- **`encode_blocks()`**: 批量编码多个块
- 无状态机，无握手协议，只做格式转换

### `decoder_model.py`
- **`decode_stream()`**: 输入 CSR 字列表，返回解码后的块和行指针
- 返回 `DecodeResult` 对象，包含 `row_ptrs`、`blocks`、`stream_done`
- 无状态机，直接解析字流

## 使用示例

```python
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 编码
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": True,
        "last_block_overall": True,
    }
]

words = encode_blocks(blocks)  # 直接得到 CSR 字列表

# 解码
result = decode_stream(words)  # 直接得到解码结果
print(result.blocks[0].values)  # [[1.0, 2.0], [3.0, 4.0]]
```

现在这两个文件只关注格式转换，不模拟硬件行为，更适合验证和测试。

```python
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 编码
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": True,
        "last_block_overall": True,
    }
]

words = encode_blocks(blocks)  # 直接得到 CSR 字列表

# 解码
result = decode_stream(words)  # 直接得到解码结果
print(result.blocks[0].values)  # [[1.0, 2.0], [3.0, 4.0]]
```

---

**User**

按行给我介绍两个代码的内容

---

**Cursor**


逐行解释这两个文件：



按行解释这两个文件：

---

## `encoder_model.py` 逐行解释

### 文件头部和导入（1-10行）

```1:10:QR/RTL/py/encoder_model.py
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
```

- 1-5行：模块文档字符串，说明这是 Block-CSR 编码的简单函数式模型
- 7行：启用类型注解的前向引用
- 9行：导入 `dataclass` 装饰器
- 10行：导入 `List` 类型提示

### CSRWord 数据类（13-21行）

```13:21:QR/RTL/py/encoder_model.py
@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0
```

- 13行：使用 `@dataclass` 定义数据类
- 14-15行：`CSRWord` 类，表示一个 CSR 流数据包
- 16行：`kind` - 包类型（META/COL_INDEX/VALUE/ROW_PTR/STREAM_END）
- 17-18行：`rows`, `cols` - 块尺寸（仅在 META 包中有效）
- 19行：`row_field` - 行指针值或 nnzb 计数
- 20行：`idx_field` - 列索引、因子ID或行索引（取决于 kind）
- 21行：`value_field` - 浮点数值（仅在 VALUE 包中有效）

### 常量定义（24-29行）

```24:29:QR/RTL/py/encoder_model.py
# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF
```

- 24行：注释说明这些常量与 RTL 匹配
- 25-29行：定义 5 种 CSR 包类型常量，与 RTL 中的 `CSR_KIND_*` 对应

### encode_block 函数（32-128行）

```32:43:QR/RTL/py/encoder_model.py
def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    last_block_overall: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
```

- 32-43行：函数签名
  - `block_rows/cols`: 块尺寸
  - `block_factor_id`: 因子ID
  - `block_col_idx`: 列索引
  - `block_data`: 稠密矩阵数据（二维列表）
  - `new_block_row`: 是否是新行的第一个块
  - `last_block_in_row`: 是否是行尾块
  - `last_block_overall`: 是否是最后一个块
  - `nnzb_before`: 编码前已发送的块数
  - `row_index`: 当前行索引
  - 返回：`(CSR字列表, 编码后的块数, 下一个行索引)`

```64:66:QR/RTL/py/encoder_model.py
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index
```

- 64行：初始化输出字列表
- 65行：计算编码后的块数（当前块+1）
- 66行：初始化下一个行索引（默认不变）

```68:76:QR/RTL/py/encoder_model.py
    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )
```

- 68-76行：如果是新行的第一个块，先发送 ROW_PTR 包
  - `row_field`: 编码前的块总数
  - `idx_field`: 当前行索引

```78:87:QR/RTL/py/encoder_model.py
    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )
```

- 78-87行：发送 META 包
  - `rows/cols`: 块尺寸
  - `row_field`: nnzb 快照
  - `idx_field`: 因子ID

```89:95:QR/RTL/py/encoder_model.py
    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )
```

- 89-95行：发送 COL_INDEX 包，`idx_field` 为列索引

```97:106:QR/RTL/py/encoder_model.py
    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )
```

- 97-106行：发送 VALUE 包（行主序）
  - 遍历所有行列，每个元素一个 VALUE 包
  - 104行：边界检查，越界填 0.0

```108:117:QR/RTL/py/encoder_model.py
    # Emit ROW_PTR at end of row
    if last_block_in_row or last_block_overall:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
```

- 108-117行：如果是行尾或最后一个块，发送 ROW_PTR 包
  - `row_field`: 编码后的块总数
  - `idx_field`: 下一个行索引
  - 110行：行索引+1

```119:126:QR/RTL/py/encoder_model.py
    # Emit STREAM_END if last block overall
    if last_block_overall:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )
```

- 119-126行：如果是最后一个块，发送 STREAM_END 包
  - `row_field`: 最终块总数

```128:128:QR/RTL/py/encoder_model.py
    return words, nnzb_after, next_row_index
```

- 128行：返回 CSR 字列表、编码后的块数、下一个行索引

### encode_blocks 函数（131-162行）

```131:142:QR/RTL/py/encoder_model.py
def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row, last_block_overall

    Returns:
        List of CSRWord objects representing the entire stream.
    """
```

- 131-142行：批量编码函数，输入块字典列表，返回 CSR 字列表

```143:160:QR/RTL/py/encoder_model.py
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            last_block_overall=blk.get("last_block_overall", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)
```

- 143-145行：初始化输出列表、块计数器、行索引
- 147行：遍历每个块
- 148-159行：调用 `encode_block` 编码单个块，更新 `nnzb` 和 `row_idx`
- 160行：将当前块的 CSR 字追加到总列表

```162:162:QR/RTL/py/encoder_model.py
    return words
```

- 162行：返回所有 CSR 字

### demo 函数（165-195行）

```165:188:QR/RTL/py/encoder_model.py
def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]
```

- 165-188行：示例数据，两个 2×2 块

```190:195:QR/RTL/py/encoder_model.py
    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")
```

- 190行：编码所有块
- 191-195行：打印每个 CSR 字的详细信息

```198:199:QR/RTL/py/encoder_model.py
if __name__ == "__main__":
    demo()
```

- 198-199行：主程序入口，直接运行文件时执行 demo

---

## `decoder_model.py` 逐行解释

### 文件头部和导入（1-10行）

```1:10:QR/RTL/py/decoder_model.py
"""Simple functional model for Block-CSR decoding.

This module provides a simple function to convert CSR stream words back into
dense matrix blocks, matching the format consumed by `QR/RTL/decoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
```

- 1-5行：模块文档字符串，说明这是解码模型
- 7行：启用类型注解前向引用
- 9行：导入 `dataclass`
- 10行：导入 `List`, `Optional` 类型提示

### CSRWord 数据类（13-21行）

```13:21:QR/RTL/py/decoder_model.py
@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0
```

- 与 encoder 中的 `CSRWord` 相同，用于表示输入流

### 常量定义（24-29行）

```24:29:QR/RTL/py/decoder_model.py
# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF
```

- 与 encoder 中的常量一致

### DecodedBlock 数据类（32-40行）

```32:40:QR/RTL/py/decoder_model.py
@dataclass
class DecodedBlock:
    """A decoded block with all its metadata."""
    rows: int
    cols: int
    nnzb_prefix: int
    factor_id: int
    col_idx: int
    values: List[List[float]]
```

- 32-40行：解码后的块结构
  - `rows/cols`: 块尺寸
  - `nnzb_prefix`: nnzb 前缀计数
  - `factor_id`: 因子ID
  - `col_idx`: 列索引
  - `values`: 稠密矩阵数据

### DecodeResult 数据类（43-48行）

```43:48:QR/RTL/py/decoder_model.py
@dataclass
class DecodeResult:
    """Result of decoding a CSR stream."""
    row_ptrs: List[tuple[int, int]]  # List of (row_index, nnzb_prefix) tuples
    blocks: List[DecodedBlock]
    stream_done: bool
```

- 43-48行：解码结果
  - `row_ptrs`: 行指针列表，每个元素是 `(行索引, nnzb前缀)`
  - `blocks`: 解码后的块列表
  - `stream_done`: 是否遇到 STREAM_END

### decode_stream 函数（51-128行）

```51:61:QR/RTL/py/decoder_model.py
def decode_stream(words: List[CSRWord], max_rows: int = 6, max_cols: int = 6) -> DecodeResult:
    """Decode a CSR word stream into blocks and row pointers.

    Args:
        words: List of CSRWord objects from the encoder
        max_rows: Maximum rows in a block (for allocating storage)
        max_cols: Maximum cols in a block (for allocating storage)

    Returns:
        DecodeResult containing row_ptrs, blocks, and stream_done flag.
    """
```

- 51-61行：解码函数签名
  - `words`: CSR 字列表
  - `max_rows/cols`: 最大块尺寸（用于空块分配）
  - 返回：`DecodeResult`

```62:64:QR/RTL/py/decoder_model.py
    row_ptrs = []
    blocks = []
    stream_done = False
```

- 62-64行：初始化输出列表和标志

```66:72:QR/RTL/py/decoder_model.py
    i = 0
    while i < len(words):
        word = words[i]

        if word.kind == CSR_KIND_ROW_PTR:
            row_ptrs.append((word.idx_field, word.row_field))
            i += 1
```

- 66行：初始化索引
- 67行：遍历所有字
- 68行：获取当前字
- 70-72行：如果是 ROW_PTR，记录到 `row_ptrs`，索引+1

```74:114:QR/RTL/py/decoder_model.py
        elif word.kind == CSR_KIND_META:
            # Start of a new block
            rows = word.rows
            cols = word.cols
            nnzb_prefix = word.row_field
            factor_id = word.idx_field

            # Read COL_INDEX
            if i + 1 < len(words) and words[i + 1].kind == CSR_KIND_COL_INDEX:
                col_idx = words[i + 1].idx_field
                i += 2
            else:
                col_idx = 0
                i += 1

            # Read VALUE words
            values = []
            if rows > 0 and cols > 0:
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        if i < len(words) and words[i].kind == CSR_KIND_VALUE:
                            row.append(words[i].value_field)
                            i += 1
                        else:
                            row.append(0.0)
                    values.append(row)
            else:
                # Empty block
                values = [[0.0 for _ in range(max_cols)] for _ in range(max_rows)]

            blocks.append(
                DecodedBlock(
                    rows=rows,
                    cols=cols,
                    nnzb_prefix=nnzb_prefix,
                    factor_id=factor_id,
                    col_idx=col_idx,
                    values=values,
                )
            )
```

- 74行：如果是 META，开始解码一个新块
- 76-79行：从 META 提取尺寸、nnzb 前缀、因子ID
- 81-87行：读取下一个 COL_INDEX（如果存在），否则 `col_idx=0`
- 89-100行：读取 VALUE 包（行主序），按 `rows×cols` 重组为二维列表
- 101-103行：空块时分配 `max_rows×max_cols` 的零矩阵
- 105-114行：创建 `DecodedBlock` 并追加到 `blocks`

```116:122:QR/RTL/py/decoder_model.py
        elif word.kind == CSR_KIND_STREAM_END:
            stream_done = True
            i += 1

        else:
            # Skip unknown words
            i += 1
```

- 116-118行：遇到 STREAM_END 时设置标志
- 120-122行：跳过未知类型的字

```124:128:QR/RTL/py/decoder_model.py
    return DecodeResult(
        row_ptrs=row_ptrs,
        blocks=blocks,
        stream_done=stream_done,
    )
```

- 124-128行：返回解码结果

### demo 函数（131-171行）

```131:157:QR/RTL/py/decoder_model.py
def demo():
    """Example usage."""
    from encoder_model import encode_blocks

    # Encode some blocks
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 8,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]
```

- 131-157行：示例数据，导入 encoder 用于测试

```159:171:QR/RTL/py/decoder_model.py
    words = encode_blocks(blocks)
    print(f"Encoded {len(words)} words")

    # Decode them back
    result = decode_stream(words)
    print(f"\nDecoded {len(result.blocks)} blocks:")
    for i, blk in enumerate(result.blocks):
        print(f"  Block {i}: rows={blk.rows}, cols={blk.cols}, "
              f"factor_id={blk.factor_id}, col_idx={blk.col_idx}")
        print(f"    Values: {blk.values}")

    print(f"\nRow pointers: {result.row_ptrs}")
    print(f"Stream done: {result.stream_done}")
```

- 159行：编码块
- 163行：解码
- 164-168行：打印每个块的详细信息
- 170-171行：打印行指针和流结束标志

```174:175:QR/RTL/py/decoder_model.py
if __name__ == "__main__":
    demo()
```

- 174-175行：主程序入口

---

## 总结

- `encoder_model.py`：将稠密矩阵块编码为 CSR 字流
- `decoder_model.py`：将 CSR 字流解码回稠密矩阵块

两者都是函数式接口，不模拟硬件状态机，只做格式转换，适合验证和测试。

```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
```

```python
@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0
```

```python
# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF
```

```python
def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    last_block_overall: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
```

```python
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index
```

```python
    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )
```

```python
    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )
```

```python
    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )
```

```python
    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )
```

```python
    # Emit ROW_PTR at end of row
    if last_block_in_row or last_block_overall:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
```

```python
    # Emit STREAM_END if last block overall
    if last_block_overall:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )
```

```python
    return words, nnzb_after, next_row_index
```

```python
def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row, last_block_overall

    Returns:
        List of CSRWord objects representing the entire stream.
    """
```

```python
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            last_block_overall=blk.get("last_block_overall", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)
```

```python
    return words
```

```python
def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]
```

```python
    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")
```

```python
if __name__ == "__main__":
    demo()
```

```python
"""Simple functional model for Block-CSR decoding.

This module provides a simple function to convert CSR stream words back into
dense matrix blocks, matching the format consumed by `QR/RTL/decoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
```

```python
@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0
```

```python
# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF
```

```python
@dataclass
class DecodedBlock:
    """A decoded block with all its metadata."""
    rows: int
    cols: int
    nnzb_prefix: int
    factor_id: int
    col_idx: int
    values: List[List[float]]
```

```python
@dataclass
class DecodeResult:
    """Result of decoding a CSR stream."""
    row_ptrs: List[tuple[int, int]]  # List of (row_index, nnzb_prefix) tuples
    blocks: List[DecodedBlock]
    stream_done: bool
```

```python
def decode_stream(words: List[CSRWord], max_rows: int = 6, max_cols: int = 6) -> DecodeResult:
    """Decode a CSR word stream into blocks and row pointers.

    Args:
        words: List of CSRWord objects from the encoder
        max_rows: Maximum rows in a block (for allocating storage)
        max_cols: Maximum cols in a block (for allocating storage)

    Returns:
        DecodeResult containing row_ptrs, blocks, and stream_done flag.
    """
```

```python
    row_ptrs = []
    blocks = []
    stream_done = False
```

```python
    i = 0
    while i < len(words):
        word = words[i]

        if word.kind == CSR_KIND_ROW_PTR:
            row_ptrs.append((word.idx_field, word.row_field))
            i += 1
```

```python
        elif word.kind == CSR_KIND_META:
            # Start of a new block
            rows = word.rows
            cols = word.cols
            nnzb_prefix = word.row_field
            factor_id = word.idx_field

            # Read COL_INDEX
            if i + 1 < len(words) and words[i + 1].kind == CSR_KIND_COL_INDEX:
                col_idx = words[i + 1].idx_field
                i += 2
            else:
                col_idx = 0
                i += 1

            # Read VALUE words
            values = []
            if rows > 0 and cols > 0:
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        if i < len(words) and words[i].kind == CSR_KIND_VALUE:
                            row.append(words[i].value_field)
                            i += 1
                        else:
                            row.append(0.0)
                    values.append(row)
            else:
                # Empty block
                values = [[0.0 for _ in range(max_cols)] for _ in range(max_rows)]

            blocks.append(
                DecodedBlock(
                    rows=rows,
                    cols=cols,
                    nnzb_prefix=nnzb_prefix,
                    factor_id=factor_id,
                    col_idx=col_idx,
                    values=values,
                )
            )
```

```python
        elif word.kind == CSR_KIND_STREAM_END:
            stream_done = True
            i += 1

        else:
            # Skip unknown words
            i += 1
```

```python
    return DecodeResult(
        row_ptrs=row_ptrs,
        blocks=blocks,
        stream_done=stream_done,
    )
```

```python
def demo():
    """Example usage."""
    from encoder_model import encode_blocks

    # Encode some blocks
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 8,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]
```

```python
    words = encode_blocks(blocks)
    print(f"Encoded {len(words)} words")

    # Decode them back
    result = decode_stream(words)
    print(f"\nDecoded {len(result.blocks)} blocks:")
    for i, blk in enumerate(result.blocks):
        print(f"  Block {i}: rows={blk.rows}, cols={blk.cols}, "
              f"factor_id={blk.factor_id}, col_idx={blk.col_idx}")
        print(f"    Values: {blk.values}")

    print(f"\nRow pointers: {result.row_ptrs}")
    print(f"Stream done: {result.stream_done}")
```

```python
if __name__ == "__main__":
    demo()
```

---

**User**

不需要last_block_overall吧

---

**Cursor**


检查代码中 `last_block_overall` 的使用，并移除它：



移除 `last_block_overall`，在 `encode_blocks` 中自动判断最后一个块：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    is_last_block: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row
        is_last_block: True if this is the last block in the entire stream
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )

    # Emit STREAM_END if last block overall
    if is_last_block:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row, last_block_overall

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            last_block_overall=blk.get("last_block_overall", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```

```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    is_last_block: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row
        is_last_block: True if this is the last block in the entire stream
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )

    # Emit STREAM_END if last block overall
    if is_last_block:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for i, blk in enumerate(blocks):
        is_last_block = (i == len(blocks) - 1)
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            is_last_block=is_last_block,
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
            "last_block_overall": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
            "last_block_overall": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```


更新 demo 函数，移除 `last_block_overall`：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    is_last_block: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row
        is_last_block: True if this is the last block in the entire stream
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )

    # Emit STREAM_END if last block overall
    if is_last_block:
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for i, blk in enumerate(blocks):
        is_last_block = (i == len(blocks) - 1)
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            is_last_block=is_last_block,
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```


更新 `decoder_model.py` 中的 demo：


```python
"""Simple functional model for Block-CSR decoding.

This module provides a simple function to convert CSR stream words back into
dense matrix blocks, matching the format consumed by `QR/RTL/decoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


@dataclass
class DecodedBlock:
    """A decoded block with all its metadata."""
    rows: int
    cols: int
    nnzb_prefix: int
    factor_id: int
    col_idx: int
    values: List[List[float]]


@dataclass
class DecodeResult:
    """Result of decoding a CSR stream."""
    row_ptrs: List[tuple[int, int]]  # List of (row_index, nnzb_prefix) tuples
    blocks: List[DecodedBlock]
    stream_done: bool


def decode_stream(words: List[CSRWord], max_rows: int = 6, max_cols: int = 6) -> DecodeResult:
    """Decode a CSR word stream into blocks and row pointers.

    Args:
        words: List of CSRWord objects from the encoder
        max_rows: Maximum rows in a block (for allocating storage)
        max_cols: Maximum cols in a block (for allocating storage)

    Returns:
        DecodeResult containing row_ptrs, blocks, and stream_done flag.
    """
    row_ptrs = []
    blocks = []
    stream_done = False

    i = 0
    while i < len(words):
        word = words[i]

        if word.kind == CSR_KIND_ROW_PTR:
            row_ptrs.append((word.idx_field, word.row_field))
            i += 1

        elif word.kind == CSR_KIND_META:
            # Start of a new block
            rows = word.rows
            cols = word.cols
            nnzb_prefix = word.row_field
            factor_id = word.idx_field

            # Read COL_INDEX
            if i + 1 < len(words) and words[i + 1].kind == CSR_KIND_COL_INDEX:
                col_idx = words[i + 1].idx_field
                i += 2
            else:
                col_idx = 0
                i += 1

            # Read VALUE words
            values = []
            if rows > 0 and cols > 0:
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        if i < len(words) and words[i].kind == CSR_KIND_VALUE:
                            row.append(words[i].value_field)
                            i += 1
                        else:
                            row.append(0.0)
                    values.append(row)
            else:
                # Empty block
                values = [[0.0 for _ in range(max_cols)] for _ in range(max_rows)]

            blocks.append(
                DecodedBlock(
                    rows=rows,
                    cols=cols,
                    nnzb_prefix=nnzb_prefix,
                    factor_id=factor_id,
                    col_idx=col_idx,
                    values=values,
                )
            )

        elif word.kind == CSR_KIND_STREAM_END:
            stream_done = True
            i += 1

        else:
            # Skip unknown words
            i += 1

    return DecodeResult(
        row_ptrs=row_ptrs,
        blocks=blocks,
        stream_done=stream_done,
    )


def demo():
    """Example usage."""
    from encoder_model import encode_blocks

    # Encode some blocks
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 8,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Encoded {len(words)} words")

    # Decode them back
    result = decode_stream(words)
    print(f"\nDecoded {len(result.blocks)} blocks:")
    for i, blk in enumerate(result.blocks):
        print(f"  Block {i}: rows={blk.rows}, cols={blk.cols}, "
              f"factor_id={blk.factor_id}, col_idx={blk.col_idx}")
        print(f"    Values: {blk.values}")

    print(f"\nRow pointers: {result.row_ptrs}")
    print(f"Stream done: {result.stream_done}")


if __name__ == "__main__":
    demo()

```


测试修改后的代码：



已移除 `last_block_overall` 参数。改动如下：

## 主要改动

### `encoder_model.py`

1. **`encode_block()` 函数**：
   - 将参数 `last_block_overall` 改为 `is_last_block`
   - 第 109 行：`ROW_PTR_END` 的发送条件改为仅检查 `last_block_in_row`（不再需要 `last_block_overall`）
   - 第 120 行：`STREAM_END` 的发送条件改为检查 `is_last_block`

2. **`encode_blocks()` 函数**：
   - 第 147-148 行：自动判断最后一个块 `is_last_block = (i == len(blocks) - 1)`
   - 不再需要用户传入 `last_block_overall`，函数内部自动处理

3. **`demo()` 函数**：
   - 移除了所有块字典中的 `last_block_overall` 字段

### `decoder_model.py`

- 更新了 `demo()` 函数中的示例数据，移除了 `last_block_overall` 字段

## 使用方式

现在使用更简单：

```python
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": False,
        # 不再需要 last_block_overall
    },
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 4,
        "block_data": [[5.0, 6.0], [7.0, 8.0]],
        "new_block_row": False,
        "last_block_in_row": True,
        # 最后一个块会自动发送 STREAM_END
    },
]

words = encode_blocks(blocks)  # 自动处理最后一个块
```

测试通过，功能正常。

```python
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": False,
        # 不再需要 last_block_overall
    },
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 4,
        "block_data": [[5.0, 6.0], [7.0, 8.0]],
        "new_block_row": False,
        "last_block_in_row": True,
        # 最后一个块会自动发送 STREAM_END
    },
]

words = encode_blocks(blocks)  # 自动处理最后一个块
```

---

**User**

在 encode_blocks 中自动判断最后一个块是什么意思？我不需要啊，我最后一个block不就是通过last_block_in_row判断的吗

---

**Cursor**


移除 `is_last_block` 的自动判断，改为在 `last_block_in_row=True` 时发送 `STREAM_END`。



移除 `is_last_block`，改为在 `last_block_in_row=True` 时发送 `STREAM_END`。


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for i, blk in enumerate(blocks):
        is_last_block = (i == len(blocks) - 1)
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            is_last_block=is_last_block,
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```


更新 `encode_blocks`，移除自动判断逻辑：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words")
    for i, word in enumerate(words):
        print(f"  {i}: kind={word.kind:02x}, rows={word.rows}, cols={word.cols}, "
              f"row_field={word.row_field}, idx_field={word.idx_field}, "
              f"value={word.value_field}")


if __name__ == "__main__":
    demo()

```


测试修改后的代码：



已按你的要求修改：

## 主要改动

1. **移除了 `is_last_block` 参数**：`encode_block()` 不再需要这个参数
2. **移除了自动判断逻辑**：`encode_blocks()` 不再自动判断最后一个块
3. **简化逻辑**：当 `last_block_in_row=True` 时，自动发送 `STREAM_END`

现在的逻辑是：
- 如果 `last_block_in_row=True`，这个块就是整个流的最后一个块
- 此时会发送 `ROW_PTR_END` 和 `STREAM_END`

使用方式：

```python
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": False,  # 不是最后一个
    },
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 4,
        "block_data": [[5.0, 6.0], [7.0, 8.0]],
        "new_block_row": False,
        "last_block_in_row": True,  # 最后一个块，会自动发送 STREAM_END
    },
]
```

测试通过，功能正常。

```python
blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 3,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": False,  # 不是最后一个
    },
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 4,
        "block_data": [[5.0, 6.0], [7.0, 8.0]],
        "new_block_row": False,
        "last_block_in_row": True,  # 最后一个块，会自动发送 STREAM_END
    },
]
```

---

**User**

是合理的吧 因为我一行就是一次流

---

**Cursor**


是的，这个逻辑是合理的。

你的场景是：
- 每次处理一行（一个 block-row）
- 一行处理完，流就结束
- 所以 `last_block_in_row=True` 就表示整个流结束

因此代码逻辑是正确的：
- 当 `last_block_in_row=True` 时，发送 `ROW_PTR_END`（标记行结束）
- 同时发送 `STREAM_END`（标记整个流结束）

这样设计的好处：
1. 简单：不需要额外的 `last_block_overall` 标志
2. 符合你的使用场景：一行 = 一次流
3. 与 RTL 一致：RTL 中 `last_block_in_row` 和 `last_block_overall` 在单行场景下是等价的

代码已按此逻辑实现，无需修改。

---

**User**

你是什么model？

---

**Cursor**


我是 **Auto**，由 Cursor 设计的 agent router，用于协助编程任务。

我可以：
- 阅读和分析代码
- 编写和修改代码
- 运行命令和测试
- 解释代码逻辑
- 调试问题

关于你的项目，我理解你在做 QR 分解的硬件实现，涉及：
- SystemVerilog RTL 设计（encoder/decoder）
- Block CSR 格式的编码/解码
- Python 验证模型

需要我继续处理什么？

---

**User**

教下我怎么用我的@encoder_model.py 和@decoder_model.py 

---

**Cursor**


使用这两个模型的方法：

## 基本使用

### 1. 编码（Encoder）

```python
from encoder_model import encode_blocks

# 准备你的块数据
blocks = [
    {
        "block_rows": 6,           # 块的行数
        "block_cols": 6,           # 块的列数
        "block_factor_id": 5,      # 因子ID（来自哪个因子）
        "block_col_idx": 0,       # 列索引（对应哪个变量，如y1=0, x1=1）
        "block_data": [            # 稠密矩阵数据（行主序）
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            # ... 更多行
        ],
        "new_block_row": True,     # 是否是新行的第一个块
        "last_block_in_row": False, # 是否是行尾块
    },
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 5,
        "block_col_idx": 1,        # 下一个变量列
        "block_data": [[...]],     # 下一个块的数据
        "new_block_row": False,    # 同一行的后续块
        "last_block_in_row": True, # 最后一个块（会自动发送STREAM_END）
    },
]

# 编码成 CSR 流
words = encode_blocks(blocks)
print(f"生成了 {len(words)} 个 CSR 字")
```

### 2. 解码（Decoder）

```python
from decoder_model import decode_stream

# 解码 CSR 流
result = decode_stream(words)

# 查看结果
print(f"解码了 {len(result.blocks)} 个块")
print(f"行指针: {result.row_ptrs}")
print(f"流结束: {result.stream_done}")

# 访问每个块
for i, blk in enumerate(result.blocks):
    print(f"块 {i}:")
    print(f"  尺寸: {blk.rows}×{blk.cols}")
    print(f"  因子ID: {blk.factor_id}")
    print(f"  列索引: {blk.col_idx}")
    print(f"  数据: {blk.values}")
```

## 完整示例：编码-解码验证

```python
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 模拟你的 QR 分解结果：一行 g 包含多个块
blocks = [
    # 第一个块：y1 列的 R 块（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,      # 来自 factor f1
        "block_col_idx": 0,        # y1 的列索引
        "block_data": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [0.0, 0.0, 12.0, 13.0, 14.0, 15.0],
            [0.0, 0.0, 0.0, 16.0, 17.0, 18.0],
            [0.0, 0.0, 0.0, 0.0, 19.0, 20.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 21.0],
        ],
        "new_block_row": True,     # 这是新行的第一个块
        "last_block_in_row": False,
    },
    # 第二个块：x1 列的方阵（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 1,        # x1 的列索引
        "block_data": [
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0, 32.0, 33.0],
            # ... 更多行
        ],
        "new_block_row": False,
        "last_block_in_row": True, # 最后一个块
    },
]

# 编码
words = encode_blocks(blocks)
print(f"编码完成，共 {len(words)} 个 CSR 字")

# 解码验证
result = decode_stream(words)
print(f"\n解码完成，共 {len(result.blocks)} 个块")

# 验证数据是否正确
for i, (original, decoded) in enumerate(zip(blocks, result.blocks)):
    if original["block_data"] == decoded.values:
        print(f"✓ 块 {i} 数据匹配")
    else:
        print(f"✗ 块 {i} 数据不匹配")
```

## 实际使用场景

### 场景 1：验证 RTL 仿真输出

```python
# 从 RTL 仿真日志中读取 CSR 字
def parse_rtl_output(log_file):
    words = []
    # 解析日志，构造 CSRWord 对象
    # ...
    return words

# 解码并验证
rtl_words = parse_rtl_output("simulation.log")
result = decode_stream(rtl_words)

# 与期望值对比
expected_blocks = [...]  # 你的期望值
assert len(result.blocks) == len(expected_blocks)
```

### 场景 2：生成测试向量

```python
# 生成测试用的块数据
def generate_test_blocks():
    blocks = []
    for col_idx in range(3):  # 3 个变量列
        blocks.append({
            "block_rows": 6,
            "block_cols": 6,
            "block_factor_id": 1,
            "block_col_idx": col_idx,
            "block_data": [[float(i*6+j) for j in range(6)] for i in range(6)],
            "new_block_row": (col_idx == 0),
            "last_block_in_row": (col_idx == 2),
        })
    return blocks

# 生成 CSR 流用于测试
test_blocks = generate_test_blocks()
test_words = encode_blocks(test_blocks)
```

### 场景 3：调试单个块

```python
from encoder_model import encode_block

# 单独编码一个块（用于调试）
words, nnzb_after, next_row_idx = encode_block(
    block_rows=2,
    block_cols=2,
    block_factor_id=7,
    block_col_idx=3,
    block_data=[[1.0, 2.0], [3.0, 4.0]],
    new_block_row=True,
    last_block_in_row=True,
    nnzb_before=0,
    row_index=0,
)

print(f"编码后块数: {nnzb_after}")
print(f"下一个行索引: {next_row_idx}")
for word in words:
    print(f"  {word.kind:02x}: {word}")
```

## 关键点

1. `new_block_row=True`：新行的第一个块，会发送 `ROW_PTR_START`
2. `last_block_in_row=True`：行尾块，会发送 `ROW_PTR_END` 和 `STREAM_END`
3. `block_data`：必须是二维列表，行主序
4. `block_col_idx`：对应变量列的索引（如 y1=0, x1=1, x2=2）

需要更具体的示例吗？

```python
from encoder_model import encode_blocks

# 准备你的块数据
blocks = [
    {
        "block_rows": 6,           # 块的行数
        "block_cols": 6,           # 块的列数
        "block_factor_id": 5,      # 因子ID（来自哪个因子）
        "block_col_idx": 0,       # 列索引（对应哪个变量，如y1=0, x1=1）
        "block_data": [            # 稠密矩阵数据（行主序）
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            # ... 更多行
        ],
        "new_block_row": True,     # 是否是新行的第一个块
        "last_block_in_row": False, # 是否是行尾块
    },
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 5,
        "block_col_idx": 1,        # 下一个变量列
        "block_data": [[...]],     # 下一个块的数据
        "new_block_row": False,    # 同一行的后续块
        "last_block_in_row": True, # 最后一个块（会自动发送STREAM_END）
    },
]

# 编码成 CSR 流
words = encode_blocks(blocks)
print(f"生成了 {len(words)} 个 CSR 字")
```

```python
from decoder_model import decode_stream

# 解码 CSR 流
result = decode_stream(words)

# 查看结果
print(f"解码了 {len(result.blocks)} 个块")
print(f"行指针: {result.row_ptrs}")
print(f"流结束: {result.stream_done}")

# 访问每个块
for i, blk in enumerate(result.blocks):
    print(f"块 {i}:")
    print(f"  尺寸: {blk.rows}×{blk.cols}")
    print(f"  因子ID: {blk.factor_id}")
    print(f"  列索引: {blk.col_idx}")
    print(f"  数据: {blk.values}")
```

```python
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 模拟你的 QR 分解结果：一行 g 包含多个块
blocks = [
    # 第一个块：y1 列的 R 块（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,      # 来自 factor f1
        "block_col_idx": 0,        # y1 的列索引
        "block_data": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [0.0, 0.0, 12.0, 13.0, 14.0, 15.0],
            [0.0, 0.0, 0.0, 16.0, 17.0, 18.0],
            [0.0, 0.0, 0.0, 0.0, 19.0, 20.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 21.0],
        ],
        "new_block_row": True,     # 这是新行的第一个块
        "last_block_in_row": False,
    },
    # 第二个块：x1 列的方阵（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 1,        # x1 的列索引
        "block_data": [
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0, 32.0, 33.0],
            # ... 更多行
        ],
        "new_block_row": False,
        "last_block_in_row": True, # 最后一个块
    },
]

# 编码
words = encode_blocks(blocks)
print(f"编码完成，共 {len(words)} 个 CSR 字")

# 解码验证
result = decode_stream(words)
print(f"\n解码完成，共 {len(result.blocks)} 个块")

# 验证数据是否正确
for i, (original, decoded) in enumerate(zip(blocks, result.blocks)):
    if original["block_data"] == decoded.values:
        print(f"✓ 块 {i} 数据匹配")
    else:
        print(f"✗ 块 {i} 数据不匹配")
```

```python
# 从 RTL 仿真日志中读取 CSR 字
def parse_rtl_output(log_file):
    words = []
    # 解析日志，构造 CSRWord 对象
    # ...
    return words

# 解码并验证
rtl_words = parse_rtl_output("simulation.log")
result = decode_stream(rtl_words)

# 与期望值对比
expected_blocks = [...]  # 你的期望值
assert len(result.blocks) == len(expected_blocks)
```

```python
# 生成测试用的块数据
def generate_test_blocks():
    blocks = []
    for col_idx in range(3):  # 3 个变量列
        blocks.append({
            "block_rows": 6,
            "block_cols": 6,
            "block_factor_id": 1,
            "block_col_idx": col_idx,
            "block_data": [[float(i*6+j) for j in range(6)] for i in range(6)],
            "new_block_row": (col_idx == 0),
            "last_block_in_row": (col_idx == 2),
        })
    return blocks

# 生成 CSR 流用于测试
test_blocks = generate_test_blocks()
test_words = encode_blocks(test_blocks)
```

```python
from encoder_model import encode_block

# 单独编码一个块（用于调试）
words, nnzb_after, next_row_idx = encode_block(
    block_rows=2,
    block_cols=2,
    block_factor_id=7,
    block_col_idx=3,
    block_data=[[1.0, 2.0], [3.0, 4.0]],
    new_block_row=True,
    last_block_in_row=True,
    nnzb_before=0,
    row_index=0,
)

print(f"编码后块数: {nnzb_after}")
print(f"下一个行索引: {next_row_idx}")
for word in words:
    print(f"  {word.kind:02x}: {word}")
```

---

**User**

我的意思是使用的代码是什么

---

**Cursor**


提供可直接运行的代码示例：


```python
"""Example usage of encoder_model and decoder_model."""

from encoder_model import encode_blocks, encode_block
from decoder_model import decode_stream, CSRWord

# ============================================================================
# 示例 1: 使用 encode_blocks 批量编码多个块
# ============================================================================
print("=" * 60)
print("示例 1: 批量编码")
print("=" * 60)

blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 0,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": False,
    },
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 7,
        "block_col_idx": 1,
        "block_data": [[5.0, 6.0], [7.0, 8.0]],
        "new_block_row": False,
        "last_block_in_row": True,
    },
]

words = encode_blocks(blocks)
print(f"编码了 {len(blocks)} 个块，生成 {len(words)} 个 CSR 字")
print(f"前 3 个字: {words[0].kind:02x}, {words[1].kind:02x}, {words[2].kind:02x}")

# ============================================================================
# 示例 2: 解码 CSR 流
# ============================================================================
print("\n" + "=" * 60)
print("示例 2: 解码 CSR 流")
print("=" * 60)

result = decode_stream(words)
print(f"解码得到 {len(result.blocks)} 个块")
print(f"行指针: {result.row_ptrs}")
print(f"流结束: {result.stream_done}")

for i, blk in enumerate(result.blocks):
    print(f"\n块 {i}:")
    print(f"  尺寸: {blk.rows}×{blk.cols}")
    print(f"  因子ID: {blk.factor_id}")
    print(f"  列索引: {blk.col_idx}")
    print(f"  数据: {blk.values}")

# ============================================================================
# 示例 3: 单独编码一个块（用于调试）
# ============================================================================
print("\n" + "=" * 60)
print("示例 3: 单独编码一个块")
print("=" * 60)

words_single, nnzb_after, next_row_idx = encode_block(
    block_rows=3,
    block_cols=3,
    block_factor_id=5,
    block_col_idx=2,
    block_data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    new_block_row=True,
    last_block_in_row=True,
    nnzb_before=0,
    row_index=0,
)

print(f"编码后块数: {nnzb_after}")
print(f"下一个行索引: {next_row_idx}")
print(f"生成了 {len(words_single)} 个 CSR 字")

# ============================================================================
# 示例 4: 验证编码-解码的往返一致性
# ============================================================================
print("\n" + "=" * 60)
print("示例 4: 验证往返一致性")
print("=" * 60)

# 原始数据
original_blocks = [
    {
        "block_rows": 2,
        "block_cols": 2,
        "block_factor_id": 10,
        "block_col_idx": 0,
        "block_data": [[1.0, 2.0], [3.0, 4.0]],
        "new_block_row": True,
        "last_block_in_row": True,
    },
]

# 编码
encoded_words = encode_blocks(original_blocks)

# 解码
decoded_result = decode_stream(encoded_words)

# 验证
if len(decoded_result.blocks) == len(original_blocks):
    print("✓ 块数量匹配")
    for i, (orig, dec) in enumerate(zip(original_blocks, decoded_result.blocks)):
        if orig["block_data"] == dec.values:
            print(f"✓ 块 {i} 数据完全匹配")
        else:
            print(f"✗ 块 {i} 数据不匹配")
            print(f"  原始: {orig['block_data']}")
            print(f"  解码: {dec.values}")
else:
    print(f"✗ 块数量不匹配: 原始={len(original_blocks)}, 解码={len(decoded_result.blocks)}")

# ============================================================================
# 示例 5: 实际场景 - 模拟 QR 分解的一行 g
# ============================================================================
print("\n" + "=" * 60)
print("示例 5: 模拟 QR 分解的一行 g")
print("=" * 60)

# 假设 g1 行包含：y1 列的 R 块 + x1 列的方阵 + x2 列的方阵
qr_blocks = [
    # y1 列的 R 块（6×6，上三角）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 0,  # y1
        "block_data": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [0.0, 0.0, 12.0, 13.0, 14.0, 15.0],
            [0.0, 0.0, 0.0, 16.0, 17.0, 18.0],
            [0.0, 0.0, 0.0, 0.0, 19.0, 20.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 21.0],
        ],
        "new_block_row": True,
        "last_block_in_row": False,
    },
    # x1 列的方阵（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 1,  # x1
        "block_data": [[float(i*6+j+22) for j in range(6)] for i in range(6)],
        "new_block_row": False,
        "last_block_in_row": False,
    },
    # x2 列的方阵（6×6）
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 2,  # x2
        "block_data": [[float(i*6+j+58) for j in range(6)] for i in range(6)],
        "new_block_row": False,
        "last_block_in_row": True,  # 最后一个块
    },
]

qr_words = encode_blocks(qr_blocks)
print(f"QR 行包含 {len(qr_blocks)} 个块，编码成 {len(qr_words)} 个 CSR 字")

qr_result = decode_stream(qr_words)
print(f"解码得到 {len(qr_result.blocks)} 个块")
print(f"行指针: {qr_result.row_ptrs}")
print(f"流结束: {qr_result.stream_done}")

```


运行示例代码：



已创建示例文件 `example_usage.py`。使用方式如下：

## 最简单的使用代码

### 编码（把块数据转成 CSR 流）

```python
from encoder_model import encode_blocks

blocks = [
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 0,
        "block_data": [[你的矩阵数据]],
        "new_block_row": True,
        "last_block_in_row": False,
    },
    # ... 更多块
]

words = encode_blocks(blocks)  # 得到 CSR 字列表
```

### 解码（把 CSR 流转回块数据）

```python
from decoder_model import decode_stream

result = decode_stream(words)

# 访问结果
for blk in result.blocks:
    print(blk.values)  # 块的数据
    print(blk.factor_id)  # 因子ID
    print(blk.col_idx)  # 列索引
```

## 运行示例

```bash
cd QR/RTL/py
python3 example_usage.py
```

示例文件包含 5 个场景：
1. 批量编码多个块
2. 解码 CSR 流
3. 单独编码一个块（调试用）
4. 验证编码-解码往返一致性
5. 模拟 QR 分解的实际场景

可直接运行查看效果，或参考示例代码修改。

```python
from encoder_model import encode_blocks

blocks = [
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 0,
        "block_data": [[你的矩阵数据]],
        "new_block_row": True,
        "last_block_in_row": False,
    },
    # ... 更多块
]

words = encode_blocks(blocks)  # 得到 CSR 字列表
```

```python
from decoder_model import decode_stream

result = decode_stream(words)

# 访问结果
for blk in result.blocks:
    print(blk.values)  # 块的数据
    print(blk.factor_id)  # 因子ID
    print(blk.col_idx)  # 列索引
```

```shellscript
cd QR/RTL/py
python3 example_usage.py
```

---

**User**

为什么给我写了一个新的py？你教我怎么用不就行了吗？例如我直接Python3 encoder之类的不行吗？

---

**Cursor**


删除示例文件，改为直接使用现有文件：



直接使用现有文件：

## 直接运行

```bash
cd QR/RTL/py

# 运行 encoder 的 demo
python3 encoder_model.py

# 运行 decoder 的 demo
python3 decoder_model.py
```

## 在 Python 代码中使用

### 方法 1：导入使用

```python
# 在你的 Python 脚本中
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 编码
blocks = [
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 0,
        "block_data": [[你的数据]],
        "new_block_row": True,
        "last_block_in_row": False,
    },
]
words = encode_blocks(blocks)

# 解码
result = decode_stream(words)
print(result.blocks[0].values)
```

### 方法 2：在 Python 交互式环境中

```bash
python3
```

```python
>>> from encoder_model import encode_blocks
>>> blocks = [{"block_rows": 2, "block_cols": 2, ...}]
>>> words = encode_blocks(blocks)
>>> from decoder_model import decode_stream
>>> result = decode_stream(words)
```

两个文件都有 `if __name__ == "__main__":`，可以直接运行查看 demo。需要修改 demo 中的测试数据，直接编辑这两个文件即可。

```shellscript
cd QR/RTL/py

# 运行 encoder 的 demo
python3 encoder_model.py

# 运行 decoder 的 demo
python3 decoder_model.py
```

```python
# 在你的 Python 脚本中
from encoder_model import encode_blocks
from decoder_model import decode_stream

# 编码
blocks = [
    {
        "block_rows": 6,
        "block_cols": 6,
        "block_factor_id": 1,
        "block_col_idx": 0,
        "block_data": [[你的数据]],
        "new_block_row": True,
        "last_block_in_row": False,
    },
]
words = encode_blocks(blocks)

# 解码
result = decode_stream(words)
print(result.blocks[0].values)
```

```shellscript
python3
```

```python
>>> from encoder_model import encode_blocks
>>> blocks = [{"block_rows": 2, "block_cols": 2, ...}]
>>> words = encode_blocks(blocks)
>>> from decoder_model import decode_stream
>>> result = decode_stream(words)
```

---

**User**

我不太理解@zsh (2-18) 为什么输出这样

---

**Cursor**


解释这个输出的含义：



解释输出含义：

## 输入数据

Demo 中有 2 个块：
- 块 1：2×2 矩阵 `[[1.0, 2.0], [3.0, 4.0]]`，`new_block_row=True`
- 块 2：2×2 矩阵 `[[5.0, 6.0], [7.0, 8.0]]`，`last_block_in_row=True`

## 输出解释

```
0: kind=04  → ROW_PTR_START (新行开始，row_field=0表示之前有0个块)
1: kind=01  → META (块1的元数据：rows=2, cols=2, factor_id=7)
2: kind=02  → COL_INDEX (块1的列索引：idx_field=3)
3-6: kind=03 → VALUE (块1的4个数值：1.0, 2.0, 3.0, 4.0)
7: kind=01  → META (块2的元数据：rows=2, cols=2, factor_id=7)
8: kind=02  → COL_INDEX (块2的列索引：idx_field=4)
9-12: kind=03 → VALUE (块2的4个数值：5.0, 6.0, 7.0, 8.0)
13: kind=04  → ROW_PTR_END (行结束，row_field=2表示现在有2个块)
14: kind=ff  → STREAM_END (流结束)
```

## 为什么这样输出？

编码流程：

1. 块 1（`new_block_row=True`）：
   - 先发 `ROW_PTR_START`（字 0）
   - 再发 `META`（字 1）
   - 再发 `COL_INDEX`（字 2）
   - 再发 4 个 `VALUE`（字 3-6）

2. 块 2（`last_block_in_row=True`）：
   - 发 `META`（字 7）
   - 发 `COL_INDEX`（字 8）
   - 发 4 个 `VALUE`（字 9-12）
   - 发 `ROW_PTR_END`（字 13）
   - 发 `STREAM_END`（字 14）

## 关键点

- `kind=04`：行指针（ROW_PTR），用于标记行的开始和结束
- `kind=01`：元数据（META），包含块的尺寸和因子ID
- `kind=02`：列索引（COL_INDEX），表示这个块对应哪个变量列
- `kind=03`：数值（VALUE），矩阵中的每个浮点数
- `kind=ff`：流结束（STREAM_END），标记整个流结束

这就是为什么有 15 个字：1个ROW_PTR_START + 2个META + 2个COL_INDEX + 8个VALUE + 1个ROW_PTR_END + 1个STREAM_END = 15。

需要我画个图更清楚地展示吗？

```plaintext
0: kind=04  → ROW_PTR_START (新行开始，row_field=0表示之前有0个块)
1: kind=01  → META (块1的元数据：rows=2, cols=2, factor_id=7)
2: kind=02  → COL_INDEX (块1的列索引：idx_field=3)
3-6: kind=03 → VALUE (块1的4个数值：1.0, 2.0, 3.0, 4.0)
7: kind=01  → META (块2的元数据：rows=2, cols=2, factor_id=7)
8: kind=02  → COL_INDEX (块2的列索引：idx_field=4)
9-12: kind=03 → VALUE (块2的4个数值：5.0, 6.0, 7.0, 8.0)
13: kind=04  → ROW_PTR_END (行结束，row_field=2表示现在有2个块)
14: kind=ff  → STREAM_END (流结束)
```

---

**User**

不太理解 输出数据按照数组的格式给我，理论上应该按照数组的格式存的吧？

---

**Cursor**


修改输出为数组格式，更清晰地展示数据结构：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words\n")
    
    # 按数组格式输出
    print("CSR Words Array:")
    print("[")
    for i, word in enumerate(words):
        kind_name = {
            0x01: "META",
            0x02: "COL_INDEX", 
            0x03: "VALUE",
            0x04: "ROW_PTR",
            0xFF: "STREAM_END"
        }.get(word.kind, f"UNKNOWN({word.kind:02x})")
        
        if word.kind == CSR_KIND_META:
            print(f"  [{i}] {kind_name:12} rows={word.rows}, cols={word.cols}, factor_id={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_COL_INDEX:
            print(f"  [{i}] {kind_name:12} col_idx={word.idx_field}")
        elif word.kind == CSR_KIND_VALUE:
            print(f"  [{i}] {kind_name:12} value={word.value_field}")
        elif word.kind == CSR_KIND_ROW_PTR:
            print(f"  [{i}] {kind_name:12} row_idx={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_STREAM_END:
            print(f"  [{i}] {kind_name:12} final_nnzb={word.row_field}")
    print("]")
    
    # 按类型分组统计
    print("\n按类型统计:")
    from collections import Counter
    kinds = [word.kind for word in words]
    kind_names = {0x01: "META", 0x02: "COL_INDEX", 0x03: "VALUE", 0x04: "ROW_PTR", 0xFF: "STREAM_END"}
    for kind, count in Counter(kinds).items():
        print(f"  {kind_names.get(kind, f'0x{kind:02x}')}: {count} 个")


if __name__ == "__main__":
    demo()

```

```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words\n")
    
    # 按数组格式输出
    print("CSR Words Array:")
    print("[")
    for i, word in enumerate(words):
        kind_name = {
            0x01: "META",
            0x02: "COL_INDEX", 
            0x03: "VALUE",
            0x04: "ROW_PTR",
            0xFF: "STREAM_END"
        }.get(word.kind, f"UNKNOWN({word.kind:02x})")
        
        if word.kind == CSR_KIND_META:
            print(f"  [{i}] {kind_name:12} rows={word.rows}, cols={word.cols}, factor_id={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_COL_INDEX:
            print(f"  [{i}] {kind_name:12} col_idx={word.idx_field}")
        elif word.kind == CSR_KIND_VALUE:
            print(f"  [{i}] {kind_name:12} value={word.value_field}")
        elif word.kind == CSR_KIND_ROW_PTR:
            print(f"  [{i}] {kind_name:12} row_idx={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_STREAM_END:
            print(f"  [{i}] {kind_name:12} final_nnzb={word.row_field}")
    print("]")
    
    # 按类型分组统计
    print("\n按类型统计:")
    from collections import Counter
    kinds = [word.kind for word in words]
    kind_names = {0x01: "META", 0x02: "COL_INDEX", 0x03: "VALUE", 0x04: "ROW_PTR", 0xFF: "STREAM_END"}
    for kind, count in Counter(kinds).items():
        print(f"  {kind_names.get(kind, f'0x{kind:02x}')}: {count} 个")


if __name__ == "__main__":
    demo()

```

```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    words = encode_blocks(blocks)
    print(f"Generated {len(words)} CSR words\n")
    
    # 按数组格式输出
    print("CSR Words Array:")
    print("[")
    for i, word in enumerate(words):
        kind_name = {
            0x01: "META",
            0x02: "COL_INDEX", 
            0x03: "VALUE",
            0x04: "ROW_PTR",
            0xFF: "STREAM_END"
        }.get(word.kind, f"UNKNOWN({word.kind:02x})")
        
        if word.kind == CSR_KIND_META:
            print(f"  [{i}] {kind_name:12} rows={word.rows}, cols={word.cols}, factor_id={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_COL_INDEX:
            print(f"  [{i}] {kind_name:12} col_idx={word.idx_field}")
        elif word.kind == CSR_KIND_VALUE:
            print(f"  [{i}] {kind_name:12} value={word.value_field}")
        elif word.kind == CSR_KIND_ROW_PTR:
            print(f"  [{i}] {kind_name:12} row_idx={word.idx_field}, nnzb={word.row_field}")
        elif word.kind == CSR_KIND_STREAM_END:
            print(f"  [{i}] {kind_name:12} final_nnzb={word.row_field}")
    print("]")
    
    # 按类型分组统计
    print("\n按类型统计:")
    kinds = [word.kind for word in words]
    kind_names = {0x01: "META", 0x02: "COL_INDEX", 0x03: "VALUE", 0x04: "ROW_PTR", 0xFF: "STREAM_END"}
    for kind, count in Counter(kinds).items():
        print(f"  {kind_names.get(kind, f'0x{kind:02x}')}: {count} 个")


if __name__ == "__main__":
    demo()

```


运行查看数组格式输出：



输出已改为数组格式。数据结构如下：

## 数据结构说明

`words` 是一个**数组**，包含 15 个 `CSRWord` 对象：

```
words = [
    CSRWord(kind=ROW_PTR, ...),    # [0] 行开始标记
    CSRWord(kind=META, ...),       # [1] 块1的元数据
    CSRWord(kind=COL_INDEX, ...),  # [2] 块1的列索引
    CSRWord(kind=VALUE, ...),      # [3] 块1的值1
    CSRWord(kind=VALUE, ...),      # [4] 块1的值2
    ...
]
```

## 关键点

1. 数组索引：`[0]`, `[1]`, `[2]`... 表示在数组中的位置
2. 类型：每个字都有 `kind` 字段标识类型
3. 顺序：按编码顺序排列（ROW_PTR → META → COL_INDEX → VALUE...）

## 实际使用

```python
words = encode_blocks(blocks)

# 访问数组元素
print(words[0].kind)      # 0x04 (ROW_PTR)
print(words[1].rows)      # 2 (META中的rows)
print(words[3].value_field)  # 1.0 (第一个VALUE)

# 遍历数组
for word in words:
    if word.kind == CSR_KIND_VALUE:
        print(word.value_field)
```

这样更清楚了吗？数据确实是按数组格式存储的。

```plaintext
words = [
    CSRWord(kind=ROW_PTR, ...),    # [0] 行开始标记
    CSRWord(kind=META, ...),       # [1] 块1的元数据
    CSRWord(kind=COL_INDEX, ...),  # [2] 块1的列索引
    CSRWord(kind=VALUE, ...),      # [3] 块1的值1
    CSRWord(kind=VALUE, ...),      # [4] 块1的值2
    ...
]
```

```python
words = encode_blocks(blocks)

# 访问数组元素
print(words[0].kind)      # 0x04 (ROW_PTR)
print(words[1].rows)      # 2 (META中的rows)
print(words[3].value_field)  # 1.0 (第一个VALUE)

# 遍历数组
for word in words:
    if word.kind == CSR_KIND_VALUE:
        print(word.value_field)
```

---

**User**

还是不懂，我的理解是输出分成四类输出的吧

row_ptr[] = [0, 3, 5, ...]

col_idx[] = [y1, x1, x2, y2, x3, ...]

block_info[] = [(rows=6,cols=6,factor=f1), (6,6,f1), (6,6,f1), (2,6,f3), (6,6,f3), ...]

values[]  = [...]



---

**Cursor**


展示解码后的四个数组格式（Total Buffer 的存储格式），而不是编码流。修改 demo 以输出这四个数组：



修改 demo，展示解码后的四个数组格式：


```python
"""Simple functional model for Block-CSR encoding.

This module provides a simple function to convert dense matrix blocks into
CSR stream words, matching the format produced by `QR/RTL/encoder.sv`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class CSRWord:
    """A single CSR stream word."""
    kind: int = 0
    rows: int = 0
    cols: int = 0
    row_field: int = 0
    idx_field: int = 0
    value_field: float = 0.0


# CSR word kinds (matches RTL)
CSR_KIND_META = 0x01
CSR_KIND_COL_INDEX = 0x02
CSR_KIND_VALUE = 0x03
CSR_KIND_ROW_PTR = 0x04
CSR_KIND_STREAM_END = 0xFF


def encode_block(
    block_rows: int,
    block_cols: int,
    block_factor_id: int,
    block_col_idx: int,
    block_data: List[List[float]],
    new_block_row: bool = False,
    last_block_in_row: bool = False,
    nnzb_before: int = 0,
    row_index: int = 0,
) -> tuple[List[CSRWord], int, int]:
    """Encode a single dense block into CSR words.

    Args:
        block_rows: Number of rows in the block
        block_cols: Number of columns in the block
        block_factor_id: Identifier of the originating factor
        block_col_idx: Column index of this block
        block_data: Dense matrix data (list of lists, row-major)
        new_block_row: True if this is the first block in a new row
        last_block_in_row: True if this is the last block in the current row (and entire stream)
        nnzb_before: Number of blocks emitted before this one
        row_index: Current row index (for ROW_PTR)

    Returns:
        Tuple of (csr_words, nnzb_after, next_row_index):
        - csr_words: List of CSRWord objects for this block
        - nnzb_after: Number of blocks after encoding this one
        - next_row_index: Next row index (incremented if last_block_in_row)
    """
    words = []
    nnzb_after = nnzb_before + 1
    next_row_index = row_index

    # Emit ROW_PTR at start of new row
    if new_block_row:
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_before,
                idx_field=row_index,
            )
        )

    # Emit META word
    words.append(
        CSRWord(
            kind=CSR_KIND_META,
            rows=block_rows,
            cols=block_cols,
            row_field=nnzb_before,
            idx_field=block_factor_id,
        )
    )

    # Emit COL_INDEX word
    words.append(
        CSRWord(
            kind=CSR_KIND_COL_INDEX,
            idx_field=block_col_idx,
        )
    )

    # Emit VALUE words (row-major order)
    if block_rows > 0 and block_cols > 0:
        for r in range(block_rows):
            for c in range(block_cols):
                words.append(
                    CSRWord(
                        kind=CSR_KIND_VALUE,
                        value_field=block_data[r][c] if r < len(block_data) and c < len(block_data[r]) else 0.0,
                    )
                )

    # Emit ROW_PTR at end of row
    if last_block_in_row:
        next_row_index = row_index + 1
        words.append(
            CSRWord(
                kind=CSR_KIND_ROW_PTR,
                row_field=nnzb_after,
                idx_field=next_row_index,
            )
        )
        # Emit STREAM_END when last block in row (which is also last block overall)
        words.append(
            CSRWord(
                kind=CSR_KIND_STREAM_END,
                row_field=nnzb_after,
            )
        )

    return words, nnzb_after, next_row_index


def encode_blocks(blocks: List[dict]) -> List[CSRWord]:
    """Encode a sequence of blocks into a CSR word stream.

    Args:
        blocks: List of block dictionaries, each with keys:
            - block_rows, block_cols, block_factor_id, block_col_idx
            - block_data (list of lists)
            - new_block_row, last_block_in_row

    Returns:
        List of CSRWord objects representing the entire stream.
    """
    words = []
    nnzb = 0
    row_idx = 0

    for blk in blocks:
        block_words, nnzb, row_idx = encode_block(
            block_rows=blk["block_rows"],
            block_cols=blk["block_cols"],
            block_factor_id=blk["block_factor_id"],
            block_col_idx=blk["block_col_idx"],
            block_data=blk["block_data"],
            new_block_row=blk.get("new_block_row", False),
            last_block_in_row=blk.get("last_block_in_row", False),
            nnzb_before=nnzb,
            row_index=row_idx,
        )
        words.extend(block_words)

    return words


def demo():
    """Example usage."""
    from decoder_model import decode_stream
    
    blocks = [
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 3,
            "block_data": [[1.0, 2.0], [3.0, 4.0]],
            "new_block_row": True,
            "last_block_in_row": False,
        },
        {
            "block_rows": 2,
            "block_cols": 2,
            "block_factor_id": 7,
            "block_col_idx": 4,
            "block_data": [[5.0, 6.0], [7.0, 8.0]],
            "new_block_row": False,
            "last_block_in_row": True,
        },
    ]

    # 编码
    words = encode_blocks(blocks)
    print(f"编码完成，生成 {len(words)} 个 CSR 字\n")
    
    # 解码
    result = decode_stream(words)
    
    # 构建四个数组（Total Buffer 存储格式）
    row_ptr = []
    col_idx = []
    block_info = []
    values = []
    
    # 从 row_ptrs 构建 row_ptr 数组
    for row_idx, nnzb_prefix in result.row_ptrs:
        row_ptr.append(nnzb_prefix)
    
    # 从 blocks 构建其他数组
    for blk in result.blocks:
        col_idx.append(blk.col_idx)
        block_info.append((blk.rows, blk.cols, blk.factor_id))
        # 将二维数组展平成一维（行主序）
        for row in blk.values:
            values.extend(row)
    
    # 输出四个数组
    print("Total Buffer 存储格式（四个数组）:")
    print("=" * 60)
    print(f"row_ptr[]   = {row_ptr}")
    print(f"col_idx[]   = {col_idx}")
    print(f"block_info[] = {block_info}")
    print(f"values[]    = {values}")
    print("=" * 60)


if __name__ == "__main__":
    demo()

```


运行查看四个数组的输出：



输出已改为四个数组格式。解释如下：

## 四个数组的含义

```
row_ptr[]   = [0, 2]
```
- 行指针数组：`row_ptr[0]=0` 表示第 0 行从第 0 个块开始，`row_ptr[1]=2` 表示第 1 行从第 2 个块开始（即第 0 行有 2 个块）

```
col_idx[]   = [3, 4]
```
- 列索引数组：第 0 个块对应列索引 3，第 1 个块对应列索引 4

```
block_info[] = [(2, 2, 7), (2, 2, 7)]
```
- 块信息数组：每个元素是 `(rows, cols, factor_id)`
  - 第 0 个块：2×2，因子ID=7
  - 第 1 个块：2×2，因子ID=7

```
values[]    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```
- 数值数组：所有块的数据按行主序展平
  - 前 4 个值 `[1.0, 2.0, 3.0, 4.0]` 是第 0 个块
  - 后 4 个值 `[5.0, 6.0, 7.0, 8.0]` 是第 1 个块

这就是 Total Buffer 中存储的格式。

```plaintext
row_ptr[]   = [0, 2]
```

```plaintext
col_idx[]   = [3, 4]
```

```plaintext
block_info[] = [(2, 2, 7), (2, 2, 7)]
```

```plaintext
values[]    = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

---

**User**

为什么需要decoder_model？我的encoder的输出不就应该是这样吗？

---

**Cursor**

需要区分两件事：

1. **Encoder 的真实输出**  
   - 硬件的 `encoder.sv` 会按照 CSR 流的协议，把一个块拆成一连串“包”：  
     `ROW_PTR → META → COL_INDEX → VALUE → ... → ROW_PTR → STREAM_END`  
   - 这是为了满足硬件接口（逐拍输出、固定宽度），所以输出的是一个“字流”（stream）。

2. **你在 Total Buffer 里希望存的东西**  
   - `row_ptr[] / col_idx[] / block_info[] / values[]` 这四个数组，是 Total Buffer 的最终存储结构。  
   - 为了得到这四个数组，需要把字流重新整理、分类、还原块信息。

`decoder_model.py` 的作用，就是扮演 Total Buffer 的“接收端”：  
它把 encoder 流出来的字流解析成四个数组，方便验证 WBS、调试、或与 RTL 行为比对。

换句话说：  
- 如果你只想验证数学结果、看四个数组，那你可以直接调用 `encoder_model.encode_blocks(blocks)` 得到字流，再手动解析。  
- 我为了方便你查看结果，写了 `decode_stream(words)`，可以快速把字流变回你熟悉的四个数组，帮助确认流程无误。

在硬件中也是如此：  
- encoder 负责打包成流  
- Total Buffer 或其它模块负责解析流，写入 `row_ptr` / `col_idx` / `block_info` / `values` 四个存储器

因此，`decoder_model` 不是硬件里必须的模块，只是一个**软件工具**，用于验证和理解 encoder 的输出，确保字流内容正确。

---

**User**

不是的，“包”应该是encoder的中间结果吧（理解不一定准确，要是不正确你帮我指正），这是为了扁平化，我的encoder的输出就应该是Total Buffer 里希望存的东西。decoder的作用是从Total Buffer 里取东西时，复原成类似@encoder_model.py (167-173) 的矩阵的样子之类的。我的RTL做的不是这个功能吗？

---

**Cursor**

你抓到了“包=扁平化”的核心，但结论里少了一个步骤：**Total Buffer** 才是负责从“包”构建 `row_ptr[] / col_idx[] / block_info[] / values[]` 的模块，`encoder` 只是把块“扁平化成包”。`decoder` 的功能恰好与 `encoder` 相反，它把包还原成矩阵，用在后端解码或验证。

这条硬件链路其实是：

```
Global Controller → encoder（包流）→ Total Buffer（写四个数组）→
从 Total Buffer 读出 → decoder（包流 → 矩阵）→ 回代/后处理
```

也就是说：

1. **encoder.sv 输出的是真正的基金（包/stream）**
   - 这是它对外的协议；它不能直接写多个 RAM，因为：  
     - 接口宽度有限，不能一次吐出上千位  
     - 流控简单，后端可以 backpressure  
     - 粘合逻辑清晰

2. **Total Buffer 负责把包落盘成四个数组**
   - 它读取 `encoder` 的 stream  
   - 针对不同 kind（ROW_PTR/META/COL_INDEX/VALUE）写到对应的存储器  
   - 最终得到你想要的四个数组

3. **decoder（无论 RTL 的 `decoder.sv` 还是 Python 的 `decode_stream`）都是从这四个数组取数据时，重新拼成矩阵**
   - 在硬件里，这一步通常发生在回代单元或 TB 中  
   - 在软件里，我们用它做验证/调试

所以，Python 模型里先“encode”（得到包），再 `decode_stream` 出四个数组只是为了展示 **Total Buffer 的存储效果**（把中间那一层做掉），方便你直观对比。

总结：

- `encoder` 对外给流  
- Total Buffer 把流转成四个数组  
- `decoder`（或别的消费端）再把数组读出来还原成矩阵

你的理解思路没偏，只是把“Total Buffer 做聚合”这一环忽略了。

```plaintext
Global Controller → encoder（包流）→ Total Buffer（写四个数组）→
从 Total Buffer 读出 → decoder（包流 → 矩阵）→ 回代/后处理
```

