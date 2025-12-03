# Encoder 直接写 Buffer - 需要确定的事项清单

## 当前状态

- ✅ Encoder 已经实现（输出 CSR 包流）
- ❌ 需要改成直接写 Buffer
- ❓ Buffer 的接口和结构待确定

---

## 需要确定的核心事项

### 1. Buffer 的 RAM 结构

**问题：用方案 A（多 RAM）还是方案 B（单 RAM）？**

#### 选项 A：多个独立 RAM
```systemverilog
// Encoder 需要输出 4 组 RAM 接口
output [ADDR_WIDTH-1:0]  row_ptr_addr,
output [DATA_WIDTH-1:0]  row_ptr_data,
output                   row_ptr_we,

output [ADDR_WIDTH-1:0]  col_idx_addr,
output [DATA_WIDTH-1:0]  col_idx_data,
output                   col_idx_we,

output [ADDR_WIDTH-1:0]  block_info_addr,
output [BLOCK_INFO_WIDTH-1:0] block_info_data,
output                   block_info_we,

output [ADDR_WIDTH-1:0]  values_addr,
output [DATA_WIDTH-1:0]  values_data,  // 或 [127:0] 如果要用宽端口
output                   values_we
```

#### 选项 B：单一 RAM，分区域
```systemverilog
// Encoder 只需要输出 1 组 RAM 接口
output [ADDR_WIDTH-1:0]  ram_addr,
output [DATA_WIDTH-1:0]  ram_data,
output                   ram_we

// 但需要知道基址常量
parameter ROW_PTR_BASE   = 16'h0000,
parameter COL_IDX_BASE   = 16'h0100,
parameter BLOCK_INFO_BASE= 16'h0200,
parameter VALUES_BASE     = 16'h0300
```

**需要确定：**
- [ ] 选择方案 A 还是 B？
- [ ] 如果方案 A，每个 RAM 的位宽是多少？
  - `row_ptr_ram`: 32-bit?
  - `col_idx_ram`: 32-bit?
  - `block_info_ram`: 16-bit? 32-bit?
  - `values_ram`: 32-bit? 128-bit?（宽端口提高效率）

---

### 2. 写地址管理

**问题：谁管理写指针？Encoder 还是 Buffer？**

#### 选项 1：Encoder 管理（推荐）
```systemverilog
// Encoder 内部维护计数器
reg [ADDR_WIDTH-1:0] row_ptr_wr_idx;
reg [ADDR_WIDTH-1:0] block_idx;      // 用于 col_idx 和 block_info
reg [ADDR_WIDTH-1:0] values_wr_idx;

// Encoder 直接输出地址
assign row_ptr_addr = row_ptr_wr_idx;
assign col_idx_addr = block_idx;
assign block_info_addr = block_idx;
assign values_addr = values_wr_idx;
```

**优点：**
- Encoder 完全控制写入顺序
- 逻辑简单，不需要与 Buffer 同步

**缺点：**
- Encoder 需要知道 Buffer 的地址空间大小
- 如果 Buffer 满了，需要流控机制

#### 选项 2：Buffer 管理
```systemverilog
// Encoder 输出写请求
output logic row_ptr_wr_req;
output logic col_idx_wr_req;
output logic block_info_wr_req;
output logic values_wr_req;

// Buffer 返回写地址
input [ADDR_WIDTH-1:0] row_ptr_wr_addr;
input [ADDR_WIDTH-1:0] col_idx_wr_addr;
input [ADDR_WIDTH-1:0] block_info_wr_addr;
input [ADDR_WIDTH-1:0] values_wr_addr;
```

**优点：**
- Buffer 可以控制地址分配
- 支持动态地址管理

**缺点：**
- 接口更复杂
- 需要握手协议

**需要确定：**
- [ ] 选择选项 1（Encoder 管理）还是选项 2（Buffer 管理）？
- [ ] 如果选项 1，每个 RAM 的地址宽度是多少？
  - `row_ptr`: 需要支持多少行？（如 8-bit = 256 行）
  - `col_idx/block_info`: 需要支持多少块？（如 10-bit = 1024 块）
  - `values`: 需要支持多少值？（如 16-bit = 65536 个浮点数）

---

### 3. block_info 的格式

**问题：block_info 需要存哪些字段？**

```systemverilog
// 选项 1：最小化（只存尺寸）
typedef struct packed {
    logic [2:0] rows;  // 最多 6 行
    logic [2:0] cols;  // 最多 6 列
} block_info_t;  // 6 bits

// 选项 2：包含 factor_id
typedef struct packed {
    logic [2:0] rows;
    logic [2:0] cols;
    logic [IDX_WIDTH-1:0] factor_id;  // 16 bits
} block_info_t;  // 22 bits

// 选项 3：完整信息
typedef struct packed {
    logic [2:0] rows;
    logic [2:0] cols;
    logic [IDX_WIDTH-1:0] factor_id;
    logic [7:0] flags;  // 如 is_upper_tri 等
} block_info_t;  // 30 bits
```

**需要确定：**
- [ ] block_info 需要哪些字段？
  - [ ] rows, cols（必需）
  - [ ] factor_id（可选，调试用）
  - [ ] flags（可选，如 is_upper_tri）
- [ ] block_info 的位宽是多少？
  - 如果 6 bits，可以用 8-bit RAM（浪费 2 bits）
  - 如果 22 bits，可以用 32-bit RAM（浪费 10 bits）
  - 如果 30 bits，必须用 32-bit RAM

---

### 4. 流控机制

**问题：如果 Buffer 暂时写满，Encoder 怎么办？**

#### 选项 1：简单流控（推荐）
```systemverilog
// Buffer 输出"可写"信号
input logic row_ptr_ready;
input logic col_idx_ready;
input logic block_info_ready;
input logic values_ready;

// Encoder 只在所有 RAM 都 ready 时才写
assign block_ready = row_ptr_ready && col_idx_ready && 
                     block_info_ready && values_ready;
```

#### 选项 2：复杂流控
```systemverilog
// Buffer 输出每个 RAM 的写满状态
input logic row_ptr_full;
input logic col_idx_full;
input logic block_info_full;
input logic values_full;

// Encoder 需要处理部分写满的情况
// 可能需要暂停某些写入，继续其他写入
```

**需要确定：**
- [ ] 选择简单流控还是复杂流控？
- [ ] Buffer 如何检测写满？
  - 固定大小（地址计数器达到最大值）
  - 动态大小（由外部配置）

---

### 5. 写入时序

**问题：多个 RAM 可以同时写入吗？**

#### 选项 1：串行写入（简单）
```systemverilog
// 每个时钟周期只写一个 RAM
// 状态机控制写入顺序：
// 1. 写 row_ptr（如果新行）
// 2. 写 block_info
// 3. 写 col_idx
// 4. 写 values（逐个）
```

#### 选项 2：并行写入（高效）
```systemverilog
// 某些 RAM 可以同时写入
// 例如：block_info 和 col_idx 可以同时写（地址相同）
always_comb begin
    if (state == WRITE_BLOCK_INFO) begin
        block_info_we = 1'b1;
        col_idx_we = 1'b1;  // 同时写
        block_info_addr = block_idx;
        col_idx_addr = block_idx;
    end
end
```

**需要确定：**
- [ ] 选择串行写入还是并行写入？
- [ ] 如果可以并行，哪些 RAM 可以同时写？

---

### 6. 初始化与复位

**问题：写指针如何初始化？**

```systemverilog
// 选项 1：从 0 开始（每次复位清零）
always_ff @(posedge clk) begin
    if (rst) begin
        row_ptr_wr_idx <= '0;
        block_idx <= '0;
        values_wr_idx <= '0;
    end
end

// 选项 2：支持基址偏移（多个 Encoder 写同一个 Buffer）
parameter ROW_PTR_BASE = 0;
parameter BLOCK_BASE = 0;
parameter VALUES_BASE = 0;
```

**需要确定：**
- [ ] 写指针从 0 开始，还是支持基址偏移？
- [ ] 如果支持基址偏移，如何配置？（参数？输入信号？）

---

### 7. 边界检查

**问题：如何防止地址溢出？**

```systemverilog
// 选项 1：Encoder 检查
always_comb begin
    if (row_ptr_wr_idx >= MAX_ROWS) begin
        row_ptr_ready = 1'b0;  // 停止写入
    end
end

// 选项 2：Buffer 检查
// Buffer 检测到地址溢出时，拉低 ready 信号
```

**需要确定：**
- [ ] 谁负责边界检查？Encoder 还是 Buffer？
- [ ] 溢出时如何处理？（停止写入？报错？）

---

## 推荐配置（快速开始）

如果还不确定，建议先用这个配置：

### 方案 A：多 RAM，Encoder 管理地址，简单流控

```systemverilog
// Buffer 接口
output [7:0]   row_ptr_addr,      // 8-bit = 256 行
output [31:0]  row_ptr_data,
output         row_ptr_we,

output [9:0]   col_idx_addr,      // 10-bit = 1024 块
output [31:0]  col_idx_data,
output         col_idx_we,

output [9:0]   block_info_addr,   // 10-bit = 1024 块
output [31:0]  block_info_data,   // 32-bit，包含 rows(3) + cols(3) + factor_id(16) + 保留(10)
output         block_info_we,

output [15:0]  values_addr,       // 16-bit = 65536 个值
output [31:0]  values_data,       // 32-bit 浮点数
output         values_we,

// 流控
input          buffer_ready       // 所有 RAM 都 ready
```

### block_info 格式
```systemverilog
typedef struct packed {
    logic [2:0] rows;              // [2:0]
    logic [2:0] cols;              // [5:3]
    logic [IDX_WIDTH-1:0] factor_id; // [21:6]
    logic [9:0] reserved;          // [31:22]
} block_info_t;
```

---

## 下一步行动

1. **确定 RAM 结构**（方案 A 或 B）
2. **确定 block_info 格式**（需要哪些字段）
3. **确定地址宽度**（每个 RAM 需要多大）
4. **确定流控机制**（简单还是复杂）
5. **修改 encoder.sv**（移除包流，添加 RAM 接口）
6. **实现 buffer.sv**（如果还没有）

需要我帮你实现某个部分吗？

