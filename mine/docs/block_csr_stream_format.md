# Block-CSR Stream Format (encoder.sv / decoder.sv)

本文件记录 `encoder.sv` 与 `decoder.sv` 所使用的流式协议，便于后续模块（Total Buffer、BackSub 等）对接。

## 数据处理背景

- QR/Update 阵列会输出一个 **稠密的 block**，其行/列尺寸由 `factor_type` 决定（可为 `6×6`、`2×6`、`2×3` 等）。
- 每个 block 对应全局稀疏矩阵的若干连续行，以及若干离散的列。
- `encoder.sv` 负责将该 block 编码成 **Block CSR** 形式并写入 Total Buffer；
- `decoder.sv` 相反，将流恢复成稠密 block，供回代或其他模块使用；
- 编码流中同时携带 `row_ptr` 信息，使得 Total Buffer 能够构建 Block CSR 的行指针数组。

## 流式字格式

所有数据通过 `csr_word` (packed struct) 传输：

```systemverilog
typedef struct packed {
    logic [7:0]                 kind;       // 数据类型
    logic [2:0]                 rows;       // block 行数（仅 META 有效）
    logic [2:0]                 cols;       // block 列数（仅 META 有效）
    logic [ROWPTR_WIDTH-1:0]    row_field;  // 行相关字段（row_ptr 或 row_base）
    logic [IDX_WIDTH-1:0]       idx_field;  // 列索引或 block-row id
    logic [DATA_WIDTH-1:0]      value_field;// 浮点数值（仅 VALUE 有效）
} csr_word_t;
```

`kind` 取值说明：

| kind | 含义 | 其它字段含义 |
|------|------|--------------|
| `8'h01` (`CSR_KIND_META`) | Block 头部 | `rows/cols` = block 尺寸；`row_field` = 全局行基址 `row_base`; `idx_field` = 编码前的 block 序号前缀 `nnzb_prefix` |
| `8'h02` (`CSR_KIND_COL_INDEX`) | 列索引 | `idx_field` = 全局列号 |
| `8'h03` (`CSR_KIND_VALUE`) | 稠密值 | `value_field` = 对应元素（row-major） |
| `8'h04` (`CSR_KIND_ROW_PTR`) | Row_ptr 条目 | `idx_field` = block-row ID；`row_field` = 当前 block 数量前缀（即 row_ptr 值） |
| `8'hFF` (`CSR_KIND_STREAM_END`) | 全部结束 | `row_field` = 总 block 数快照；`csr_last=1` |

注意：`rows/cols/row_field/idx_field/value_field` 在不同 kind 下的语义不同，未用字段均置零。

## Encoder 接口速览

```systemverilog
module encoder #(
    parameter MAX_ROWS = 6,
    parameter MAX_COLS = 6,
    parameter DATA_WIDTH = 32,
    parameter IDX_WIDTH = 16,
    parameter ROWPTR_WIDTH = 16
)(
    input  logic clk, rst,
    input  logic block_valid,
    output logic block_ready,
    input  logic [2:0] block_rows,
    input  logic [2:0] block_cols,
    input  logic [IDX_WIDTH-1:0] row_base,
    input  logic [ROWPTR_WIDTH-1:0] row_block_id,
    input  logic [ROWPTR_WIDTH-1:0] next_row_block_id,
    input  logic new_block_row,
    input  logic last_block_in_row,
    input  logic last_block_overall,
    input  logic [IDX_WIDTH-1:0] col_indices [MAX_COLS-1:0],
    input  logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],
    output logic csr_valid,
    input  logic csr_ready,
    output logic csr_last,
    output csr_word_t csr_word
);
```

- `block_valid/block_ready`：一次提交一个 block；
- `row_block_id` / `next_row_block_id`：block-row 的起止 ID；
- `new_block_row`：块是该 block-row 的首块 → 发 `ROW_PTR` 起点；
- `last_block_in_row`：块是该 block-row 的尾块 → 发 `ROW_PTR` 终点；
- `last_block_overall`：整个矩阵最后一个 block → 在尾部追加 `STREAM_END`；
- 输出流可直接写入 Total Buffer（若需要地址方式，可改写为 FIFO）。

## Decoder 接口速览

```systemverilog
module decoder #(
    parameter MAX_ROWS = 6,
    parameter MAX_COLS = 6,
    parameter DATA_WIDTH = 32,
    parameter IDX_WIDTH = 16,
    parameter ROWPTR_WIDTH = 16
)(
    input  logic clk, rst,
    input  logic csr_valid,
    output logic csr_ready,
    input  logic csr_last,
    input  csr_word_t csr_word,
    output logic row_ptr_valid,
    input  logic row_ptr_ready,
    output logic [ROWPTR_WIDTH-1:0] row_ptr_value,
    output logic [ROWPTR_WIDTH-1:0] row_ptr_row_id,
    output logic block_valid,
    input  logic block_ready,
    output logic [2:0] block_rows,
    output logic [2:0] block_cols,
    output logic [IDX_WIDTH-1:0] row_base,
    output logic [ROWPTR_WIDTH-1:0] nnzb_prefix,
    output logic [IDX_WIDTH-1:0] col_indices [MAX_COLS-1:0],
    output logic [DATA_WIDTH-1:0] block_data [MAX_ROWS-1:0][MAX_COLS-1:0],
    output logic stream_done
);
```

- `row_ptr_valid/ready`：直接转发 encoder 发出的 row_ptr 条目；
- `block_valid/ready`：当一个 block 聚合完成后输出；
- `stream_done`：看到 `STREAM_END` 后拉高一个周期；
- 输出的 `col_indices`、`block_data` 与 encoder 输入保持一致，方便后续验证或解码；
- `nnzb_prefix` 与 `row_ptr` 结合可重建 CSR 数组。

## 典型时序

以下顺序演示一条 block-row 内含两个 block 的情形：

1. 首块 (`new_block_row=1`)：
   - encoder 先发 `ROW_PTR(idx=row_block_id, value=nnzb_before)`；
   - 发 `META`、若干 `COL_INDEX`、`VALUE`；
2. 同一 block-row 的第二块：
   - 不再发送起始 `ROW_PTR`；
   - 仍然发 `META` → `COL_INDEX` → `VALUE`；
   - 因为 `last_block_in_row=1`，块尾额外发送 `ROW_PTR(idx=next_row_block_id, value=nnzb_after)`；
3. 所有 block 完成且 `last_block_overall=1` → 追加 `STREAM_END`（`csr_last=1`）。

## 集成注意事项

- `row_block_id` 与 `next_row_block_id` 的定义由上游决定（例如：block-row 顺序编号，从 0 开始递增）。
- `nnzb_count` 在硬件中按 block 粒度计数，可直接用作 CSR row_ptr 值（表示 “此 block-row 前一共出现了多少个 block”）。
- 当 block 尺寸小于最大尺寸时，未使用的 `col_indices`、`block_data` 需由上游清零，以便仿真/验证。
- 若 Total Buffer 采用 RAM 接口，可在 encoder 输出端加一层简单的 adapter，将 `csr_word` 写入 FIFO 或 BRAM。

## 与其它模块的协同

- `update.sv` / `b.sv` 等模块产出的 block 与列索引，需要由 Global Controller 收集后传给 encoder。
- decoder 输出的 block 可直接送入回代/输出 buffer，也可作单元测试的 golden 数据。
- Row pointer 输出既可直接写入独立存储，也可在软件侧与 CSR 数组聚合。

如需扩展到更多 block 尺寸或其它字段（例如 block 类型 ID），只需扩展 `META` 中的 `rows/cols` 或增加保留位即可。
