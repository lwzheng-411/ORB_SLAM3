// -----------------------------------------------------------------------------
// total_buffer.sv
// -----------------------------------------------------------------------------
// Total Buffer - 存储 Block CSR 数据的四个数组
// 
// 使用 4 个独立的 RAM（BRAM）存储：
// 1. row_ptr[]   - 行指针数组
// 2. col_idx[]   - 列索引数组
// 3. block_info[] - 块元数据数组
// 4. values[]    - 数值数组
//
// 每个 RAM 都是同步读写（1 个时钟周期延迟）
// -----------------------------------------------------------------------------

module total_buffer #(
    parameter int ROW_PTR_ADDR_WIDTH  = 8,   // 2^8 = 256 行
    parameter int BLOCK_ADDR_WIDTH    = 10,  // 2^10 = 1024 块
    parameter int VALUES_ADDR_WIDTH   = 16,  // 2^16 = 65536 个值
    parameter int DATA_WIDTH          = 32,  // 32 位数据
    parameter int IDX_WIDTH          = 16   // 索引宽度
) (
    input  logic clk,
    input  logic rst,
    
    // ====================================================================
    // 写端口（来自 Encoder）
    // ====================================================================
    
    // row_ptr RAM 写端口
    input  logic [ROW_PTR_ADDR_WIDTH-1:0]  row_ptr_wr_addr,
    input  logic [DATA_WIDTH-1:0]          row_ptr_wr_data,
    input  logic                           row_ptr_wr_en,
    
    // col_idx RAM 写端口
    input  logic [BLOCK_ADDR_WIDTH-1:0]    col_idx_wr_addr,
    input  logic [DATA_WIDTH-1:0]          col_idx_wr_data,
    input  logic                           col_idx_wr_en,
    
    // block_info RAM 写端口
    input  logic [BLOCK_ADDR_WIDTH-1:0]    block_info_wr_addr,
    input  logic [DATA_WIDTH-1:0]          block_info_wr_data,
    input  logic                           block_info_wr_en,
    
    // values RAM 写端口
    input  logic [VALUES_ADDR_WIDTH-1:0]   values_wr_addr,
    input  logic [DATA_WIDTH-1:0]          values_wr_data,
    input  logic                           values_wr_en,
    
    // ====================================================================
    // 读端口（给 Decoder 或其他模块）
    // ====================================================================
    
    // row_ptr RAM 读端口
    input  logic [ROW_PTR_ADDR_WIDTH-1:0]  row_ptr_rd_addr,
    output logic [DATA_WIDTH-1:0]          row_ptr_rd_data,
    
    // col_idx RAM 读端口
    input  logic [BLOCK_ADDR_WIDTH-1:0]    col_idx_rd_addr,
    output logic [DATA_WIDTH-1:0]          col_idx_rd_data,
    
    // block_info RAM 读端口
    input  logic [BLOCK_ADDR_WIDTH-1:0]    block_info_rd_addr,
    output logic [DATA_WIDTH-1:0]         block_info_rd_data,
    
    // values RAM 读端口
    input  logic [VALUES_ADDR_WIDTH-1:0]   values_rd_addr,
    output logic [DATA_WIDTH-1:0]         values_rd_data,
    
    // ====================================================================
    // 状态输出（流控）
    // ====================================================================
    output logic buffer_ready  // 所有 RAM 都 ready（简化版：总是 ready）
);

    // ====================================================================
    // 1. row_ptr RAM
    // 存储：行指针数组，每个元素是 32 位整数
    // 大小：256 个元素（8-bit 地址）
    // ====================================================================
    logic [DATA_WIDTH-1:0] row_ptr_mem [0:(1<<ROW_PTR_ADDR_WIDTH)-1];
    
    always_ff @(posedge clk) begin
        // 写操作
        if (row_ptr_wr_en) begin
            row_ptr_mem[row_ptr_wr_addr] <= row_ptr_wr_data;
        end
        // 读操作（同步读，延迟 1 个时钟周期）
        row_ptr_rd_data <= row_ptr_mem[row_ptr_rd_addr];
    end
    
    // ====================================================================
    // 2. col_idx RAM
    // 存储：列索引数组，每个元素是 32 位整数（实际只用低 16 位）
    // 大小：1024 个元素（10-bit 地址）
    // ====================================================================
    logic [DATA_WIDTH-1:0] col_idx_mem [0:(1<<BLOCK_ADDR_WIDTH)-1];
    
    always_ff @(posedge clk) begin
        if (col_idx_wr_en) begin
            col_idx_mem[col_idx_wr_addr] <= col_idx_wr_data;
        end
        col_idx_rd_data <= col_idx_mem[col_idx_rd_addr];
    end
    
    // ====================================================================
    // 3. block_info RAM
    // 存储：块元数据，格式：{rows[2:0], cols[2:0], factor_id[15:0], reserved[9:0]}
    // 大小：1024 个元素（10-bit 地址）
    // ====================================================================
    logic [DATA_WIDTH-1:0] block_info_mem [0:(1<<BLOCK_ADDR_WIDTH)-1];
    
    always_ff @(posedge clk) begin
        if (block_info_wr_en) begin
            block_info_mem[block_info_wr_addr] <= block_info_wr_data;
        end
        block_info_rd_data <= block_info_mem[block_info_rd_addr];
    end
    
    // ====================================================================
    // 4. values RAM
    // 存储：所有块的数值数据（行主序展平）
    // 大小：65536 个元素（16-bit 地址）
    // ====================================================================
    logic [DATA_WIDTH-1:0] values_mem [0:(1<<VALUES_ADDR_WIDTH)-1];
    
    always_ff @(posedge clk) begin
        if (values_wr_en) begin
            values_mem[values_wr_addr] <= values_wr_data;
        end
        values_rd_data <= values_mem[values_rd_addr];
    end
    
    // ====================================================================
    // 流控（简化版：总是 ready）
    // 如果需要检查写满，可以添加地址边界检查
    // ====================================================================
    assign buffer_ready = 1'b1;
    
    // 可选：边界检查版本
    // assign buffer_ready = (row_ptr_wr_addr < (1<<ROW_PTR_ADDR_WIDTH)) && 
    //                       (col_idx_wr_addr < (1<<BLOCK_ADDR_WIDTH)) && 
    //                       (block_info_wr_addr < (1<<BLOCK_ADDR_WIDTH)) &&
    //                       (values_wr_addr < (1<<VALUES_ADDR_WIDTH));

endmodule

