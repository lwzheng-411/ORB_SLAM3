// -----------------------------------------------------------------------------
// unpacker.sv
// -----------------------------------------------------------------------------
// 接收 encoder.sv 输出的 Block-CSR 包流（csr_word_t），
// 解析后写入 Total Buffer 的四个数组：
//   - row_ptr[]
//   - col_idx[]
//   - block_info[]
//   - values[]
//
// 解析逻辑完全封装在此模块，便于保持 encoder.sv 输出协议不变。
// -----------------------------------------------------------------------------

module unpacker #(
    parameter int DATA_WIDTH          = 32,
    parameter int IDX_WIDTH           = 16,
    parameter int ROWPTR_WIDTH        = 16,
    parameter int ROW_PTR_ADDR_WIDTH  = 8,   // 2^8 = 256 row_ptr entries
    parameter int BLOCK_ADDR_WIDTH    = 10,  // 2^10 = 1024 blocks
    parameter int VALUES_ADDR_WIDTH   = 16   // 2^16 = 65536 values
) (
    input  logic clk,
    input  logic rst,

    // CSR 输入流（来自 encoder.sv）
    input  logic      csr_valid,
    output logic      csr_ready,
    input  logic      csr_last,
    input  csr_word_t csr_word,

    // 写 Total Buffer 的四个数组（与 total_buffer.sv 匹配）
    output logic [ROW_PTR_ADDR_WIDTH-1:0] row_ptr_wr_addr,
    output logic [DATA_WIDTH-1:0]         row_ptr_wr_data,
    output logic                          row_ptr_wr_en,

    output logic [BLOCK_ADDR_WIDTH-1:0]   col_idx_wr_addr,
    output logic [DATA_WIDTH-1:0]         col_idx_wr_data,
    output logic                          col_idx_wr_en,

    output logic [BLOCK_ADDR_WIDTH-1:0]   block_info_wr_addr,
    output logic [DATA_WIDTH-1:0]         block_info_wr_data,
    output logic                          block_info_wr_en,

    output logic [VALUES_ADDR_WIDTH-1:0]  values_wr_addr,
    output logic [DATA_WIDTH-1:0]         values_wr_data,
    output logic                          values_wr_en,

    // 流结束脉冲（可选）
    output logic stream_done
);

    // ---------------------------------------------------------------------
    // CSR stream word definition (must match encoder.sv)
    // ---------------------------------------------------------------------
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
        ST_IDLE,
        ST_WAIT_COL,
        ST_VALUES
    } state_e;

    state_e state_q, state_d;

    // 写地址指针
    logic [ROW_PTR_ADDR_WIDTH-1:0] row_ptr_wr_ptr_q, row_ptr_wr_ptr_d;
    logic [BLOCK_ADDR_WIDTH-1:0]   block_wr_ptr_q,   block_wr_ptr_d;
    logic [VALUES_ADDR_WIDTH-1:0]  values_wr_ptr_q,  values_wr_ptr_d;

    // META 缓存
    logic [2:0]            pending_rows_q, pending_rows_d;
    logic [2:0]            pending_cols_q, pending_cols_d;
    logic [IDX_WIDTH-1:0]  pending_factor_id_q, pending_factor_id_d;
    logic [5:0]            values_remaining_q, values_remaining_d; // up to 36

    // 行指针写数据缓存（无需额外状态）
    logic stream_done_q, stream_done_d;

    // ---------------------------------------------------------------------
    // 默认值
    // ---------------------------------------------------------------------
    always_comb begin
        state_d             = state_q;

        csr_ready           = 1'b0;
        stream_done_d       = 1'b0;

        row_ptr_wr_en       = 1'b0;
        row_ptr_wr_addr     = row_ptr_wr_ptr_q;
        row_ptr_wr_data     = '0;

        col_idx_wr_en       = 1'b0;
        col_idx_wr_addr     = block_wr_ptr_q;
        col_idx_wr_data     = '0;

        block_info_wr_en    = 1'b0;
        block_info_wr_addr  = block_wr_ptr_q;
        block_info_wr_data  = '0;

        values_wr_en        = 1'b0;
        values_wr_addr      = values_wr_ptr_q;
        values_wr_data      = '0;

        row_ptr_wr_ptr_d    = row_ptr_wr_ptr_q;
        block_wr_ptr_d      = block_wr_ptr_q;
        values_wr_ptr_d     = values_wr_ptr_q;

        pending_rows_d      = pending_rows_q;
        pending_cols_d      = pending_cols_q;
        pending_factor_id_d = pending_factor_id_q;
        values_remaining_d  = values_remaining_q;

        // -----------------------------------------------------------------
        case (state_q)
            ST_IDLE: begin
                if (!csr_valid) begin
                    csr_ready = 1'b1;
                end else begin
                    unique case (csr_word.kind)
                        CSR_KIND_ROW_PTR: begin
                            csr_ready       = 1'b1;
                            row_ptr_wr_en   = 1'b1;
                            row_ptr_wr_addr = row_ptr_wr_ptr_q;
                            row_ptr_wr_data = {{(DATA_WIDTH-ROWPTR_WIDTH){1'b0}}, csr_word.row_field};
                            row_ptr_wr_ptr_d= row_ptr_wr_ptr_q + 1'b1;
                        end

                        CSR_KIND_META: begin
                            csr_ready            = 1'b1;
                            pending_rows_d       = csr_word.rows;
                            pending_cols_d       = csr_word.cols;
                            pending_factor_id_d  = csr_word.idx_field;
                            values_remaining_d   = csr_word.rows * csr_word.cols;
                            state_d              = ST_WAIT_COL;
                        end

                        CSR_KIND_STREAM_END: begin
                            csr_ready      = 1'b1;
                            stream_done_d  = 1'b1;
                        end

                        default: begin
                            csr_ready = 1'b0; // unexpected word type
                        end
                    endcase
                end
            end

            ST_WAIT_COL: begin
                if (!csr_valid) begin
                    csr_ready = 1'b1;
                end else if (csr_word.kind == CSR_KIND_COL_INDEX) begin
                    csr_ready          = 1'b1;

                    // 写 col_idx
                    col_idx_wr_en      = 1'b1;
                    col_idx_wr_addr    = block_wr_ptr_q;
                    col_idx_wr_data    = {{(DATA_WIDTH-IDX_WIDTH){1'b0}}, csr_word.idx_field};

                    // 写 block_info：{rows, cols, factor_id}
                    block_info_wr_en   = 1'b1;
                    block_info_wr_addr = block_wr_ptr_q;
                    block_info_wr_data = pack_block_info(pending_rows_q,
                                                         pending_cols_q,
                                                         pending_factor_id_q);

                    block_wr_ptr_d     = block_wr_ptr_q + 1'b1;

                    if (values_remaining_q == 0) begin
                        state_d = ST_IDLE;
                    end else begin
                        state_d = ST_VALUES;
                    end
                end else begin
                    csr_ready = 1'b0; // 等待 COL_INDEX
                end
            end

            ST_VALUES: begin
                if (!csr_valid) begin
                    csr_ready = 1'b1;
                end else if (csr_word.kind == CSR_KIND_VALUE) begin
                    csr_ready          = 1'b1;
                    values_wr_en       = 1'b1;
                    values_wr_addr     = values_wr_ptr_q;
                    values_wr_data     = csr_word.value_field;
                    values_wr_ptr_d    = values_wr_ptr_q + 1'b1;

                    if (values_remaining_q > 0) begin
                        values_remaining_d = values_remaining_q - 1'b1;
                    end

                    if (values_remaining_q <= 1) begin
                        state_d = ST_IDLE;
                    end
                end else begin
                    csr_ready = 1'b0; // 只接受 VALUE
                end
            end

            default: begin
                state_d = ST_IDLE;
            end
        endcase
    end

    // ---------------------------------------------------------------------
    // Sequential logic
    // ---------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state_q             <= ST_IDLE;
            row_ptr_wr_ptr_q    <= '0;
            block_wr_ptr_q      <= '0;
            values_wr_ptr_q     <= '0;
            pending_rows_q      <= '0;
            pending_cols_q      <= '0;
            pending_factor_id_q <= '0;
            values_remaining_q  <= '0;
            stream_done_q       <= 1'b0;
        end else begin
            state_q             <= state_d;
            row_ptr_wr_ptr_q    <= row_ptr_wr_ptr_d;
            block_wr_ptr_q      <= block_wr_ptr_d;
            values_wr_ptr_q     <= values_wr_ptr_d;
            pending_rows_q      <= pending_rows_d;
            pending_cols_q      <= pending_cols_d;
            pending_factor_id_q <= pending_factor_id_d;
            values_remaining_q  <= values_remaining_d;
            stream_done_q       <= stream_done_d;
        end
    end

    assign stream_done = stream_done_q;

    // ---------------------------------------------------------------------
    // Helper: 打包 block_info
    // block_info_data 格式：{reserved[9:0], factor_id[15:0], cols[2:0], rows[2:0]}
    // ---------------------------------------------------------------------
    function automatic logic [DATA_WIDTH-1:0] pack_block_info (
        input logic [2:0]           rows,
        input logic [2:0]           cols,
        input logic [IDX_WIDTH-1:0] factor_id
    );
        logic [DATA_WIDTH-1:0] payload;
        payload               = '0;
        payload[2:0]          = rows;
        payload[5:3]          = cols;
        payload[21:6]         = factor_id;
        return payload;
    endfunction

endmodule

