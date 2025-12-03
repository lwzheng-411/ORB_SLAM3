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
