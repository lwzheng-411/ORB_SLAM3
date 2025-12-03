`timescale 1ns/1ps

module decoder_tb;
    localparam int MAX_ROWS     = 6;
    localparam int MAX_COLS     = 6;
    localparam int DATA_WIDTH   = 32;
    localparam int IDX_WIDTH    = 16;
    localparam int ROWPTR_WIDTH = 16;

    logic clk;
    logic rst;

    logic                csr_valid;
    logic                csr_ready;
    logic                csr_last;
    decoder::csr_word_t  csr_word;

    logic                     row_ptr_valid;
    logic                     row_ptr_ready;
    logic [ROWPTR_WIDTH-1:0]  row_ptr_value;
    logic [ROWPTR_WIDTH-1:0]  row_ptr_row_id;

    logic                     block_valid;
    logic                     block_ready;
    logic [2:0]               block_rows;
    logic [2:0]               block_cols;
    logic [ROWPTR_WIDTH-1:0]  nnzb_prefix;
    logic [IDX_WIDTH-1:0]     block_factor_id;
    logic [IDX_WIDTH-1:0]     block_col_idx;
    logic [DATA_WIDTH-1:0]    block_data [0:MAX_ROWS-1][0:MAX_COLS-1];

    logic stream_done;

    typedef struct packed {
        logic [ROWPTR_WIDTH-1:0] row_id;
        logic [ROWPTR_WIDTH-1:0] value;
    } row_ptr_entry_t;

    row_ptr_entry_t row_ptr_entries[$];
    bit             block_seen;

    logic [DATA_WIDTH-1:0] captured_block [0:MAX_ROWS-1][0:MAX_COLS-1];

    initial clk = 1'b0;
    always #5 clk = ~clk;

    decoder dut (
        .clk             (clk),
        .rst             (rst),
        .csr_valid       (csr_valid),
        .csr_ready       (csr_ready),
        .csr_last        (csr_last),
        .csr_word        (csr_word),
        .row_ptr_valid   (row_ptr_valid),
        .row_ptr_ready   (row_ptr_ready),
        .row_ptr_value   (row_ptr_value),
        .row_ptr_row_id  (row_ptr_row_id),
        .block_valid     (block_valid),
        .block_ready     (block_ready),
        .block_rows      (block_rows),
        .block_cols      (block_cols),
        .nnzb_prefix     (nnzb_prefix),
        .block_factor_id (block_factor_id),
        .block_col_idx   (block_col_idx),
        .block_data      (block_data),
        .stream_done     (stream_done)
    );

    // Capture row_ptr outputs
    always_ff @(posedge clk) begin
        if (!rst && row_ptr_valid && row_ptr_ready) begin
            row_ptr_entries.push_back('{row_id: row_ptr_row_id, value: row_ptr_value});
        end
    end

    // Capture block when ready
    always_ff @(posedge clk) begin
        if (!rst && block_valid && block_ready) begin
            block_seen <= 1'b1;
            for (int r = 0; r < MAX_ROWS; r++) begin
                for (int c = 0; c < MAX_COLS; c++) begin
                    captured_block[r][c] <= block_data[r][c];
                end
            end
        end
    end

    // Task to send a CSR word
    task automatic send_word(decoder::csr_word_t word, bit last_flag);
        begin
            csr_word  <= word;
            csr_last  <= last_flag;
            csr_valid <= 1'b1;
            do @(posedge clk); while (!csr_ready);
            csr_valid <= 1'b0;
            csr_last  <= 1'b0;
            @(posedge clk);
        end
    endtask

    // Helper to zero captured block
    task automatic clear_captured_block();
        begin
            for (int r = 0; r < MAX_ROWS; r++) begin
                for (int c = 0; c < MAX_COLS; c++) begin
                    captured_block[r][c] = '0;
                end
            end
        end
    endtask

    initial begin
        csr_valid      = 1'b0;
        csr_last       = 1'b0;
        row_ptr_ready  = 1'b1;
        block_ready    = 1'b1;
        block_seen     = 1'b0;
        clear_captured_block();

        rst = 1'b1;
        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // Prepare stimulus words
        decoder::csr_word_t words[$];
        decoder::csr_word_t wtemp = '{default:'0};

        wtemp.kind      = 8'h04;
        wtemp.row_field = 0;
        wtemp.idx_field = 0;
        words.push_back(wtemp);

        wtemp = '{default:'0};
        wtemp.kind      = 8'h01;
        wtemp.rows      = 3'd6;
        wtemp.cols      = 3'd6;
        wtemp.row_field = 0;
        wtemp.idx_field = 16'd7;
        words.push_back(wtemp);

        wtemp = '{default:'0};
        wtemp.kind      = 8'h02;
        wtemp.idx_field = 16'd3;
        words.push_back(wtemp);

        wtemp = '{default:'0};
        wtemp.kind = 8'h03;
        for (int v = 0; v < 36; v++) begin
            wtemp.value_field = v + 32'd1;
            words.push_back(wtemp);
        end

        wtemp = '{default:'0};
        wtemp.kind      = 8'h04;
        wtemp.row_field = 1;
        wtemp.idx_field = 1;
        words.push_back(wtemp);

        wtemp = '{default:'0};
        wtemp.kind      = 8'hFF;
        wtemp.row_field = 1;
        words.push_back(wtemp);

        for (int i = 0; i < words.size(); i++) begin
            bit last_flag = (i == words.size()-1);
            send_word(words[i], last_flag);
        end

        wait (stream_done);
        repeat (2) @(posedge clk);

        if (row_ptr_entries.size() != 2) begin
            $display("[%0t] ERROR: row_ptr entries expected 2 got %0d", $time, row_ptr_entries.size());
            $fatal(1);
        end

        if (row_ptr_entries[0].row_id !== 0 || row_ptr_entries[0].value !== 0) begin
            $display("[%0t] ERROR: row_ptr[0] mismatch (id=%0d val=%0d)", $time,
                     row_ptr_entries[0].row_id, row_ptr_entries[0].value);
            $fatal(1);
        end

        if (row_ptr_entries[1].row_id !== 1 || row_ptr_entries[1].value !== 1) begin
            $display("[%0t] ERROR: row_ptr[1] mismatch (id=%0d val=%0d)", $time,
                     row_ptr_entries[1].row_id, row_ptr_entries[1].value);
            $fatal(1);
        end

        if (!block_seen) begin
            $display("[%0t] ERROR: block output not seen", $time);
            $fatal(1);
        end

        if (block_rows !== 3'd6 || block_cols !== 3'd6) begin
            $display("[%0t] ERROR: block dimensions mismatch rows=%0d cols=%0d", $time, block_rows, block_cols);
            $fatal(1);
        end

        if (nnzb_prefix !== 0 || block_factor_id !== 16'd7 || block_col_idx !== 16'd3) begin
            $display("[%0t] ERROR: block metadata mismatch nnzb=%0d factor=%0d col_idx=%0d",
                     $time, nnzb_prefix, block_factor_id, block_col_idx);
            $fatal(1);
        end

        for (int r = 0; r < 6; r++) begin
            for (int c = 0; c < 6; c++) begin
                int expected_val = r*6 + c + 1;
                if (captured_block[r][c] !== expected_val) begin
                    $display("[%0t] ERROR: block data mismatch at (%0d,%0d) got %0d expected %0d",
                             $time, r, c, captured_block[r][c], expected_val);
                    $fatal(1);
                end
            end
        end

        $display("decoder_tb PASS");
        $finish;
    end

endmodule

