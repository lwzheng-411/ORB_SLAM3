`timescale 1ns/1ps

module encoder_tb;
    // ---------------------------------------------------------------------
    // Parameters matching encoder defaults
    // ---------------------------------------------------------------------
    localparam int MAX_ROWS     = 6;
    localparam int MAX_COLS     = 6;
    localparam int DATA_WIDTH   = 32;
    localparam int IDX_WIDTH    = 16;
    localparam int ROWPTR_WIDTH = 16;

    // DUT signals
    logic clk;
    logic rst;

    logic                 block_valid;
    logic                 block_ready;
    logic [2:0]           block_rows;
    logic [2:0]           block_cols;
    logic [IDX_WIDTH-1:0] block_factor_id;
    logic [IDX_WIDTH-1:0] block_col_idx;
    logic                 new_block_row;
    logic                 last_block_in_row;
    logic                 last_block_overall;
    logic [DATA_WIDTH-1:0] block_data [0:MAX_ROWS-1][0:MAX_COLS-1];

    logic                 csr_valid;
    logic                 csr_ready;
    logic                 csr_last;
    encoder::csr_word_t   csr_word;

    // Capture queue
    encoder::csr_word_t captured[$];
    bit got_last;

    // Clock generation
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // DUT instantiation
    encoder dut (
        .clk               (clk),
        .rst               (rst),
        .block_valid       (block_valid),
        .block_ready       (block_ready),
        .block_rows        (block_rows),
        .block_cols        (block_cols),
        .block_factor_id   (block_factor_id),
        .block_col_idx     (block_col_idx),
        .new_block_row     (new_block_row),
        .last_block_in_row (last_block_in_row),
        .last_block_overall(last_block_overall),
        .block_data        (block_data),
        .csr_valid         (csr_valid),
        .csr_ready         (csr_ready),
        .csr_last          (csr_last),
        .csr_word          (csr_word)
    );

    // Monitor CSR stream
    always_ff @(posedge clk) begin
        if (!rst && csr_valid && csr_ready) begin
            captured.push_back(csr_word);
            if (csr_last) got_last <= 1'b1;
        end
    end

    // Helper: clear block data
    task automatic init_block_data6x6();
        integer r, c;
        begin
            for (r = 0; r < MAX_ROWS; r++) begin
                for (c = 0; c < MAX_COLS; c++) begin
                    block_data[r][c] = r*MAX_COLS + c + 32'd1;
                end
            end
        end
    endtask

    // Apply a single block to the encoder
    task automatic send_block();
        begin
        @(posedge clk);
        block_rows          <= 3'd6;
        block_cols          <= 3'd6;
            block_factor_id     <= 16'd7;
            block_col_idx       <= 16'd3;
            new_block_row       <= 1'b1;
            last_block_in_row   <= 1'b1;
            last_block_overall  <= 1'b1;
            block_valid         <= 1'b1;

            // Hold until accepted
            while (!block_ready) @(posedge clk);

            @(posedge clk);
            block_valid         <= 1'b0;
            new_block_row       <= 1'b0;
            last_block_in_row   <= 1'b0;
            last_block_overall  <= 1'b0;
        end
    endtask

    // Comparison helper
    task automatic check_word(string tag, encoder::csr_word_t act, encoder::csr_word_t exp);
        begin
            if (act.kind        !== exp.kind        ||
                act.rows        !== exp.rows        ||
                act.cols        !== exp.cols        ||
                act.row_field   !== exp.row_field   ||
                act.idx_field   !== exp.idx_field   ||
                act.value_field !== exp.value_field) begin
                $display("[%0t] ERROR: %s mismatch", $time, tag);
                $display("  Expected kind=%0h rows=%0d cols=%0d row_field=%0d idx_field=%0d value=%0d",
                         exp.kind, exp.rows, exp.cols, exp.row_field, exp.idx_field, exp.value_field);
                $display("  Got      kind=%0h rows=%0d cols=%0d row_field=%0d idx_field=%0d value=%0d",
                         act.kind, act.rows, act.cols, act.row_field, act.idx_field, act.value_field);
                $fatal(1);
            end
        end
    endtask

    // Test sequence
    initial begin
        // Default assignments
        csr_ready          = 1'b1;
        block_valid        = 1'b0;
        block_rows         = '0;
        block_cols         = '0;
        block_factor_id    = '0;
        block_col_idx      = '0;
        new_block_row      = 1'b0;
        last_block_in_row  = 1'b0;
        last_block_overall = 1'b0;
        got_last           = 1'b0;

        init_block_data6x6();
        $display("Input 6x6 matrix:");
        for (int r = 0; r < 6; r++) begin
            $display("  [%0d %0d %0d %0d %0d %0d]",
                     block_data[r][0], block_data[r][1], block_data[r][2],
                     block_data[r][3], block_data[r][4], block_data[r][5]);
        end

        // Reset sequence
        rst = 1'b1;
        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // Send block
        send_block();

        // Wait for stream end
        wait (got_last);
        repeat (2) @(posedge clk);

        // Expected words
        encoder::csr_word_t expected [$];
        encoder::csr_word_t word_template = '{default:'0};

        word_template.kind      = 8'h04;
        word_template.row_field = 0;
        word_template.idx_field = 0;
        expected.push_back(word_template);

        word_template = '{default:'0};
        word_template.kind      = 8'h01;
        word_template.rows      = 3'd6;
        word_template.cols      = 3'd6;
        word_template.row_field = 0;
        word_template.idx_field = 16'd7;
        expected.push_back(word_template);

        word_template = '{default:'0};
        word_template.kind      = 8'h02;
        word_template.idx_field = 16'd3;
        expected.push_back(word_template);

        word_template = '{default:'0};
        word_template.kind = 8'h03;
        for (int v = 0; v < 36; v++) begin
            word_template.value_field = v + 32'd1;
            expected.push_back(word_template);
        end

        word_template = '{default:'0};
        word_template.kind      = 8'h04;
        word_template.row_field = 1;
        word_template.idx_field = 1;
        expected.push_back(word_template);

        word_template = '{default:'0};
        word_template.kind      = 8'hFF;
        word_template.row_field = 1;
        expected.push_back(word_template);

        if (captured.size() != expected.size()) begin
            $display("[%0t] ERROR: Expected %0d words, got %0d", $time, expected.size(), captured.size());
            $fatal(1);
        end

        for (int i = 0; i < expected.size(); i++) begin
            check_word($sformatf("word[%0d]", i), captured[i], expected[i]);
        end

        $display("encoder_tb PASS");
        $finish;
    end

endmodule

