`timescale 1ns/1ps

module unpacker_tb;
    localparam int DATA_WIDTH          = 32;
    localparam int IDX_WIDTH           = 16;
    localparam int ROWPTR_WIDTH        = 16;
    localparam int ROW_PTR_ADDR_WIDTH  = 8;
    localparam int BLOCK_ADDR_WIDTH    = 10;
    localparam int VALUES_ADDR_WIDTH   = 16;

    logic clk;
    logic rst;

    // CSR stream signals
    logic                 csr_valid;
    logic                 csr_ready;
    logic                 csr_last;
    unpacker::csr_word_t  csr_word;

    // Connections between unpacker and total_buffer
    logic [ROW_PTR_ADDR_WIDTH-1:0] row_ptr_wr_addr;
    logic [DATA_WIDTH-1:0]         row_ptr_wr_data;
    logic                          row_ptr_wr_en;

    logic [BLOCK_ADDR_WIDTH-1:0]   col_idx_wr_addr;
    logic [DATA_WIDTH-1:0]         col_idx_wr_data;
    logic                          col_idx_wr_en;

    logic [BLOCK_ADDR_WIDTH-1:0]   block_info_wr_addr;
    logic [DATA_WIDTH-1:0]         block_info_wr_data;
    logic                          block_info_wr_en;

    logic [VALUES_ADDR_WIDTH-1:0]  values_wr_addr;
    logic [DATA_WIDTH-1:0]         values_wr_data;
    logic                          values_wr_en;

    logic stream_done;

    // Total buffer read ports
    logic [ROW_PTR_ADDR_WIDTH-1:0] row_ptr_rd_addr;
    logic [DATA_WIDTH-1:0]         row_ptr_rd_data;

    logic [BLOCK_ADDR_WIDTH-1:0]   col_idx_rd_addr;
    logic [DATA_WIDTH-1:0]         col_idx_rd_data;

    logic [BLOCK_ADDR_WIDTH-1:0]   block_info_rd_addr;
    logic [DATA_WIDTH-1:0]         block_info_rd_data;

    logic [VALUES_ADDR_WIDTH-1:0]  values_rd_addr;
    logic [DATA_WIDTH-1:0]         values_rd_data;

    logic buffer_ready;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    unpacker dut_unpacker (
        .clk                 (clk),
        .rst                 (rst),
        .csr_valid           (csr_valid),
        .csr_ready           (csr_ready),
        .csr_last            (csr_last),
        .csr_word            (csr_word),
        .row_ptr_wr_addr     (row_ptr_wr_addr),
        .row_ptr_wr_data     (row_ptr_wr_data),
        .row_ptr_wr_en       (row_ptr_wr_en),
        .col_idx_wr_addr     (col_idx_wr_addr),
        .col_idx_wr_data     (col_idx_wr_data),
        .col_idx_wr_en       (col_idx_wr_en),
        .block_info_wr_addr  (block_info_wr_addr),
        .block_info_wr_data  (block_info_wr_data),
        .block_info_wr_en    (block_info_wr_en),
        .values_wr_addr      (values_wr_addr),
        .values_wr_data      (values_wr_data),
        .values_wr_en        (values_wr_en),
        .stream_done         (stream_done)
    );

    total_buffer #(
        .ROW_PTR_ADDR_WIDTH (ROW_PTR_ADDR_WIDTH),
        .BLOCK_ADDR_WIDTH   (BLOCK_ADDR_WIDTH),
        .VALUES_ADDR_WIDTH  (VALUES_ADDR_WIDTH),
        .DATA_WIDTH         (DATA_WIDTH),
        .IDX_WIDTH          (IDX_WIDTH)
    ) dut_buffer (
        .clk                 (clk),
        .rst                 (rst),
        .row_ptr_wr_addr     (row_ptr_wr_addr),
        .row_ptr_wr_data     (row_ptr_wr_data),
        .row_ptr_wr_en       (row_ptr_wr_en),
        .col_idx_wr_addr     (col_idx_wr_addr),
        .col_idx_wr_data     (col_idx_wr_data),
        .col_idx_wr_en       (col_idx_wr_en),
        .block_info_wr_addr  (block_info_wr_addr),
        .block_info_wr_data  (block_info_wr_data),
        .block_info_wr_en    (block_info_wr_en),
        .values_wr_addr      (values_wr_addr),
        .values_wr_data      (values_wr_data),
        .values_wr_en        (values_wr_en),
        .row_ptr_rd_addr     (row_ptr_rd_addr),
        .row_ptr_rd_data     (row_ptr_rd_data),
        .col_idx_rd_addr     (col_idx_rd_addr),
        .col_idx_rd_data     (col_idx_rd_data),
        .block_info_rd_addr  (block_info_rd_addr),
        .block_info_rd_data  (block_info_rd_data),
        .values_rd_addr      (values_rd_addr),
        .values_rd_data      (values_rd_data),
        .buffer_ready        (buffer_ready)
    );

    // Task to send CSR word into unpacker
    task automatic send_word(unpacker::csr_word_t word, bit last_flag);
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

    // Read helper tasks (synchronous read => capture next cycle)
    task automatic read_row_ptr(input int addr, output logic [DATA_WIDTH-1:0] value);
        begin
            row_ptr_rd_addr <= addr[ROW_PTR_ADDR_WIDTH-1:0];
            @(posedge clk);
            value = row_ptr_rd_data;
        end
    endtask

    task automatic read_col_idx(input int addr, output logic [DATA_WIDTH-1:0] value);
        begin
            col_idx_rd_addr <= addr[BLOCK_ADDR_WIDTH-1:0];
            @(posedge clk);
            value = col_idx_rd_data;
        end
    endtask

    task automatic read_block_info(input int addr, output logic [DATA_WIDTH-1:0] value);
        begin
            block_info_rd_addr <= addr[BLOCK_ADDR_WIDTH-1:0];
            @(posedge clk);
            value = block_info_rd_data;
        end
    endtask

    task automatic read_value(input int addr, output logic [DATA_WIDTH-1:0] value);
        begin
            values_rd_addr <= addr[VALUES_ADDR_WIDTH-1:0];
            @(posedge clk);
            value = values_rd_data;
        end
    endtask

    initial begin
        csr_valid        = 1'b0;
        csr_last         = 1'b0;
        row_ptr_rd_addr  = '0;
        col_idx_rd_addr  = '0;
        block_info_rd_addr = '0;
        values_rd_addr   = '0;

        rst = 1'b1;
        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        unpacker::csr_word_t words[$];
        unpacker::csr_word_t w = '{default:'0};

        w.kind      = 8'h04;
        w.row_field = 0;
        w.idx_field = 0;
        words.push_back(w);

        w = '{default:'0};
        w.kind      = 8'h01;
        w.rows      = 3'd6;
        w.cols      = 3'd6;
        w.row_field = 0;
        w.idx_field = 16'd7;
        words.push_back(w);

        w = '{default:'0};
        w.kind      = 8'h02;
        w.idx_field = 16'd3;
        words.push_back(w);

        w = '{default:'0};
        w.kind = 8'h03;
        for (int v = 0; v < 36; v++) begin
            w.value_field = v + 32'd1;
            words.push_back(w);
        end

        w = '{default:'0};
        w.kind      = 8'h04;
        w.row_field = 1;
        w.idx_field = 1;
        words.push_back(w);

        w = '{default:'0};
        w.kind      = 8'hFF;
        w.row_field = 1;
        words.push_back(w);

        for (int i = 0; i < words.size(); i++) begin
            bit last_flag = (i == words.size()-1);
            send_word(words[i], last_flag);
        end

        wait (stream_done);
        repeat (2) @(posedge clk);

        logic [DATA_WIDTH-1:0] value;

        read_row_ptr(0, value);
        if (value !== 32'd0) begin
            $display("ERROR: row_ptr[0] expected 0, got %0d", value);
            $fatal(1);
        end

        read_row_ptr(1, value);
        if (value !== 32'd1) begin
            $display("ERROR: row_ptr[1] expected 1, got %0d", value);
            $fatal(1);
        end

        read_col_idx(0, value);
        if (value !== 32'd3) begin
            $display("ERROR: col_idx[0] expected 3, got %0d", value);
            $fatal(1);
        end

        read_block_info(0, value);
        if (value !== 32'h000001F6) begin
            $display("ERROR: block_info[0] expected 0x1F6, got 0x%0h", value);
            $fatal(1);
        end

        for (int idx = 0; idx < 36; idx++) begin
            read_value(idx, value);
            if (value !== idx + 32'd1) begin
                $display("ERROR: values[%0d] expected %0d, got %0d", idx, idx+1, value);
                $fatal(1);
            end
        end

        $display("unpacker_tb PASS");
        $finish;
    end

endmodule

