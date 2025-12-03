`timescale 1ns/1ps

module total_buffer_tb;
    localparam int ROW_PTR_ADDR_WIDTH = 4;
    localparam int BLOCK_ADDR_WIDTH   = 4;
    localparam int VALUES_ADDR_WIDTH  = 6;
    localparam int DATA_WIDTH         = 32;
    localparam int IDX_WIDTH          = 16;

    logic clk;
    logic rst;

    // Write ports
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

    // Read ports
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

    total_buffer #(
        .ROW_PTR_ADDR_WIDTH (ROW_PTR_ADDR_WIDTH),
        .BLOCK_ADDR_WIDTH   (BLOCK_ADDR_WIDTH),
        .VALUES_ADDR_WIDTH  (VALUES_ADDR_WIDTH),
        .DATA_WIDTH         (DATA_WIDTH),
        .IDX_WIDTH          (IDX_WIDTH)
    ) dut (
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

    // Simple write helper (single-cycle write)
    task automatic write_row_ptr(input int addr, input int data);
        begin
            @(posedge clk);
            row_ptr_wr_addr <= addr[ROW_PTR_ADDR_WIDTH-1:0];
            row_ptr_wr_data <= data;
            row_ptr_wr_en   <= 1'b1;
            @(posedge clk);
            row_ptr_wr_en   <= 1'b0;
        end
    endtask

    task automatic write_col_idx(input int addr, input int data);
        begin
            @(posedge clk);
            col_idx_wr_addr <= addr[BLOCK_ADDR_WIDTH-1:0];
            col_idx_wr_data <= data;
            col_idx_wr_en   <= 1'b1;
            @(posedge clk);
            col_idx_wr_en   <= 1'b0;
        end
    endtask

    task automatic write_block_info(input int addr, input int data);
        begin
            @(posedge clk);
            block_info_wr_addr <= addr[BLOCK_ADDR_WIDTH-1:0];
            block_info_wr_data <= data;
            block_info_wr_en   <= 1'b1;
            @(posedge clk);
            block_info_wr_en   <= 1'b0;
        end
    endtask

    task automatic write_value(input int addr, input int data);
        begin
            @(posedge clk);
            values_wr_addr <= addr[VALUES_ADDR_WIDTH-1:0];
            values_wr_data <= data;
            values_wr_en   <= 1'b1;
            @(posedge clk);
            values_wr_en   <= 1'b0;
        end
    endtask

    // Simple read helper (synchronous read, capture on next clock)
    function automatic int read_row_ptr(input int addr);
        begin
            row_ptr_rd_addr = addr[ROW_PTR_ADDR_WIDTH-1:0];
            @(posedge clk);
            read_row_ptr = row_ptr_rd_data;
        end
    endfunction

    function automatic int read_col_idx(input int addr);
        begin
            col_idx_rd_addr = addr[BLOCK_ADDR_WIDTH-1:0];
            @(posedge clk);
            read_col_idx = col_idx_rd_data;
        end
    endfunction

    function automatic int read_block_info(input int addr);
        begin
            block_info_rd_addr = addr[BLOCK_ADDR_WIDTH-1:0];
            @(posedge clk);
            read_block_info = block_info_rd_data;
        end
    endfunction

    function automatic int read_value(input int addr);
        begin
            values_rd_addr = addr[VALUES_ADDR_WIDTH-1:0];
            @(posedge clk);
            read_value = values_rd_data;
        end
    endfunction

    initial begin
        row_ptr_wr_en    = 1'b0;
        col_idx_wr_en    = 1'b0;
        block_info_wr_en = 1'b0;
        values_wr_en     = 1'b0;

        row_ptr_rd_addr    = '0;
        col_idx_rd_addr    = '0;
        block_info_rd_addr = '0;
        values_rd_addr     = '0;

        rst = 1'b1;
        repeat (3) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // Perform writes
        write_row_ptr(0, 32'd0);
        write_row_ptr(1, 32'd1);

        write_col_idx(0, 32'd3);

        write_block_info(0, 32'h000001F6);

        for (int idx = 0; idx < 36; idx++) begin
            write_value(idx, idx + 32'd1);
        end

        // Allow writes to settle
        @(posedge clk);

        if (read_row_ptr(0) !== 32'd0 || read_row_ptr(1) !== 32'd1) begin
            $display("ERROR: row_ptr readback mismatch");
            $fatal(1);
        end

        if (read_col_idx(0) !== 32'd3) begin
            $display("ERROR: col_idx readback mismatch");
            $fatal(1);
        end

        if (read_block_info(0) !== 32'h000001F6) begin
            $display("ERROR: block_info readback mismatch");
            $fatal(1);
        end

        for (int idx = 0; idx < 36; idx++) begin
            int rd_val = read_value(idx);
            if (rd_val !== idx + 32'd1) begin
                $display("ERROR: values[%0d] readback mismatch got %0d expected %0d",
                         idx, rd_val, idx+1);
                $fatal(1);
            end
        end

        $display("total_buffer_tb PASS");
        $finish;
    end

endmodule

