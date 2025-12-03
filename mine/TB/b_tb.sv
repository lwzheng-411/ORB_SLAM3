`timescale 1ns/1ps

// Default values if not defined via command line
`ifndef TB_N
    `define TB_N 6
`endif
`ifndef TB_M
    `define TB_M 6
`endif

module b_tb;
    localparam int N = `TB_N;
    localparam int M = `TB_M;

    logic clk;
    logic rst;
    logic start;
    logic mode;
    logic [7:0] cfg_m;

    logic [31:0] c_in [0:N-1];
    logic [31:0] s_in [0:N-1];
    logic        cs_valid;
    logic [7:0]  update_group_idx;
    logic [7:0]  my_group_idx;

    logic [31:0] b_in [0:N-1];

    wire [31:0] b_out [0:N-1];
    wire        done;

    // Cycle counter
    integer cyc;
    bit running;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    b #(
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .mode(mode),
        .cfg_m(cfg_m),
        .c_in(c_in),
        .s_in(s_in),
        .cs_valid(cs_valid),
        .update_group_idx(update_group_idx),
        .my_group_idx(my_group_idx),
        .b_in(b_in),
        .b_out(b_out),
        .done(done)
    );

    // Helper: convert real to FP32
    function automatic [31:0] to_fp32(input real val);
        shortreal sr;
        begin
            sr = val;
            to_fp32 = $shortrealtobits(sr);
        end
    endfunction

    // Helper: convert FP32 to real for display
    function automatic shortreal f32(input [31:0] bits);
        f32 = $bitstoshortreal(bits);
    endfunction

    task automatic init_b_vector();
        // Initialize b vector: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for (int i = 0; i < N; i++) begin
            b_in[i] = to_fp32((i + 1) * 1.0);
        end
    endtask

    task automatic init_cs_coefficients();
        // Initialize c/s coefficients for Givens rotation
        // For testing, use simple values: c=0.6, s=0.8 (for 3-4-5 triangle)
        for (int i = 0; i < N; i++) begin
            c_in[i] = to_fp32(0.6);
            s_in[i] = to_fp32(0.8);
        end
    endtask

    // Simulate R module providing c/s signals
    // b module needs cs_valid to be high for each row injection
    // Only provide cs_valid when b module is in QR_UPDATE state (done == 0)
    reg [7:0] inject_row_sim;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            inject_row_sim <= 8'd0;
            cs_valid <= 1'b0;
            update_group_idx <= 8'd0;
        end else if (!mode && !done && (inject_row_sim < cfg_m)) begin
            // Provide cs_valid for each row, similar to R module behavior
            // Only when b module is working (done == 0)
            cs_valid <= 1'b1;
            update_group_idx <= 8'd0;  // Group 0
            inject_row_sim <= inject_row_sim + 8'd1;
        end else begin
            cs_valid <= 1'b0;
            if (done || mode) begin
                inject_row_sim <= 8'd0;  // Reset when done or mode changes
            end
        end
    end

    // Stimulus
    initial begin
        rst = 1'b1;
        start = 1'b0;
        mode  = 1'b0;  // QR mode
        cfg_m = M;     // Use all 6 rows
        my_group_idx = 8'd0;

        init_b_vector();
        init_cs_coefficients();

        #50 rst = 1'b0;
        #20;

        // Mode 0 : QR update
        $display("Starting b vector update (N=%0d, cfg_m=%0d)...", N, cfg_m);
        start = 1'b1;
        mode  = 1'b0;
        #10;
        start = 1'b0;
    end

    // Cycle counter
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cyc <= 0;
            running <= 0;
        end else begin
            if (start) begin
                running <= 1;
                cyc <= 1;  // Start counting from 1 when start is asserted
            end else if (running) begin
                // Continue counting while running
                cyc <= cyc + 1;
                // Stop counting when done goes high (computation finished)
                // The cycle when done goes high is included in the count
                if (done) begin
                    running <= 0;
                end
            end
        end
    end

    // Results - QR mode
    initial begin
        $display("Waiting for b update to complete (N=%0d, cfg_m=%0d)...", N, cfg_m);
        // Wait for start signal to be sent and take effect
        wait(start === 1'b1);
        @(posedge clk);  // Wait for start to take effect (done should be 0 now)
        // Wait one more cycle to ensure cycle counter has started
        @(posedge clk);
        // Now wait for done to go back to 1 (idle, finished)
        fork
            wait(done === 1'b1);  // Wait for done to go back to 1 (idle, computation finished)
            #200000; // timeout
        join_any

        if (done) begin
            $display("\n=== B Vector Update Results ===");
            $display("\nInput b vector:");
            $write("  b_in = [");
            for (int i = 0; i < N; i++) begin
                $write("%8.6f", f32(b_in[i]));
                if (i < N-1) $write(", ");
            end
            $display("]");

            $display("\nOutput b vector (after Givens rotation):");
            $write("  b_out = [");
            for (int i = 0; i < N; i++) begin
                $write("%8.6f", f32(b_out[i]));
                if (i < N-1) $write(", ");
            end
            $display("]");

            $display("\n=== B Update Completed at cycle %0d ===", cyc);
        end else begin
            $display("ERROR: B update timeout after %0d cycles", cyc);
        end

        $display("\n=== Test Completed ===");
        #20;
        $finish;
    end

    // Waveform dump
    initial begin
`ifdef FSDB
        $fsdbDumpfile("b_tb.fsdb");
        $fsdbDumpvars(0, b_tb);
        $fsdbDumpMDA();
        // Explicitly dump internal signals that might be optimized away
        $fsdbDumpvars(2, dut);
`else
        $dumpfile("b_tb.vcd");
        $dumpvars(0, b_tb);
        // Explicitly dump internal signals that might be optimized away
        $dumpvars(2, dut);
`endif
    end

endmodule

