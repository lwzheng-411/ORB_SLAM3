`timescale 1ns/1ps

// Default values if not defined via command line
`ifndef TB_N
    `define TB_N 6
`endif
`ifndef TB_M
    `define TB_M 28
`endif

module R_tb;
    localparam int N = `TB_N;
    localparam int M = `TB_M;

    logic clk;
    logic rst;
    logic start;
    logic mode;
    logic factor_type;
    logic [7:0] cfg_m;

    logic [31:0] matrix_in    [0:M-1][0:N-1];
    logic [31:0] backsub_r_in [0:N-1][0:N-1];
    logic [31:0] backsub_b_in [0:N-1];

    wire [31:0] r_out [0:N-1][0:N-1];
    wire [31:0] x_out [0:N-1];
    wire [31:0] c_out [0:N-1];
    wire [31:0] s_out [0:N-1];
    wire        cs_valid;
    wire [7:0]  update_group_idx;
    wire        done;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    R #(
        .N(N),
        .M(M)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .mode(mode),
        .matrix_in(matrix_in),
        .backsub_r_in(backsub_r_in),
        .backsub_b_in(backsub_b_in),
        .factor_type(factor_type),
        .cfg_m(cfg_m),
        .r_out(r_out),
        .x_out(x_out),
        .c_out(c_out),
        .s_out(s_out),
        .cs_valid(cs_valid),
        .update_group_idx(update_group_idx),
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

    task automatic init_matrix();
        // Initialize test matrix using first 6 columns from givensrotation.py (149-178)
        // Matrix: 28 rows x 12 columns (first 6 columns from givensrotation.py, rest extended)
        real matrix_data_6cols [0:27][0:5] = '{
            '{1.23, 4.56, 7.89, 2.34, 5.67, 8.90},
            '{2.47, 5.81, 9.14, 3.68, 7.02, 1.35},
            '{3.71, 7.05, 1.38, 5.92, 9.26, 2.59},
            '{4.95, 8.29, 2.62, 7.16, 1.50, 5.84},
            '{6.19, 9.53, 3.86, 8.40, 2.74, 7.08},
            '{7.43, 1.77, 5.10, 9.64, 3.98, 8.32},
            '{8.67, 3.01, 6.34, 1.88, 5.22, 9.56},
            '{9.91, 4.25, 7.58, 3.12, 6.46, 1.80},
            '{1.15, 5.49, 8.82, 4.36, 7.70, 2.04},
            '{2.39, 6.73, 1.06, 5.60, 8.94, 3.28},
            '{3.63, 7.97, 2.30, 6.84, 1.18, 4.52},
            '{4.87, 9.21, 3.54, 8.08, 2.42, 5.76},
            '{6.11, 1.45, 4.78, 9.32, 3.66, 7.00},
            '{7.35, 2.69, 6.02, 1.56, 4.90, 8.24},
            '{8.59, 3.93, 7.26, 2.80, 6.14, 9.48},
            '{9.83, 5.17, 8.50, 4.04, 7.38, 1.72},
            '{1.07, 6.41, 9.74, 5.28, 8.62, 2.96},
            '{2.31, 7.65, 1.98, 6.52, 9.86, 4.20},
            '{3.55, 8.89, 3.22, 7.76, 2.10, 5.44},
            '{4.79, 1.13, 4.46, 9.00, 3.34, 6.68},
            '{6.03, 2.37, 5.70, 1.24, 4.58, 7.92},
            '{7.27, 3.61, 6.94, 2.48, 5.82, 9.16},
            '{8.51, 4.85, 8.18, 3.72, 7.06, 1.40},
            '{9.75, 6.09, 9.42, 4.96, 8.30, 2.64},
            '{1.99, 7.33, 1.66, 6.20, 9.54, 3.88},
            '{3.23, 8.57, 2.90, 7.44, 1.78, 5.12},
            '{4.47, 9.81, 4.14, 8.68, 3.02, 6.36},
            '{5.71, 2.05, 5.38, 9.92, 4.26, 7.60}
        };
        
        for (int r = 0; r < M; r++) begin
            for (int c = 0; c < N; c++) begin
                if (c < 6) begin
                    // First 6 columns: use data from givensrotation.py
                    matrix_in[r][c] = to_fp32(matrix_data_6cols[r][c]);
                end else begin
                    // Columns 7-12: extend from column 5 (c=5)
                    real base_col5 = matrix_data_6cols[r][5];
                    real val = base_col5 + (c - 5) * 0.1;
                    matrix_in[r][c] = to_fp32(val);
                end
            end
        end
    endtask

    task automatic init_backsub_inputs();
        // Initialize back-substitution inputs with different values from QR results
        // R matrix: upper triangular with different pattern
        // b vector: different from QR mode
        for (int r = 0; r < N; r++) begin
            backsub_b_in[r] = to_fp32((r + 1) * 2.0); // [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
            for (int c = 0; c < N; c++) begin
                if (r == c) begin
                    backsub_r_in[r][c] = to_fp32(10.0 + r * 0.5); // Diagonal: [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
                end else if (r < c) begin
                    backsub_r_in[r][c] = to_fp32(0.5 + (c - r) * 0.3); // Upper triangle: varying values
                end else begin
                    backsub_r_in[r][c] = 32'h00000000; // Lower triangle: 0.0
                end
            end
        end
    endtask

    // Print tasks similar to systolicarray_tb_18x6.sv
    task automatic print_matrix_A();
        begin
            $display("\n=== Input Matrix A (M=%0d, N=%0d) ===", M, N);
            for (int i = 0; i < M; i++) begin
                $write("[");
                for (int j = 0; j < N; j++) begin
                    $write("%10.6f", f32(matrix_in[i][j]));
                    if (j != N-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask

    task automatic print_matrix_R();
        begin
            $display("\n=== Output Matrix R (N=%0d) ===", N);
            for (int i = 0; i < N; i++) begin
                $write("[");
                for (int j = 0; j < N; j++) begin
                    $write("%10.6f", f32(r_out[i][j]));
                    if (j != N-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask

    task automatic print_vector(input string label, input int len, input logic [31:0] vec[]);
        begin
            $write("%s = [", label);
            for (int j = 0; j < len; j++) begin
                $write("%10.6f", f32(vec[j]));
                if (j != len-1) $write(", ");
            end
            $write("]\n");
        end
    endtask

    task automatic check_upper_triangular();
        begin
            shortreal tol = 1e-3;
            int violations = 0;
            for (int i = 0; i < N; i++) begin
                for (int j = 0; j < N; j++) begin
                    if (i > j) begin
                        shortreal v = f32(r_out[i][j]);
                        if (((v < 0) ? -v : v) > tol) violations++;
                    end
                end
            end
            if (violations == 0)
                $display("PASS: R is upper-triangular within tolerance=%f", tol);
            else
                $display("FAIL: R lower-triangle violations: %0d", violations);
        end
    endtask

    task automatic print_matrix_R_input();
        begin
            $display("R matrix (upper triangular):");
            for (int i = 0; i < N; i++) begin
                $write("[");
                for (int j = 0; j < N; j++) begin
                    $write("%10.6f", f32(backsub_r_in[i][j]));
                    if (j != N-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask

    // Cycle counter
    integer cyc;
    bit running;

    // Storage for c_out and s_out values for each row and each cycle
    // Maximum 1000 cycles should be enough
    real c_out_history [0:N-1][0:999];
    real s_out_history [0:N-1][0:999];
    integer cs_valid_cycle_count;

    // Monitor: Collect c_out and s_out for each row when cs_valid is high
    // cs_valid is high during QR decomposition when cycle_count < max_dim
    always @(posedge clk) begin
        if (rst) begin
            cs_valid_cycle_count = 0;
        end else if (cs_valid && (dut.state == 3'h1)) begin  // QR_DECOMP state
            // Record c_out and s_out for all rows at this cycle
            if (cs_valid_cycle_count < 1000) begin
                for (int i = 0; i < N; i++) begin
                    c_out_history[i][cs_valid_cycle_count] = f32(c_out[i]);
                    s_out_history[i][cs_valid_cycle_count] = f32(s_out[i]);
                end
                cs_valid_cycle_count = cs_valid_cycle_count + 1;
            end
        end
    end

    // Task to print c/s history in the requested format
    task automatic print_cs_history();
        begin
            $display("\n=== Givens Rotation Coefficients (c/s) History ===");
            for (int row = 0; row < N; row++) begin
                $display("==row%0d==", row);
                $write("c_out=[");
                for (int cycle = 0; cycle < cs_valid_cycle_count; cycle++) begin
                    $write("%10.6f", c_out_history[row][cycle]);
                    if (cycle < cs_valid_cycle_count - 1) $write(",");
                end
                $display("]");
                $write("s_out=[");
                for (int cycle = 0; cycle < cs_valid_cycle_count; cycle++) begin
                    $write("%10.6f", s_out_history[row][cycle]);
                    if (cycle < cs_valid_cycle_count - 1) $write(",");
                end
                $display("]");
            end
        end
    endtask

    // Stimulus
    initial begin
        rst = 1'b1;
        start = 1'b0;
        mode  = 1'b0;
        factor_type = 1'b1; // use 6 columns
        cfg_m = M;

        init_matrix();
        init_backsub_inputs();

        #50 rst = 1'b0;
        #20;

        // Mode 0 : QR decomposition
        $display("Starting QR decomposition (N=%0d, M=%0d)...", N, M);
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
        $display("Waiting for QR to complete (N=%0d, M=%0d)...", N, M);
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
            $display("\n=== QR Decomposition Results ===");
            print_matrix_A();
            print_matrix_R();
            check_upper_triangular();
            // Print c/s history in the requested format
            print_cs_history();
            $display("\n=== QR Completed at cycle %0d ===", cyc);
        end else begin
            $display("ERROR: QR timeout after %0d cycles", cyc);
        end

        // Wait a bit before starting back-substitution
        #100;

        // Mode 1 : back substitution
        // Use different R matrix and b vector (already initialized in init_backsub_inputs)
        // The backsub_r_in and backsub_b_in are set to different values from QR results
        $display("\nStarting back-substitution...");
        $display("Using different R matrix and b vector (not QR-computed values)");
        
        @(posedge clk);
        start <= 1'b1;
        mode  <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        // Wait for start to take effect (done should be 0 now), then wait for done to go back to 1
        @(posedge clk);
        fork
            wait(done === 1'b1);  // Wait for done to go back to 1 (idle, computation finished)
            #200000; // timeout
        join_any

        if (done) begin
            $display("\n=== Back-Substitution Results ===");
            $display("Input R matrix (upper triangular):");
            print_matrix_R_input();
            $display("\nInput b vector:");
            print_vector("b_in", N, backsub_b_in);
            $display("\nSolution x:");
            print_vector("x_out", N, x_out);
            $display("\n=== Back-Substitution Completed at cycle %0d ===", cyc);
        end else begin
            $display("ERROR: Back-substitution timeout after %0d cycles", cyc);
        end

        $display("\n=== Test Completed ===");
        #20;
        $finish;
    end

    // Waveform dump
    initial begin
`ifdef FSDB
        $fsdbDumpfile("R_tb.fsdb");
        $fsdbDumpvars(0, R_tb);
        $fsdbDumpMDA();
        // Explicitly dump internal signals that might be optimized away
        $fsdbDumpvars(2, dut);
`else
        $dumpfile("R_tb.vcd");
        $dumpvars(0, R_tb);
        // Explicitly dump internal signals that might be optimized away
        $dumpvars(2, dut);
`endif
    end

endmodule

