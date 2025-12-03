`timescale 1ns/1ps

module update_tb;
    localparam int N = 6;
    localparam int M = 28;  // Changed to 28 to support 28x6 matrix input

    logic clk;
    logic rst;
    logic start;
    logic mode;
    logic [7:0] cfg_m;
    logic [7:0] cfg_n;

    logic [31:0] c_left [0:N-1];
    logic [31:0] s_left [0:N-1];
    logic        cs_valid;
    logic [7:0]  update_group_idx;
    logic [7:0]  my_group_idx;

    logic [31:0] matrix_in [0:M-1][0:N-1];
    logic [31:0] back_right_in [0:N-1];
    logic [31:0] x_bottom_in  [0:N-1];
    logic [31:0] r_ext_matrix [0:N-1][0:N-1];

    wire [31:0] matrix_out   [0:M-1][0:N-1];
    wire [31:0] left_sum_out [0:N-1];
    wire [31:0] x_top_out    [0:N-1];
    wire [31:0] c_out_right  [0:N-1];
    wire [31:0] s_out_right  [0:N-1];

    // Clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    update #(
        .N(N),
        .M(M)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .mode(mode),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .c_left(c_left),
        .s_left(s_left),
        .cs_valid(cs_valid),
        .update_group_idx(update_group_idx),
        .my_group_idx(my_group_idx),
        .matrix_in(matrix_in),
        .back_right_in(back_right_in),
        .x_bottom_in(x_bottom_in),
        .r_ext_matrix(r_ext_matrix),
        .matrix_out(matrix_out),
        .left_sum_out(left_sum_out),
        .x_top_out(x_top_out),
        .c_out_right(c_out_right),
        .s_out_right(s_out_right)
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
    function automatic real from_fp32(input [31:0] bits);
        begin
            from_fp32 = $bitstoshortreal(bits);
        end
    endfunction

    // Task to print the full matrix
    task automatic print_matrix(input string label, input int rows, input int cols, input logic [31:0] mat[][]);
        begin
            $display("\n=== %s (M=%0d, N=%0d) ===", label, rows, cols);
            for (int r = 0; r < rows; r++) begin
                $write("[");
                for (int c = 0; c < cols; c++) begin
                    $write("%10.6f", from_fp32(mat[r][c]));
                    if (c != cols-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask

    task automatic init_matrix_qr();
        // Initialize test matrix for QR update mode
        // Matrix: 28 rows x 6 columns from givensrotation.py (149-178) last 6 columns (columns 6-11)
        real matrix_data [0:27][0:5] = '{
            '{3.45,   6.78,   9.01,   1.12,   4.33,   7.55},
            '{4.79,   8.13,   2.46,   5.80,   9.24,   3.67},
            '{6.83,   1.16,   4.49,   8.72,   3.05,   7.38},
            '{9.18,   3.52,   7.86,   2.24,   6.66,   1.08},
            '{1.42,   5.76,   9.10,   3.44,   7.78,   2.12},
            '{2.66,   7.00,   1.34,   5.68,   9.90,   4.22},
            '{3.90,   8.24,   2.58,   7.92,   2.04,   6.36},
            '{5.14,   9.48,   3.82,   8.16,   4.40,   8.64},
            '{6.38,   1.72,   5.06,   9.40,   6.76,   1.92},
            '{7.62,   2.96,   6.30,   1.64,   9.12,   5.20},
            '{8.86,   4.20,   7.54,   2.88,   2.48,   8.48},
            '{1.10,   5.44,   8.78,   4.12,   5.84,   2.76},
            '{2.34,   6.68,   1.02,   5.36,   9.20,   7.04},
            '{3.58,   7.92,   2.26,   6.60,   3.56,   2.32},
            '{4.82,   9.16,   3.50,   7.84,   7.92,   7.60},
            '{6.06,   1.40,   4.74,   9.08,   3.28,   3.88},
            '{7.30,   2.64,   5.98,   1.32,   8.64,   0.16},
            '{8.54,   3.88,   7.22,   2.56,   5.00,   6.44},
            '{9.78,   5.12,   8.46,   3.80,   1.36,   3.72},
            '{2.02,   6.36,   9.70,   5.04,   7.72,   1.00},
            '{3.26,   7.60,   1.94,   6.28,   5.08,   8.28},
            '{4.50,   8.84,   3.18,   7.52,   2.44,   6.56},
            '{5.74,   1.08,   4.42,   8.76,   9.80,   4.84},
            '{6.98,   2.32,   5.66,   1.00,   8.16,   3.12},
            '{8.22,   3.56,   6.90,   2.24,   7.52,   1.40},
            '{9.46,   4.80,   8.14,   3.48,   6.88,   9.68},
            '{1.70,   6.04,   9.38,   4.72,   6.24,   8.96},
            '{2.94,   7.28,   1.62,   5.96,   5.60,   8.24}
        };
        
        // Copy matrix data to matrix_in
        for (int r = 0; r < M; r++) begin
            for (int c = 0; c < N; c++) begin
                if (r < 28 && c < 6) begin
                    matrix_in[r][c] = to_fp32(matrix_data[r][c]);
                end else begin
                    matrix_in[r][c] = to_fp32(0.0);
                end
            end
        end
    endtask

    task automatic clear_back_inputs();
        // Clear back-substitution inputs
        for (int i = 0; i < N; i++) begin
            back_right_in[i] = 32'h0;
            x_bottom_in[i]   = 32'h0;
            for (int j = 0; j < N; j++) begin
                r_ext_matrix[i][j] = 32'h0;
            end
        end
    endtask

    // R module output c/s values for all cycles (from terminal output)
    // All 28 cycles (cycles 0-27)
    real r_c_out [0:N-1][0:35];
    real r_s_out [0:N-1][0:35];
    integer cycle_idx;
    
    // Task to initialize R module c/s output from terminal data
    task automatic init_r_cs_output();
        // row0: 
        r_c_out[0] = '{0.000000,  0.445763,  0.596785,  0.682604,  0.738187,  0.777165,  0.806026,  0.828260,  0.997893,  0.991029,  0.980015,  0.966184,  0.951450,  0.937780,  0.926608,  0.918566,  0.999076,  0.995722,  0.990067,  0.982478,  0.973562,  0.964107,  0.954941,  0.946761,  0.997848,  0.994385,  0.989445,  0.983296,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        r_s_out[0] = '{1.000000,  0.895151,  0.802401,  0.730788,  0.674596,  0.629296,  0.591881,  0.560345,  0.064888,  0.133644,  0.198926,  0.257854,  0.307802,  0.347231,  0.376028,  0.395268,  0.042985,  0.092403,  0.140594,  0.186379,  0.228424,  0.265513,  0.296795,  0.321938,  0.065567,  0.105825,  0.144906,  0.182012,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        // row1: 
        r_c_out[1] = '{0.000000,  0.000000,  0.000000,  0.682715,  0.801602,  0.856256,  0.345535,  0.856241,  0.949094,  0.925402,  0.928339,  0.934422,  0.943552,  0.967769,  0.978064,  0.986583,  0.992821,  0.943902,  0.944154,  0.946030,  0.991395,  0.994018,  0.996301,  0.998081,  0.999267,  0.956986,  0.956401,  0.957101,  0.994888,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        r_s_out[1] = '{0.000000,  1.000000, -1.000000, -0.730685, -0.597858, -0.516552, -0.938406, -0.516577, -0.314992,  0.378988,  0.371735,  0.356167,  0.331224, -0.251841, -0.208303, -0.163261, -0.119609,  0.330225,  0.329505,  0.324078, -0.130901, -0.109213, -0.085932, -0.061929, -0.038285,  0.290134,  0.292057,  0.289756, -0.100981,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        // row2: 
        r_c_out[2] = '{0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.911218,  0.815435,  0.525300,  0.983946,  0.993509,  0.850479,  0.962894,  0.977314,  0.987046,  0.995568,  0.995506,  0.995668,  0.995994,  0.869992,  0.988490,  0.992090,  0.995964,  0.995767,  0.995773,  0.995988,  0.996378,  0.993185,  0.994901,  0.996363,  0.998050,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        r_s_out[2] = '{0.000000,  0.000000,  0.000000,  1.000000,  1.000000, -1.000000,  0.411924,  0.578849,  0.850917,  0.178466,  0.113755,  0.526009, -0.269881, -0.211796, -0.160436,  0.094045,  0.094705,  0.092977,  0.089424,  0.493065, -0.151288, -0.125529,  0.089755,  0.091916,  0.091850,  0.089485,  0.085030, -0.116547, -0.100860, -0.085215,  0.062421,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000};
        // row3: 
        r_c_out[3] = '{0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.792075,  0.085065,  0.111014,  0.883475,  0.982476,  0.998274,  0.998422,  0.998542,  0.774520,  0.948165,  0.967127,  0.979740,  0.991689,  0.998111,  0.998189,  0.842861,  0.985180,  0.988221,  0.991129,  0.993726,  0.998937,  0.998810,  0.998724,  0.883300,  0.000000,  0.000000,  0.000000,  0.000000};
        r_s_out[3] = '{0.000000,  0.000000,  0.000000,  0.000000,  1.000000,  1.000000,  1.000000, -1.000000, -0.610424,  0.996375, -0.993819, -0.468477,  0.186390,  0.058733,  0.056153,  0.053982,  0.632550, -0.317780, -0.254295, -0.200275,  0.128660,  0.061433,  0.060156,  0.538132, -0.171524, -0.153035, -0.132907, -0.111842,  0.046094,  0.048766,  0.050500,  0.468808,  0.000000,  0.000000,  0.000000,  0.000000};
        // row4: 
        r_c_out[4] = '{0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.835316,  0.608630,  0.998535,  0.987912,  0.802318,  0.925624,  0.958456,  0.999495,  0.998988,  0.999056,  0.999094,  0.996056,  0.872958,  0.962466,  0.999868,  0.999511,  0.999494,  0.999497,  0.999520,  0.912355,  0.967449,  0.975859,  0.999836,  0.000000,  0.000000};
        r_s_out[4] = '{0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000, -0.549770,  0.793454,  0.054106,  0.155017,  0.596896, -0.378444, -0.285241,  0.031768,  0.044982,  0.043452,  0.042558,  0.088722,  0.487796, -0.271401,  0.016255,  0.031254,  0.031806,  0.031700,  0.030979,  0.409400, -0.253065, -0.218400,  0.018093,  0.000000,  0.000000};
        // row5:
        r_c_out[5] = '{0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.758509,  0.715945,  0.965795,  0.988632,  0.998674,  0.999499,  0.999999,  0.943239,  0.969759,  0.859216,  0.995889,  0.993246,  0.999717,  0.999066,  0.955444,  0.966571,  0.921369,  0.946865,  0.997292,  0.999726,  0.999687,  0.999355};
        r_s_out[5] = '{1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  0.651663, -0.698156, -0.259307,  0.150357, -0.051482, -0.031638, -0.001582,  0.332113,  0.244063, -0.511612, -0.090577,  0.116030,  0.023795,  0.043220,  0.295173,  0.256401, -0.388690, -0.321632,  0.073544,  0.023390,  0.025033,  0.035922};
    endtask

    // Drive c_left and s_left from R module output, updating each cycle when cs_valid is high
    integer cycle_idx_reg;
    always @(posedge clk) begin
        if (rst) begin
            cycle_idx_reg = 0;
        end else if (cs_valid && (dut.state == 2'h1)) begin  // QR_UPDATE state
            // Update cycle index each cycle when cs_valid is high
            // Use all 38 cycles (cycles 0-37), no delay
            if (cycle_idx_reg < 37) begin  // Max index is 37 (0-37 = 38 cycles)
                cycle_idx_reg = cycle_idx_reg + 1;
            end
        end else if (!cs_valid || (dut.state != 2'h1)) begin
            // Reset when cs_valid goes low or state changes
            cycle_idx_reg = 0;
        end
    end
    
    // Update c_left and s_left combinationally based on cycle_idx_reg
    // In the first cycle when cs_valid becomes high, cycle_idx_reg is 0 (set at previous clock edge)
    always_comb begin
        if (rst || !cs_valid || (dut.state != 2'h1)) begin
            for (int i = 0; i < N; i++) begin
                c_left[i] = 32'h0;
                s_left[i] = 32'h0;
            end
        end else if (cycle_idx_reg < 38) begin
            // Update c_left and s_left from R module output for current cycle
            // Use all 38 cycles (cycles 0-37), no delay
            for (int row = 0; row < N; row++) begin
                c_left[row] = to_fp32(r_c_out[row][cycle_idx_reg]);
                s_left[row] = to_fp32(r_s_out[row][cycle_idx_reg]);
            end
        end else begin
            for (int i = 0; i < N; i++) begin
                c_left[i] = 32'h0;
                s_left[i] = 32'h0;
            end
        end
    end
    
    // Storage for injected data per column per cycle
    real injected_data_history [0:N-1][0:999];  // [column][cycle]
    integer injection_cycle_count;
    
    // Monitor: Record injected data for each column each cycle
    always @(posedge clk) begin
        if (rst) begin
            injection_cycle_count = 0;
            for (int col = 0; col < N; col++) begin
                for (int cyc = 0; cyc < 1000; cyc++) begin
                    injected_data_history[col][cyc] = 0.0;
                end
            end
        end else if (dut.state == 2'h1 && !mode) begin  // QR_UPDATE state
            // Record injected data for each column
            // Data is injected when cs_fire && qr_seed && (inject_row < eff_m) && (tcol < eff_n)
            // We can access dut.inject_row and check if data should be injected
            if (injection_cycle_count < 1000) begin
                for (int col = 0; col < N; col++) begin
                    // Check if this column should have data injected
                    // Data is injected when: cs_fire && qr_seed && (inject_row < eff_m) && (col < eff_n)
                    // We can access dut.y_flow[0][col] to see what was actually injected
                    injected_data_history[col][injection_cycle_count] = from_fp32(dut.y_flow[0][col]);
                end
                injection_cycle_count = injection_cycle_count + 1;
            end
        end
    end
    
    // Task to print injected data history for each column
    task automatic print_injected_data_history();
        begin
            $display("\n=== Injected Data History (per column) ===");
            $display("Total cycles recorded: %0d", injection_cycle_count);
            for (int col = 0; col < N; col++) begin
                $display("==column%0d==", col);
                $write("injected_data=[");
                for (int cycle = 0; cycle < injection_cycle_count && cycle < 100; cycle++) begin
                    $write("%10.6f", injected_data_history[col][cycle]);
                    if (cycle < injection_cycle_count - 1 && cycle < 99) $write(",");
                end
                $display("]");
            end
        end
    endtask
    
    // Task to print input c/s history from R module in the requested format
    // Print all 38 cycles (cycles 0-37), no delay
    task automatic print_input_cs_history();
        begin
            $display("\n=== Input Givens Rotation Coefficients (c/s from R module) ===");
            for (int row = 0; row < N; row++) begin
                $display("==row%0d==", row);
                $write("c_left=[");
                for (int cycle = 0; cycle < 38; cycle++) begin
                    $write("%10.6f", r_c_out[row][cycle]);
                    if (cycle < 37) $write(",");
                end
                $display("]");
                $write("s_left=[");
                for (int cycle = 0; cycle < 38; cycle++) begin
                    $write("%10.6f", r_s_out[row][cycle]);
                    if (cycle < 37) $write(",");
                end
                $display("]");
            end
        end
    endtask

    initial begin
        // Waveform dumping
        `ifdef FSDB
            $fsdbDumpfile("update_tb.fsdb");
            $fsdbDumpvars(0, update_tb);
        `else
            $dumpfile("update_tb.vcd");
            $dumpvars(0, update_tb);
        `endif
        
        rst = 1'b1;
        start = 1'b0;
        mode = 1'b0;
        cfg_m = 8'd28;  // Use all 28 rows from the 28x6 matrix
        cfg_n = 8'd6;
        cs_valid = 1'b0;
        update_group_idx = 8'd0;
        my_group_idx     = 8'd0;

        // Initialize R module c/s output data
        init_r_cs_output();
        
        init_matrix_qr();
        clear_back_inputs();

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // Kick off QR mode update
        // cs_valid starts immediately with start, no delay
        @(posedge clk);
        start <= 1'b1;
        mode  <= 1'b0;
        cs_valid <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        // Drive cs_valid for 38 cycles to provide all R module c/s outputs (cycles 0-37)
        repeat (38) @(posedge clk);
        cs_valid <= 1'b0;

        // Wait for computation to complete
        repeat (80) @(posedge clk);

        $display("\n========================================");
        $display("update_tb (N=%0d, M=%0d): QR Update Mode Results", N, M);
        $display("========================================");
        // Print input matrix
        print_matrix("Input Matrix (matrix_in)", M, N, matrix_in);
        // Print input c/s history from R module
        print_input_cs_history();
        // Print injected data history for each column
        print_injected_data_history();
        // Print only the first N rows of output matrix
        begin
            logic [31:0] matrix_out_n [0:N-1][0:N-1];
            for (int r = 0; r < N; r++) begin
                for (int c = 0; c < N; c++) begin
                    matrix_out_n[r][c] = matrix_out[r][c];
                end
            end
            print_matrix("QR Update Output Matrix (matrix_out)", N, N, matrix_out_n);
        end

        // Switch to back-sub mode with test data
        mode = 1'b1;
        for (int r = 0; r < N; r++) begin
            back_right_in[r] = to_fp32(1.0);
            x_bottom_in[r]   = to_fp32(2.0);
            for (int c = 0; c < N; c++) begin
                if (r == c) begin
                    r_ext_matrix[r][c] = to_fp32(3.0); // diagonal
                end else begin
                    r_ext_matrix[r][c] = to_fp32(1.0); // off-diagonal
                end
            end
        end

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        repeat (100) @(posedge clk);

        $display("\n========================================");
        $display("update_tb (N=%0d, M=%0d): Back-Sub Mode Results", N, M);
        $display("========================================");
        $display("left_sum_out:");
        for (int r = 0; r < N; r++) begin
            $display("  left_sum_out[%0d] = %8.4f", r, from_fp32(left_sum_out[r]));
        end

        $display("\nupdate_tb completed.");
        $finish;
    end

endmodule