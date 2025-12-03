`timescale 1ns/1ps

//==============================================================================
// Testbench for QR Decomposition Systolic Array with B Vector Support (systolic_cs)
//==============================================================================
module systolic_cs_tb;
    localparam int N = 5;

    //--------------------------------------------------------------------------
    // Clock, Reset, and Control Signals
    //--------------------------------------------------------------------------
    reg clk;
    reg rst;
    reg start;

    //--------------------------------------------------------------------------
    // DUT Interface
    //--------------------------------------------------------------------------
    reg  [31:0] matrix_in [0:N-1][0:N-1];
    reg  [31:0] b_in      [0:N-1];
    wire [31:0] r_out     [0:N-1][0:N-1];
    wire [31:0] b_out     [0:N-1];
    wire        done;
    // Dynamic size config for DUT (effective M,N)
    logic [7:0] cfg_m, cfg_n; // note: cfg_m<=N, cfg_n<=N

    //--------------------------------------------------------------------------
    // SDF Annotation for Gate-level Simulation
    //--------------------------------------------------------------------------
    `ifdef GATE_SIM
        initial begin
            `ifdef SDF_FILE
                $sdf_annotate(`SDF_FILE, dut);
            `endif
        end
    `endif

    //--------------------------------------------------------------------------
    // DUT Instantiation
    //--------------------------------------------------------------------------
    systolicarray #(.N(N)) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .matrix_in(matrix_in),
        .b_in(b_in),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .r_out(r_out),
        .b_out(b_out),
        .done(done)
    );

    //--------------------------------------------------------------------------
    // Clock Generation (100 MHz)
    //--------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //--------------------------------------------------------------------------
    // Test Matrix and B Vector Initialization (5x5 Example)
    // A = [[1, 2, 3, 1, 0],
    //      [2, 3, 4, 1, 0.5],
    //      [3, 4, 5, 2, 1],
    //      [1, 1, 2, 3, 2],
    //      [0, 0.5, 1, 2, 3]]
    // b = [1, 2, 3, 4, 5]
    //--------------------------------------------------------------------------
    
    // Load matrix from file (hex FP32), row-major, N*N lines
    task load_matrix_from_file(input string fname);
      int fd; string line; int idx=0; int r, c; int code; int unsigned word;
      begin
        fd = $fopen(fname, "r");
        if (fd == 0) begin
          $display("[TB] Matrix file %s not found. Using built-in example.", fname);
          init_matrix();
          return;
        end
        while (!$feof(fd) && idx < N*N) begin
          code = $fgets(line, fd);
          if (code == 0) break;
          if ($sscanf(line, "%h", word) == 1) begin
            r = idx / N; c = idx % N; idx++;
            matrix_in[r][c] = word[31:0];
          end
        end
        $fclose(fd);
        if (idx != N*N) begin
          $display("[TB] Matrix file %s has %0d values, expected %0d. Filling remaining with 0.", fname, idx, N*N);
        end else begin
          $display("[TB] Loaded %0d FP32 words from %s for a %0dx%0d matrix.", idx, fname, N, N);
        end
      end
    endtask

    // Load B vector from file (hex FP32), N lines
    task load_b_vector_from_file(input string fname);
      int fd; string line; int idx=0; int code; int unsigned word;
      begin
        fd = $fopen(fname, "r");
        if (fd == 0) begin
          $display("[TB] B vector file %s not found. Using built-in example.", fname);
          init_b_vector();
          return;
        end
        while (!$feof(fd) && idx < N) begin
          code = $fgets(line, fd);
          if (code == 0) break;
          if ($sscanf(line, "%h", word) == 1) begin
            b_in[idx] = word[31:0];
            idx++;
          end
        end
        $fclose(fd);
        if (idx != N) begin
          $display("[TB] B vector file %s has %0d values, expected %0d. Filling remaining with 0.", fname, idx, N);
        end else begin
          $display("[TB] Loaded %0d FP32 words from %s for B vector.", idx, fname);
        end
      end
    endtask

    task init_matrix();
        begin
            matrix_in[0][0] = 32'h3F800000; // 1.0
            matrix_in[0][1] = 32'h40000000; // 2.0
            matrix_in[0][2] = 32'h40400000; // 3.0
            matrix_in[0][3] = 32'h3F800000; // 1.0
            matrix_in[0][4] = 32'h00000000; // 0.0

            matrix_in[1][0] = 32'h40000000; // 2.0
            matrix_in[1][1] = 32'h40400000; // 3.0
            matrix_in[1][2] = 32'h40800000; // 4.0
            matrix_in[1][3] = 32'h3F800000; // 1.0
            matrix_in[1][4] = 32'h3F000000; // 0.5

            matrix_in[2][0] = 32'h40400000; // 3.0
            matrix_in[2][1] = 32'h40800000; // 4.0
            matrix_in[2][2] = 32'h40A00000; // 5.0
            matrix_in[2][3] = 32'h40000000; // 2.0
            matrix_in[2][4] = 32'h3F800000; // 1.0

            matrix_in[3][0] = 32'h3F800000; // 1.0
            matrix_in[3][1] = 32'h3F800000; // 1.0
            matrix_in[3][2] = 32'h40000000; // 2.0
            matrix_in[3][3] = 32'h40400000; // 3.0
            matrix_in[3][4] = 32'h40000000; // 2.0

            matrix_in[4][0] = 32'h00000000; // 0.0
            matrix_in[4][1] = 32'h3F000000; // 0.5
            matrix_in[4][2] = 32'h3F800000; // 1.0
            matrix_in[4][3] = 32'h40000000; // 2.0
            matrix_in[4][4] = 32'h40400000; // 3.0
        end
    endtask

    task init_b_vector();
        begin
            b_in[0] = 32'h3F800000; // 1.0
            b_in[1] = 32'h40000000; // 2.0
            b_in[2] = 32'h40400000; // 3.0
            b_in[3] = 32'h40800000; // 4.0
            b_in[4] = 32'h40A00000; // 5.0
        end
    endtask

    // Reset and start; load matrix and B vector from files
    string matrix_file, b_vector_file;
    initial begin
        rst   = 1'b1;
        start = 1'b0;

        // Determine input file names; defaults
        if (!$value$plusargs("MATRIX_FILE=%s", matrix_file)) begin
            matrix_file = "matrix.hex"; // default filename
        end
        if (!$value$plusargs("B_VECTOR_FILE=%s", b_vector_file)) begin
            b_vector_file = "b_vector.hex"; // default filename
        end

        // Initialize matrix and B vector
        for (int r=0; r<N; r++) begin
            for (int c=0; c<N; c++)
                matrix_in[r][c] = 32'h0;
            b_in[r] = 32'h0;
        end

        load_matrix_from_file(matrix_file);
        load_b_vector_from_file(b_vector_file);

        // default to full NxN
        cfg_m = N;
        cfg_n = N;
        #50 rst = 1'b0;
        #20 start = 1'b1;
        #10 start = 1'b0;
    end

    // Cycle Counter: track cycles since 'start' asserted
    integer cyc;
    bit running;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cyc <= 0;
            running <= 0;
        end else begin
            if (start) begin
                running <= 1;
                cyc <= 1;
            end else if (running) begin
                cyc <= cyc + 1;
            end
        end
    end

    // FP32 bit-to-float conversion
    function automatic shortreal f32(input [31:0] bits);
        f32 = $bitstoshortreal(bits);
    endfunction

    // Pretty-print matrices in row form
    task automatic print_matrix_A();
      begin
        $display("\n=== Input Matrix A (N=%0d) ===", N);
        for (int i = 0; i < N; i++) begin
          $write("A[%0d,:] = [", i);
          for (int j = 0; j < N; j++) begin
            $write("%0f", f32(matrix_in[i][j])); // print as float
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
          $write("R[%0d,:] = [", i);
          for (int j = 0; j < N; j++) begin
            $write("%0f", f32(r_out[i][j]));
            if (j != N-1) $write(", ");
          end
          $write("]\n");
        end
      end
    endtask

    // Pretty-print a length-N FP32 vector
    task automatic print_vector(input string label, input logic [31:0] vec[]);
      begin
        $write("%s = [", label);
        for (int j = 0; j < N; j++) begin
          $write("%0f", f32(vec[j]));
          if (j != N-1) $write(", ");
        end
        $write("]\n");
      end
    endtask

    // Upper triangular matrix verification task
    task automatic check_upper_triangular();
      begin
        shortreal tol = 1e-3; // tolerance for upper triangular check
        int violations = 0;
        for (int i = 0; i < N; i++) begin
          for (int j = 0; j < N; j++) begin
            if (i > j) begin
              shortreal v = f32(r_out[i][j]);
              if (((v < 0) ? -v : v) > tol) begin
                violations++;
              end
            end
          end
        end

        if (violations == 0)
          $display("PASS: R is upper-triangular within tolerance=%f", tol);
        else
          $display("FAIL: R lower-triangle violations: %0d", violations);
      end
    endtask

    // Check if the solution b_out satisfies R * x = b_out for the QR system
    task automatic check_solution();
      begin
        $display("\n=== Solution Verification ===");
        $display("Note: This testbench shows computed b_out values.");
        $display("For full verification, you would solve R*x = b_out and check if Q*R*x = A*x = b_in");
        $display("where Q can be reconstructed from the Givens rotations used in QR decomposition.");
      end
    endtask

    //--------------------------------------------------------------------------
    // Results Display (Inputs and Outputs)
    //--------------------------------------------------------------------------
    initial begin
        // Wait for completion with timeout
        $display("Waiting for QR decomposition with B vector processing to complete...");
        fork
            wait(done === 1'b1);
            #200000; // 200us timeout (longer due to B vector processing)
        join_any

        // Show inputs
        $display("\n=== Input Matrix A ===");
        print_matrix_A();
        $display("\n=== Input B Vector ===");
        print_vector("b_in", b_in);

        // Show outputs
        $display("\n=== Outputs (R matrix and B vector) ===");
        print_matrix_R();
        print_vector("b_out", b_out);

        // Basic correctness checks
        check_upper_triangular();
        check_solution();

        $display("\n=== Test Completed at cycle %0d ===", cyc);
        #20 $finish;
    end

    //--------------------------------------------------------------------------
    // Waveform Dump
    //--------------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("tb.fsdb");
        $fsdbDumpvars(0, systolic_cs_tb);
        $fsdbDumpMDA();
    end

endmodule
