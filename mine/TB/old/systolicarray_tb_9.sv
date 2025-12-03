`timescale 1ns/1ps

//==============================================================================
// Testbench for QR Decomposition Systolic Array
//==============================================================================
module systolicarray_tb;
    localparam int N = 9;
    localparam int DEBUG = 1;          // set 0 to disable
    localparam int DEBUG_ROW_FROM = 4; // print i>=4 to cut verbosity

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
    // Dynamic size config
    logic [7:0] cfg_m, cfg_n;

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
    // Test Matrix Initialization (5x5 Example)
    // A = [[1, 2, 3, 1, 0],
    //      [2, 3, 4, 1, 0.5],
    //      [3, 4, 5, 2, 1],
    //      [1, 1, 2, 3, 2],
    //      [0, 0.5, 1, 2, 3]]
    //--------------------------------------------------------------------------
    // Optional: load N x N matrix from file (hex FP32), row-major, N*N lines
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

    // Optional: load N-element b vector from file (hex FP32), N lines
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
          for (int i = idx; i < N; i++) b_in[i] = 32'h0;
        end else begin
          $display("[TB] Loaded %0d FP32 words from %s for a %0d-element b vector.", idx, fname, N);
        end
      end
    endtask

    task init_b_vector();
        begin
            // Initialize b vector with simple test values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            b_in[0] = 32'h3F800000; // 1.0
            b_in[1] = 32'h40000000; // 2.0
            b_in[2] = 32'h40400000; // 3.0
            b_in[3] = 32'h40800000; // 4.0
            b_in[4] = 32'h40A00000; // 5.0
            b_in[5] = 32'h40C00000; // 6.0
            b_in[6] = 32'h40E00000; // 7.0
            b_in[7] = 32'h41000000; // 8.0
            b_in[8] = 32'h41100000; // 9.0
        end
    endtask

    task init_matrix();
        begin
            // ============= ORIGINAL MATRIX (COMMENTED OUT - HAS LINEAR DEPENDENCIES) =============
            // Original matrix had identical rows (rows 5 & 7) causing numerical instability
            // matrix_in[0] = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            // matrix_in[1] = [2.0, 3.0, 4.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0];
            // matrix_in[2] = [3.0, 4.0, 5.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0];
            // matrix_in[3] = [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 1.0, 1.0];
            // matrix_in[4] = [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 1.0];
            // matrix_in[5] = [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0];
            // matrix_in[6] = [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0];
            // matrix_in[7] = [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0];
            // matrix_in[8] = [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 0.0];
            
            // matrix_in[0][0] = 32'h3F800000; // 1.0
            // matrix_in[0][1] = 32'h40000000; // 2.0
            // matrix_in[0][2] = 32'h40400000; // 3.0
            // matrix_in[0][3] = 32'h3F800000; // 1.0
            // matrix_in[0][4] = 32'h00000000; // 0.0
            // matrix_in[0][5] = 32'h00000000; // 0.0
            // matrix_in[0][6] = 32'h3F800000; // 1.0
            // matrix_in[0][7] = 32'h3F800000; // 1.0
            // matrix_in[0][8] = 32'h3F800000; // 1.0
            
            // matrix_in[1][0] = 32'h40000000; // 2.0
            // matrix_in[1][1] = 32'h40400000; // 3.0
            // matrix_in[1][2] = 32'h40800000; // 4.0
            // matrix_in[1][3] = 32'h3F800000; // 1.0
            // matrix_in[1][4] = 32'h3F000000; // 0.5
            // matrix_in[1][5] = 32'h3F800000; // 1.0
            // matrix_in[1][6] = 32'h3F000000; // 0.5
            // matrix_in[1][7] = 32'h3F800000; // 1.0
            // matrix_in[1][8] = 32'h3F800000; // 1.0
            
            // matrix_in[2][0] = 32'h40400000; // 3.0
            // matrix_in[2][1] = 32'h40800000; // 4.0
            // matrix_in[2][2] = 32'h40A00000; // 5.0
            // matrix_in[2][3] = 32'h40000000; // 2.0
            // matrix_in[2][4] = 32'h3F800000; // 1.0
            // matrix_in[2][5] = 32'h3F800000; // 1.0
            // matrix_in[2][6] = 32'h00000000; // 0.0
            // matrix_in[2][7] = 32'h3F800000; // 1.0
            // matrix_in[2][8] = 32'h3F800000; // 1.0
            
            // matrix_in[3][0] = 32'h3F800000; // 1.0
            // matrix_in[3][1] = 32'h3F800000; // 1.0
            // matrix_in[3][2] = 32'h40000000; // 2.0
            // matrix_in[3][3] = 32'h40400000; // 3.0
            // matrix_in[3][4] = 32'h40000000; // 2.0
            // matrix_in[3][5] = 32'h3F800000; // 1.0
            // matrix_in[3][6] = 32'h3F000000; // 0.5
            // matrix_in[3][7] = 32'h3F800000; // 1.0
            // matrix_in[3][8] = 32'h3F800000; // 1.0
            
            // matrix_in[4][0] = 32'h00000000; // 0.0
            // matrix_in[4][1] = 32'h3F000000; // 0.5
            // matrix_in[4][2] = 32'h3F800000; // 1.0
            // matrix_in[4][3] = 32'h40000000; // 2.0
            // matrix_in[4][4] = 32'h40400000; // 3.0
            // matrix_in[4][5] = 32'h3F800000; // 1.0
            // matrix_in[4][6] = 32'h40400000; // 3.0
            // matrix_in[4][7] = 32'h3F800000; // 1.0
            // matrix_in[4][8] = 32'h3F800000; // 1.0
            
            // matrix_in[5][0] = 32'h00000000; // 0.0
            // matrix_in[5][1] = 32'h3F000000; // 0.5
            // matrix_in[5][2] = 32'h3F800000; // 1.0
            // matrix_in[5][3] = 32'h40000000; // 2.0
            // matrix_in[5][4] = 32'h40400000; // 3.0
            // matrix_in[5][5] = 32'h3F800000; // 1.0
            // matrix_in[5][6] = 32'h40000000; // 2.0
            // matrix_in[5][7] = 32'h3F800000; // 1.0
            // matrix_in[5][8] = 32'h3F800000; // 1.0
            
            // matrix_in[6][0] = 32'h00000000; // 0.0
            // matrix_in[6][1] = 32'h3F000000; // 0.5
            // matrix_in[6][2] = 32'h3F800000; // 1.0
            // matrix_in[6][3] = 32'h40000000; // 2.0
            // matrix_in[6][4] = 32'h40400000; // 3.0
            // matrix_in[6][5] = 32'h3F800000; // 1.0
            // matrix_in[6][6] = 32'h3F800000; // 1.0
            // matrix_in[6][7] = 32'h3F800000; // 1.0
            // matrix_in[6][8] = 32'h3F800000; // 1.0
            
            // matrix_in[7][0] = 32'h00000000; // 0.0
            // matrix_in[7][1] = 32'h3F000000; // 0.5
            // matrix_in[7][2] = 32'h3F800000; // 1.0
            // matrix_in[7][3] = 32'h40000000; // 2.0
            // matrix_in[7][4] = 32'h40400000; // 3.0
            // matrix_in[7][5] = 32'h3F800000; // 1.0
            // matrix_in[7][6] = 32'h40000000; // 2.0
            // matrix_in[7][7] = 32'h3F800000; // 1.0
            // matrix_in[7][8] = 32'h3F800000; // 1.0
            
            // matrix_in[8][0] = 32'h00000000; // 0.0
            // matrix_in[8][1] = 32'h3F000000; // 0.5
            // matrix_in[8][2] = 32'h3F800000; // 1.0
            // matrix_in[8][3] = 32'h40000000; // 2.0
            // matrix_in[8][4] = 32'h40400000; // 3.0
            // matrix_in[8][5] = 32'h3F800000; // 1.0
            // matrix_in[8][6] = 32'h40000000; // 2.0
            // matrix_in[8][7] = 32'h3F800000; // 1.0
            // matrix_in[8][8] = 32'h00000000; // 0.0

            // ============= NEW WELL-CONDITIONED TEST MATRIX =============
            // Row 0: [1.0, 2.0, 3.0, 1.0, 0.5, 1.0, 0.5, 1.0, 4.0]
            matrix_in[0][0] = 32'h3F800000; // 1.0
            matrix_in[0][1] = 32'h40000000; // 2.0
            matrix_in[0][2] = 32'h40400000; // 3.0
            matrix_in[0][3] = 32'h3F800000; // 1.0
            matrix_in[0][4] = 32'h3F000000; // 0.5
            matrix_in[0][5] = 32'h3F800000; // 1.0
            matrix_in[0][6] = 32'h3F000000; // 0.5
            matrix_in[0][7] = 32'h3F800000; // 1.0
            matrix_in[0][8] = 32'h40800000; // 4.0

            // Row 1: [2.0, 3.0, 4.0, 1.0, 0.5, 2.0, 0.5, 1.0, 5.0]
            matrix_in[1][0] = 32'h40000000; // 2.0
            matrix_in[1][1] = 32'h40400000; // 3.0
            matrix_in[1][2] = 32'h40800000; // 4.0
            matrix_in[1][3] = 32'h3F800000; // 1.0
            matrix_in[1][4] = 32'h3F000000; // 0.5
            matrix_in[1][5] = 32'h40000000; // 2.0
            matrix_in[1][6] = 32'h3F000000; // 0.5
            matrix_in[1][7] = 32'h3F800000; // 1.0
            matrix_in[1][8] = 32'h40A00000; // 5.0

            // Row 2: [3.0, 4.0, 5.0, 2.0, 1.0, 3.0, 1.0, 2.0, 6.0]
            matrix_in[2][0] = 32'h40400000; // 3.0
            matrix_in[2][1] = 32'h40800000; // 4.0
            matrix_in[2][2] = 32'h40A00000; // 5.0
            matrix_in[2][3] = 32'h40000000; // 2.0
            matrix_in[2][4] = 32'h3F800000; // 1.0
            matrix_in[2][5] = 32'h40400000; // 3.0
            matrix_in[2][6] = 32'h3F800000; // 1.0
            matrix_in[2][7] = 32'h40000000; // 2.0
            matrix_in[2][8] = 32'h40C00000; // 6.0

            // Row 3: [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 1.0, 7.0]
            matrix_in[3][0] = 32'h3F800000; // 1.0
            matrix_in[3][1] = 32'h3F800000; // 1.0
            matrix_in[3][2] = 32'h40000000; // 2.0
            matrix_in[3][3] = 32'h40400000; // 3.0
            matrix_in[3][4] = 32'h40000000; // 2.0
            matrix_in[3][5] = 32'h3F800000; // 1.0
            matrix_in[3][6] = 32'h40800000; // 4.0
            matrix_in[3][7] = 32'h3F800000; // 1.0
            matrix_in[3][8] = 32'h40E00000; // 7.0

            // Row 4: [0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 3.0, 1.0, 5.0]
            matrix_in[4][0] = 32'h00000000; // 0.0
            matrix_in[4][1] = 32'h3F000000; // 0.5
            matrix_in[4][2] = 32'h3F800000; // 1.0
            matrix_in[4][3] = 32'h40000000; // 2.0
            matrix_in[4][4] = 32'h40400000; // 3.0
            matrix_in[4][5] = 32'h3F800000; // 1.0
            matrix_in[4][6] = 32'h40400000; // 3.0
            matrix_in[4][7] = 32'h3F800000; // 1.0
            matrix_in[4][8] = 32'h40A00000; // 5.0

            // Row 5: [0.5, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 1.0, 8.0]
            matrix_in[5][0] = 32'h3F000000; // 0.5
            matrix_in[5][1] = 32'h3F800000; // 1.0
            matrix_in[5][2] = 32'h40000000; // 2.0
            matrix_in[5][3] = 32'h40400000; // 3.0
            matrix_in[5][4] = 32'h40800000; // 4.0
            matrix_in[5][5] = 32'h40000000; // 2.0
            matrix_in[5][6] = 32'h40800000; // 4.0
            matrix_in[5][7] = 32'h3F800000; // 1.0
            matrix_in[5][8] = 32'h41000000; // 8.0

            // Row 6: [2.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 1.0, 3.0]
            matrix_in[6][0] = 32'h40000000; // 2.0
            matrix_in[6][1] = 32'h3F800000; // 1.0
            matrix_in[6][2] = 32'h40000000; // 2.0
            matrix_in[6][3] = 32'h40400000; // 3.0
            matrix_in[6][4] = 32'h40800000; // 4.0
            matrix_in[6][5] = 32'h40000000; // 2.0
            matrix_in[6][6] = 32'h40A00000; // 5.0
            matrix_in[6][7] = 32'h3F800000; // 1.0
            matrix_in[6][8] = 32'h40400000; // 3.0

            // Row 7: [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 5.0, 2.0, 8.0]
            matrix_in[7][0] = 32'h3F800000; // 1.0
            matrix_in[7][1] = 32'h40000000; // 2.0
            matrix_in[7][2] = 32'h40400000; // 3.0
            matrix_in[7][3] = 32'h40800000; // 4.0
            matrix_in[7][4] = 32'h40A00000; // 5.0
            matrix_in[7][5] = 32'h40000000; // 2.0
            matrix_in[7][6] = 32'h40A00000; // 5.0
            matrix_in[7][7] = 32'h40000000; // 2.0
            matrix_in[7][8] = 32'h41000000; // 8.0

            // Row 8: [0.5, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 2.0, 4.0]
            matrix_in[8][0] = 32'h3F000000; // 0.5
            matrix_in[8][1] = 32'h3F800000; // 1.0
            matrix_in[8][2] = 32'h40000000; // 2.0
            matrix_in[8][3] = 32'h40400000; // 3.0
            matrix_in[8][4] = 32'h40800000; // 4.0
            matrix_in[8][5] = 32'h40000000; // 2.0
            matrix_in[8][6] = 32'h40A00000; // 5.0
            matrix_in[8][7] = 32'h40000000; // 2.0
            matrix_in[8][8] = 32'h40800000; // 4.0
        end
    endtask

  // Reset and start; load matrix and b vector from files
  string matrix_file, b_vector_file;
  initial begin
    rst   = 1'b1;
    start = 1'b0;

    // Determine input file names; defaults to matrix.hex and b_vector.hex in TB dir
    if (!$value$plusargs("MATRIX_FILE=%s", matrix_file)) begin
      matrix_file = "matrix.hex"; // default filename
    end
    if (!$value$plusargs("B_VECTOR_FILE=%s", b_vector_file)) begin
      b_vector_file = "b_vector.hex"; // default filename
    end

    // Initialize matrix and b vector then try to load from files; fallback to built-in examples
    for (int r=0; r<N; r++) begin
      for (int c=0; c<N; c++)
        matrix_in[r][c] = 32'h0;
      b_in[r] = 32'h0;
    end

    load_matrix_from_file(matrix_file);
    load_b_vector_from_file(b_vector_file);

    #50 rst = 1'b0;
    #20 start = 1'b1;
    #10 start = 1'b0;
    // default config
    initial begin cfg_m = N; cfg_n = N; end
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

    // START formula (must match DUT): clk0,2,3,5,6 for col=0,2,3,4,5
    function automatic int START_COL(input int j);
        START_COL = (j == 0) ? 0 : (j % 2 == 0) ? (3 * j / 2 - 1) : (3 * (j-1) / 2);
    endfunction

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
        shortreal tol = 1e-3; // declare and assign together
        int violations = 0;
        for (int i = 0; i < N; i++) begin
          for (int j = 0; j < N; j++) begin
            if (i > j) begin
              shortreal v = f32(r_out[i][j]); // declare and assign together
              if (((v < 0) ? -v : v) > tol) begin
                violations++;
              end
            end
          end
        end
        // Remove duplicate print calls from check_upper_triangular

        if (violations == 0)
          $display("PASS: R is upper-triangular within tolerance=%f", tol);
        else
          $display("FAIL: R lower-triangle violations: %0d", violations);
      end
    endtask



    //--------------------------------------------------------------------------
    // Results Display (Inputs and Outputs only)
    //--------------------------------------------------------------------------
    initial begin
        // Wait for completion with timeout
        $display("Waiting for completion...");
        fork
            wait(done === 1'b1);
            #100000; // 100us timeout
        join_any

        // Show inputs
        $display("\n=== Input Matrix A ===");
        print_matrix_A();
        $display("\n=== Input Vector b ===");
        print_vector("b_in", b_in);

        // Show outputs
        $display("\n=== Outputs (b_out, R) ===");
        print_vector("b_out", b_out);
        print_matrix_R();

        // Basic correctness check
        check_upper_triangular();

        #20 $finish;
    end

    // Optional PE I/O debug prints (hierarchical access)
    generate if (DEBUG) begin : GEN_DEBUG
      always @(posedge clk) begin
        if (running) begin
          for (int i = DEBUG_ROW_FROM; i < N; i++) begin
            // sqrt at (i,i)
            if (dut.enable_flow[i][i]) begin
              $display("[cyc=%0d] SQRT(i=%0d): x=%f y=%f r=%f",
                       cyc, i,
                       f32(dut.x_orig_wire[i]),
                       f32(dut.y_orig_wire[i]),
                       f32(dut.r_new_wire[i]));
            end
            // divider at (i,i+1)
            if (i < N-1 && dut.enable_div[i]) begin
              $display("[cyc=%0d] DIV (i=%0d): r=%f c=%f s=%f",
                       cyc, i,
                       f32(dut.r_new_wire[i]),
                       f32(dut.c_out_from_pe[i][i+1]),
                       f32(dut.s_out_from_pe[i][i+1]));
            end
            // rotation along row i
            for (int j = i + 2; j < N; j++) begin
              if (dut.enable_flow[i][j]) begin
                $display("[cyc=%0d] ROT (i=%0d,j=%0d): in=%f c=%f s=%f r1'=%f out=%f",
                         cyc, i, j,
                         f32(dut.y_flow[i][j]),
                         f32(dut.c_flow[i][j]),
                         f32(dut.s_flow[i][j]),
                         f32(dut.r1_reg[i][j]),
                         f32(dut.y_out_from_pe[i][j]));
              end
            end
          end
        end
      end
    end endgenerate
    //--------------------------------------------------------------------------
    // Waveform Dump (minimal)
    //--------------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("tb.fsdb");
        $fsdbDumpvars(0, systolicarray_tb);
        $fsdbDumpMDA();
    end

endmodule