`timescale 1ns/1ps

//==============================================================================
// Testbench for QR Decomposition Systolic Array
//==============================================================================
module systolicarray_tb;
    localparam int N = 7;

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
    wire [31:0] r_out     [0:N-1][0:N-1];
    wire [31:0] c_out     [0:N-1];
    wire [31:0] s_out     [0:N-1];
    wire        done;

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
        .r_out(r_out),
        .c_out(c_out),
        .s_out(s_out),
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

    task init_matrix();
        begin
            matrix_in[0][0] = 32'h3F800000; // 1.0
            matrix_in[0][1] = 32'h40000000; // 2.0
            matrix_in[0][2] = 32'h40400000; // 3.0
            matrix_in[0][3] = 32'h3F800000; // 1.0
            matrix_in[0][4] = 32'h00000000; // 0.0
            matrix_in[0][5] = 32'h00000000; // 0.0
            matrix_in[0][6] = 32'h3F800000; // 1.0

            matrix_in[1][0] = 32'h40000000; // 2.0
            matrix_in[1][1] = 32'h40400000; // 3.0
            matrix_in[1][2] = 32'h40800000; // 4.0
            matrix_in[1][3] = 32'h3F800000; // 1.0
            matrix_in[1][4] = 32'h3F000000; // 0.5
            matrix_in[1][5] = 32'h3F800000; // 1.0
            matrix_in[1][6] = 32'h3F000000; // 0.5


            matrix_in[2][0] = 32'h40400000; // 3.0
            matrix_in[2][1] = 32'h40800000; // 4.0
            matrix_in[2][2] = 32'h40A00000; // 5.0
            matrix_in[2][3] = 32'h40000000; // 2.0
            matrix_in[2][4] = 32'h3F800000; // 1.0
            matrix_in[2][5] = 32'h3F800000; // 1.0
            matrix_in[2][6] = 32'h00000000; // 0.0

            matrix_in[3][0] = 32'h3F800000; // 1.0
            matrix_in[3][1] = 32'h3F800000; // 1.0
            matrix_in[3][2] = 32'h40000000; // 2.0
            matrix_in[3][3] = 32'h40400000; // 3.0
            matrix_in[3][4] = 32'h40000000; // 2.0
            matrix_in[3][5] = 32'h3F800000; // 1.0
            matrix_in[3][6] = 32'h3F000000; // 0.5

            matrix_in[4][0] = 32'h00000000; // 0.0
            matrix_in[4][1] = 32'h3F000000; // 0.5
            matrix_in[4][2] = 32'h3F800000; // 1.0
            matrix_in[4][3] = 32'h40000000; // 2.0
            matrix_in[4][4] = 32'h40400000; // 3.0
            matrix_in[4][5] = 32'h3F800000; // 1.0
            matrix_in[4][6] = 32'h40400000; // 3.0

            matrix_in[5][0] = 32'h00000000; // 0.0
            matrix_in[5][1] = 32'h3F000000; // 0.5
            matrix_in[5][2] = 32'h3F800000; // 1.0
            matrix_in[5][3] = 32'h40000000; // 2.0
            matrix_in[5][4] = 32'h40400000; // 3.0
            matrix_in[5][5] = 32'h3F800000; // 1.0
            matrix_in[5][6] = 32'h40000000; // 2.0

            matrix_in[6][0] = 32'h00000000; // 0.0
            matrix_in[6][1] = 32'h3F000000; // 0.5
            matrix_in[6][2] = 32'h3F800000; // 1.0
            matrix_in[6][3] = 32'h40000000; // 2.0
            matrix_in[6][4] = 32'h40400000; // 3.0
            matrix_in[6][5] = 32'h3F800000; // 1.0
            matrix_in[6][6] = 32'h3F800000; // 1.0 
        end
    endtask

  // Reset and start; load matrix from file via +MATRIX_FILE=<path> (hex words)
  string matrix_file;
  initial begin
    rst   = 1'b1;
    start = 1'b0;

    // Determine input file name; default to matrix.hex in TB dir
    if (!$value$plusargs("MATRIX_FILE=%s", matrix_file)) begin
      matrix_file = "matrix.hex"; // default filename
    end

    // Initialize matrix then try to load from file; fallback to built-in example
    for (int r=0; r<N; r++)
      for (int c=0; c<N; c++)
        matrix_in[r][c] = 32'h0;

    load_matrix_from_file(matrix_file);

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

        // Show outputs
        $display("\n=== Outputs (c, s, R) ===");
        print_vector("c", c_out);
        print_vector("s", s_out);
        print_matrix_R();

        // Basic correctness check
        check_upper_triangular();

        #20 $finish;
    end
    //--------------------------------------------------------------------------
    // Waveform Dump (minimal)
    //--------------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("tb.fsdb");
        $fsdbDumpvars(0, systolicarray_tb);
        $fsdbDumpMDA();
    end

endmodule