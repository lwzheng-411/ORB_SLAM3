`timescale 1ns/1ps

module systolicarray_tb_back;

    // ----------------------------------------------------------------------
    // Parameters
    // ----------------------------------------------------------------------
    localparam int N = 5;              // Must match DUT parameter
    localparam int CLK_PERIOD_NS = 10; // 100MHz
    localparam int TIMEOUT_CYCLES = 2000; // Extended for QR+BackSub phases

    // ----------------------------------------------------------------------
    // Clock & Reset
    // ----------------------------------------------------------------------
    reg clk;
    reg rst;
    reg start;

    // ----------------------------------------------------------------------
    // DUT I/O
    // ----------------------------------------------------------------------
    reg  [31:0] matrix_in [0:N-1][0:N-1];
    reg  [31:0] b_in      [0:N-1];

    wire [31:0] r_out     [0:N-1][0:N-1];
    wire [31:0] b_out     [0:N-1];
    wire [31:0] x_out     [0:N-1];
    wire        done;
    // Dynamic size config
    logic [7:0] cfg_m, cfg_n;

    // ----------------------------------------------------------------------
    // Helpers: float32 conversions
    // ----------------------------------------------------------------------
    function automatic [31:0] to_float32(input real r);
        shortreal sr;
        begin
            sr = r;
            to_float32 = $shortrealtobits(sr);
        end
    endfunction

    // FP32 bits to shortreal for pretty print
    function automatic shortreal f32(input [31:0] bits);
        f32 = $bitstoshortreal(bits);
    endfunction

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
            // Default example (same as reference TB)
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

    // Pretty-print helpers
    task automatic print_matrix_A();
      begin
        $display("\n=== Input Matrix A (N=%0d) ===", N);
        for (int ii = 0; ii < N; ii++) begin
          $write("A[%0d,:] = [", ii);
          for (int jj = 0; jj < N; jj++) begin
            $write("%0f", f32(matrix_in[ii][jj]));
            if (jj != N-1) $write(", ");
          end
          $write("]\n");
        end
      end
    endtask

    task automatic print_matrix_R();
      begin
        $display("\n=== Output Matrix R (N=%0d) ===", N);
        for (int ii = 0; ii < N; ii++) begin
          $write("R[%0d,:] = [", ii);
          for (int jj = 0; jj < N; jj++) begin
            $write("%0f", f32(r_out[ii][jj]));
            if (jj != N-1) $write(", ");
          end
          $write("]\n");
        end
      end
    endtask

    task automatic print_vector(input string label, input logic [31:0] vec[]);
      begin
        $write("%s = [", label);
        for (int jj = 0; jj < N; jj++) begin
          $write("%0f", f32(vec[jj]));
          if (jj != N-1) $write(", ");
        end
        $write("]\n");
      end
    endtask

    task automatic check_upper_triangular();
      begin
        shortreal tol = 1e-3;
        int violations = 0;
        for (int ii = 0; ii < N; ii++) begin
          for (int jj = 0; jj < N; jj++) begin
            if (ii > jj) begin
              shortreal v = f32(r_out[ii][jj]);
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

    function automatic real to_real32(input [31:0] bits);
        shortreal sr;
        begin
            sr = $bitstoshortreal(bits);
            to_real32 = sr;
        end
    endfunction

    // ----------------------------------------------------------------------
    // DUT Instance
    // ----------------------------------------------------------------------
    systolicarray #(
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .matrix_in(matrix_in),
        .b_in(b_in),
        .cfg_m(cfg_m),
        .cfg_n(cfg_n),
        .r_out(r_out),
        .b_out(b_out),
        .x_out(x_out),
        .done(done)
    );

    // ----------------------------------------------------------------------
    // Clock generator
    // ----------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD_NS/2) clk = ~clk;
    end

    // ----------------------------------------------------------------------
    // Stimulus
    // ----------------------------------------------------------------------
    integer i, j;
    integer cycles;

    task automatic init_inputs_identity_matrix_and_seq_b;
        begin
            for (i = 0; i < N; i = i + 1) begin
                for (j = 0; j < N; j = j + 1) begin
                    matrix_in[i][j] = (i == j) ? 32'h3F800000 /* 1.0f */ : 32'h00000000 /* 0.0f */;
                end
            end
            for (i = 0; i < N; i = i + 1) begin
                b_in[i] = to_float32(i + 1.0);
            end
        end
    endtask

    task automatic print_outputs;
        begin
            $display("==== R matrix (float32) ====");
            for (i = 0; i < N; i = i + 1) begin
                for (j = 0; j < N; j = j + 1) begin
                    $write("%0.6f ", to_real32(r_out[i][j]));
                end
                $write("\n");
            end
            $display("==== b_out (float32) ====");
            for (i = 0; i < N; i = i + 1) begin
                $display("b_out[%0d] = %0.6f (0x%08h)", i, to_real32(b_out[i]), b_out[i]);
            end
            $display("==== x_out (float32) ====");
            for (i = 0; i < N; i = i + 1) begin
                $display("x_out[%0d] = %0.6f (0x%08h)", i, to_real32(x_out[i]), x_out[i]);
            end
        end
    endtask

    // ----------------------------------------------------------------------
    // Reset/start and file-driven initialization (aligned with reference TB)
    // ----------------------------------------------------------------------
    string matrix_file, b_vector_file;
    integer cyc; bit running;
    initial begin
        rst   = 1'b1;
        start = 1'b0;

        // Determine input file names; defaults
        if (!$value$plusargs("MATRIX_FILE=%s", matrix_file)) matrix_file = "matrix.hex";
        if (!$value$plusargs("B_VECTOR_FILE=%s", b_vector_file)) b_vector_file = "b_vector.hex";

        // Clear arrays
        for (int r=0; r<N; r++) begin
            for (int c=0; c<N; c++) matrix_in[r][c] = 32'h0;
            b_in[r] = 32'h0;
        end

        // Load from files or fallback to built-ins
        load_matrix_from_file(matrix_file);
        load_b_vector_from_file(b_vector_file);

        cfg_m = N;
        cfg_n = N;
        #50 rst = 1'b0;
        #20 start = 1'b1;
        #10 start = 1'b0;
    end

    // Cycle counter
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cyc <= 0; running <= 0;
        end else begin
            if (start) begin
                running <= 1; cyc <= 1;
            end else if (running) begin
                cyc <= cyc + 1;
            end
        end
    end

    // Results display and basic checks
    initial begin
        $display("Waiting for QR+BackSub to complete...");
        fork
            wait(done === 1'b1);
            #200000; // timeout
        join_any

        $display("\n=== Input Matrix A ===");
        print_matrix_A();
        $display("\n=== Input B Vector ===");
        print_vector("b_in", b_in);

        $display("\n=== Outputs (R, b_out, x_out) ===");
        print_matrix_R();
        print_vector("b_out", b_out);
        print_vector("x_out", x_out);

        check_upper_triangular();

        $display("\n=== Test Completed at cycle %0d ===", cyc);
        #20 $finish;
    end

    // Waveform dump (FSDB to Sim/tb.fsdb)
    initial begin
        $fsdbDumpfile("/hpc/home/connect.lzheng842/projects/systolic/QR/Sim/tb.fsdb");
        $fsdbDumpvars(0, systolicarray_tb_back);
        $fsdbDumpMDA();
        // Add specific signal groups for debugging
        $fsdbDumpvars(1, systolicarray_tb_back.dut.state);
        $fsdbDumpvars(1, systolicarray_tb_back.dut.cycle_count);
        $fsdbDumpvars(1, systolicarray_tb_back.dut.qr_enable_src);
        $fsdbDumpvars(1, systolicarray_tb_back.dut.back_enable_src);
    end

endmodule


