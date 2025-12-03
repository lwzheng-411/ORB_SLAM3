`timescale 1ns/1ps

//==============================================================================
// Testbench for QR Decomposition Systolic Array (18x6 input, 6x6 array)
//==============================================================================
module systolicarray_tb_18x6;
    localparam int N = 6;   // array size 6x6
    localparam int M = 18;  // input rows

    reg clk;
    reg rst;
    reg start;

    // DUT I/O (FP32)
    reg  [31:0] matrix_in [0:M-1][0:N-1];
    reg  [31:0] b_in      [0:M-1];
    wire [31:0] r_out     [0:N-1][0:N-1];
    wire [31:0] b_out     [0:N-1];
    wire        done;

    //----------------------------------------------------------------------------
    // SDF Annotation for Gate-level Simulation
    //----------------------------------------------------------------------------
    `ifdef GATE_SIM
        initial begin
            `ifdef SDF_FILE
                $sdf_annotate(`SDF_FILE, dut);
            `endif
        end
    `endif

    // DUT instantiation
    systolicarray_back_moreIP #(
        .N(N),
        .M(M)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .matrix_in(matrix_in),
        .b_in(b_in),
        .r_out(r_out),
        .b_out(b_out),
        .x_out(),
        .done(done)
    );

    // Clock 100MHz
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Helpers
    function automatic shortreal f32(input [31:0] bits);
        f32 = $bitstoshortreal(bits);
    endfunction

    // Style-aligned printer (no-arg wrapper)
    task automatic print_matrix_A();
      begin
        $display("\n=== Input Matrix A (M=%0d,N=%0d) ===", M, N);
        for (int i = 0; i < M; i++) begin
          $write("A[%0d,:] = [", i);
          for (int j = 0; j < N; j++) begin
            $write("%0f", f32(matrix_in[i][j]));
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

    task automatic print_vector(input string label, input int len, input logic [31:0] vec[]);
      begin
        $write("%s = [", label);
        for (int j = 0; j < len; j++) begin
          $write("%0f", f32(vec[j]));
          if (j != len-1) $write(", ");
        end
        $write("]\n");
      end
    endtask

    // Simple checks
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

    // File loaders (hex FP32)
    task load_matrix_from_file(input string fname);
      int fd; string line; int idx=0; int r, c; int code; int unsigned word;
      begin
        fd = $fopen(fname, "r");
        if (fd == 0) begin
          $display("[TB] Matrix file %s not found. Using built-in example.", fname);
          init_matrix();
          return;
        end
        while (!$feof(fd) && idx < M*N) begin
          code = $fgets(line, fd);
          if (code == 0) break;
          if ($sscanf(line, "%h", word) == 1) begin
            r = idx / N; c = idx % N; idx++;
            matrix_in[r][c] = word[31:0];
          end
        end
        $fclose(fd);
        if (idx != M*N)
          $display("[TB] Matrix file %s has %0d values, expected %0d. Filling remaining with 0.", fname, idx, M*N);
        else
          $display("[TB] Loaded %0d FP32 words from %s for a %0dx%0d matrix.", idx, fname, M, N);
      end
    endtask

    task load_b_vector_from_file(input string fname);
      int fd; string line; int idx=0; int code; int unsigned word;
      begin
        fd = $fopen(fname, "r");
        if (fd == 0) begin
          $display("[TB] B vector file %s not found. Using built-in example.", fname);
          init_b_vector();
          return;
        end
        while (!$feof(fd) && idx < M) begin
          code = $fgets(line, fd);
          if (code == 0) break;
          if ($sscanf(line, "%h", word) == 1) begin
            b_in[idx] = word[31:0];
            idx++;
          end
        end
        $fclose(fd);
        if (idx != M) begin
          $display("[TB] B vector file %s has %0d values, expected %0d. Filling remaining with 0.", fname, idx, M);
          for (int i = idx; i < M; i++) b_in[i] = 32'h0;
        end else begin
          $display("[TB] Loaded %0d FP32 words from %s for a %0d-element b vector.", idx, fname, M);
        end
      end
    endtask

    // Built-in example (embedded 18x6 matrix)
    task init_matrix();
      begin
        // Row 0
        matrix_in[0][0] = 32'h42C80000; matrix_in[0][1] = 32'h00000000; matrix_in[0][2] = 32'h00000000; matrix_in[0][3] = 32'h00000000; matrix_in[0][4] = 32'h00000000; matrix_in[0][5] = 32'h00000000;
        // Row 1
        matrix_in[1][0] = 32'h00000000; matrix_in[1][1] = 32'h42C80000; matrix_in[1][2] = 32'h00000000; matrix_in[1][3] = 32'h00000000; matrix_in[1][4] = 32'h00000000; matrix_in[1][5] = 32'h00000000;
        // Row 2
        matrix_in[2][0] = 32'h00000000; matrix_in[2][1] = 32'h00000000; matrix_in[2][2] = 32'h42C80000; matrix_in[2][3] = 32'h00000000; matrix_in[2][4] = 32'h00000000; matrix_in[2][5] = 32'h00000000;
        // Row 3
        matrix_in[3][0] = 32'h00000000; matrix_in[3][1] = 32'h00000000; matrix_in[3][2] = 32'h00000000; matrix_in[3][3] = 32'h419EE148; matrix_in[3][4] = 32'h40000000; matrix_in[3][5] = 32'hBFFD70A4;
        // Row 4
        matrix_in[4][0] = 32'h00000000; matrix_in[4][1] = 32'h00000000; matrix_in[4][2] = 32'h00000000; matrix_in[4][3] = 32'hBFF851EC; matrix_in[4][4] = 32'h419F3333; matrix_in[4][5] = 32'hBF051EB8;
        // Row 5
        matrix_in[5][0] = 32'h00000000; matrix_in[5][1] = 32'h00000000; matrix_in[5][2] = 32'h00000000; matrix_in[5][3] = 32'h400147AE; matrix_in[5][4] = 32'h3EC28F5C; matrix_in[5][5] = 32'h419F3333;
        // Row 6
        matrix_in[6][0] = 32'h3F800000; matrix_in[6][1] = 32'h3CA3D70A; matrix_in[6][2] = 32'h3C23D70A; matrix_in[6][3] = 32'h00000000; matrix_in[6][4] = 32'h00000000; matrix_in[6][5] = 32'h00000000;
        // Row 7
        matrix_in[7][0] = 32'hBCA3D70A; matrix_in[7][1] = 32'h3F800000; matrix_in[7][2] = 32'h3C23D70A; matrix_in[7][3] = 32'h00000000; matrix_in[7][4] = 32'h00000000; matrix_in[7][5] = 32'h00000000;
        // Row 8
        matrix_in[8][0] = 32'hBC23D70A; matrix_in[8][1] = 32'hBC23D70A; matrix_in[8][2] = 32'h3F800000; matrix_in[8][3] = 32'h00000000; matrix_in[8][4] = 32'h00000000; matrix_in[8][5] = 32'h00000000;
        // Row 9
        matrix_in[9][0]  = 32'h3EDC28F6; matrix_in[9][1]  = 32'h3F83D70A; matrix_in[9][2]  = 32'h3F851EB8; matrix_in[9][3]  = 32'h3F800000; matrix_in[9][4]  = 32'h00000000; matrix_in[9][5]  = 32'h00000000;
        // Row 10
        matrix_in[10][0] = 32'hBF828F5C; matrix_in[10][1] = 32'h3EC7AE14; matrix_in[10][2] = 32'hBF30A3D7; matrix_in[10][3] = 32'h00000000; matrix_in[10][4] = 32'h3F800000; matrix_in[10][5] = 32'h00000000;
        // Row 11
        matrix_in[11][0] = 32'h3F19999A; matrix_in[11][1] = 32'hBF666666; matrix_in[11][2] = 32'h00000000; matrix_in[11][3] = 32'h00000000; matrix_in[11][4] = 32'h00000000; matrix_in[11][5] = 32'h3F800000;
        // Row 12
        matrix_in[12][0] = 32'hC151EB85; matrix_in[12][1] = 32'hC289F5C3; matrix_in[12][2] = 32'h4287DC29; matrix_in[12][3] = 32'hBD75C28F; matrix_in[12][4] = 32'h3F933333; matrix_in[12][5] = 32'hBF59999A;
        // Row 13
        matrix_in[13][0] = 32'h42781EB8; matrix_in[13][1] = 32'h411D70A4; matrix_in[13][2] = 32'hC230A3D7; matrix_in[13][3] = 32'h4136147B; matrix_in[13][4] = 32'h3F8F5C29; matrix_in[13][5] = 32'hC1423D71;
        // Row 14
        matrix_in[14][0] = 32'h419E3D71; matrix_in[14][1] = 32'hC2AC3333; matrix_in[14][2] = 32'hC21C47AE; matrix_in[14][3] = 32'hC1263D71; matrix_in[14][4] = 32'h3E947AE1; matrix_in[14][5] = 32'hC0B00000;
        // Row 15
        matrix_in[15][0] = 32'hC15CCCCD; matrix_in[15][1] = 32'h40F4CCCD; matrix_in[15][2] = 32'h4265F5C3; matrix_in[15][3] = 32'hBCA3D70A; matrix_in[15][4] = 32'h415F0A3D; matrix_in[15][5] = 32'hBD75C28F;
        // Row 16
        matrix_in[16][0] = 32'hC1173333; matrix_in[16][1] = 32'h4182CCCD; matrix_in[16][2] = 32'hBFF47AE1; matrix_in[16][3] = 32'h415EE148; matrix_in[16][4] = 32'hC0370A3D; matrix_in[16][5] = 32'hC161C28F;
        // Row 17
        matrix_in[17][0] = 32'hBC23D70A; matrix_in[17][1] = 32'hBC23D70A; matrix_in[17][2] = 32'hBE6B851F; matrix_in[17][3] = 32'hC07C28F6; matrix_in[17][4] = 32'hC19EB852; matrix_in[17][5] = 32'hBDA3D70A;
      end
    endtask

    task init_b_vector();
      begin
        for (int i=0; i<M; i++) b_in[i] = 32'h3F800000; // 1.0
      end
    endtask

    // Stimulus
    integer cyc; bit running;
    initial begin
      rst   = 1'b1;
      start = 1'b0;

      // initialize embedded inputs
      init_matrix();
      init_b_vector();

      #50 rst = 1'b0;
      #20 start = 1'b1;
      #10 start = 1'b0;
    end

    // cycle counter
    always @(posedge clk or posedge rst) begin
      if (rst) begin
        cyc <= 0; running <= 0;
      end else begin
        if (start) begin running <= 1; cyc <= 1; end
        else if (running) cyc <= cyc + 1;
      end
    end

    // Results
    initial begin
      $display("Waiting for QR to complete (18x6)...");
      fork
        wait(done === 1'b1);
                #200000;
      join_any

      $display("\n=== Inputs ===");
      print_matrix_A();
      $display("\n=== Outputs ===");
      print_vector("b_out", N, b_out);
      print_matrix_R();
      check_upper_triangular();

      $display("\n=== Test Completed at cycle %0d ===", cyc);
      #20 $finish;
    end

    // Waveform
    initial begin
`ifdef FSDB
      $fsdbDumpfile("tb_18x6.fsdb");
      $fsdbDumpvars(0, systolicarray_tb_18x6);
      $fsdbDumpMDA();
`else
      $dumpfile("tb_18x6.vcd");
      $dumpvars(0, systolicarray_tb_18x6);
`endif
    end

endmodule


