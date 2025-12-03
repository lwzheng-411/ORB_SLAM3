`timescale 1ns/1ps

//==============================================================================
// Testbench for 5x5 Matrix QR Decomposition using True 2D Systolic Array
// Reference: "A Truly Two-Dimensional Systolic Array FPGA Implementation of QR Decomposition"
//==============================================================================
module baseline_5x5_tb;
    
    // Parameters matching the baseline module
    localparam int M = 5;  // Number of rows
    localparam int N = 5;  // Number of columns
    localparam int DEBUG = 1;  // Enable debug prints

    //--------------------------------------------------------------------------
    // Clock, Reset, and Control Signals
    //--------------------------------------------------------------------------
    reg clk;
    reg rst;
    reg start;

    //--------------------------------------------------------------------------
    // DUT Interface
    //--------------------------------------------------------------------------
    reg  [31:0] matrix_in [0:M-1][0:N-1];
    wire [31:0] r_out     [0:M-1][0:N-1];
    wire [31:0] c_out     [0:M-1];
    wire [31:0] s_out     [0:M-1];
    wire        done;

    //--------------------------------------------------------------------------
    // DUT Instantiation
    //--------------------------------------------------------------------------
    baseline #(
        .M(M),
        .N(N)
    ) dut (
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
    // Test Matrix Initialization
    // Well-conditioned 5x5 test matrix for QR decomposition
    //--------------------------------------------------------------------------
    task init_test_matrix();
        begin
            $display("[TB] Initializing 5x5 test matrix...");
            
            // Test Matrix A (well-conditioned for numerical stability):
            // A = [[3.0, 2.0, 1.0, 4.0, 2.0],
            //      [1.0, 4.0, 2.0, 1.0, 3.0],
            //      [2.0, 1.0, 3.0, 2.0, 1.0],
            //      [1.0, 2.0, 1.0, 5.0, 2.0],
            //      [2.0, 3.0, 2.0, 1.0, 4.0]]
            
            // Row 0: [3.0, 2.0, 1.0, 4.0, 2.0]
            matrix_in[0][0] = 32'h40400000; // 3.0
            matrix_in[0][1] = 32'h40000000; // 2.0
            matrix_in[0][2] = 32'h3F800000; // 1.0
            matrix_in[0][3] = 32'h40800000; // 4.0
            matrix_in[0][4] = 32'h40000000; // 2.0
            
            // Row 1: [1.0, 4.0, 2.0, 1.0, 3.0]
            matrix_in[1][0] = 32'h3F800000; // 1.0
            matrix_in[1][1] = 32'h40800000; // 4.0
            matrix_in[1][2] = 32'h40000000; // 2.0
            matrix_in[1][3] = 32'h3F800000; // 1.0
            matrix_in[1][4] = 32'h40400000; // 3.0
            
            // Row 2: [2.0, 1.0, 3.0, 2.0, 1.0]
            matrix_in[2][0] = 32'h40000000; // 2.0
            matrix_in[2][1] = 32'h3F800000; // 1.0
            matrix_in[2][2] = 32'h40400000; // 3.0
            matrix_in[2][3] = 32'h40000000; // 2.0
            matrix_in[2][4] = 32'h3F800000; // 1.0
            
            // Row 3: [1.0, 2.0, 1.0, 5.0, 2.0]
            matrix_in[3][0] = 32'h3F800000; // 1.0
            matrix_in[3][1] = 32'h40000000; // 2.0
            matrix_in[3][2] = 32'h3F800000; // 1.0
            matrix_in[3][3] = 32'h40A00000; // 5.0
            matrix_in[3][4] = 32'h40000000; // 2.0
            
            // Row 4: [2.0, 3.0, 2.0, 1.0, 4.0]
            matrix_in[4][0] = 32'h40000000; // 2.0
            matrix_in[4][1] = 32'h40400000; // 3.0
            matrix_in[4][2] = 32'h40000000; // 2.0
            matrix_in[4][3] = 32'h3F800000; // 1.0
            matrix_in[4][4] = 32'h40800000; // 4.0
        end
    endtask

    //--------------------------------------------------------------------------
    // Alternative test matrices for comprehensive testing
    //--------------------------------------------------------------------------
    task init_identity_matrix();
        begin
            $display("[TB] Initializing 5x5 identity matrix...");
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    if (i == j) 
                        matrix_in[i][j] = 32'h3F800000; // 1.0
                    else 
                        matrix_in[i][j] = 32'h00000000; // 0.0
                end
            end
        end
    endtask

    task init_hilbert_matrix();
        begin
            $display("[TB] Initializing 5x5 Hilbert matrix (challenging case)...");
            // Hilbert matrix: H(i,j) = 1/(i+j+1)
            // Note: This is a challenging matrix for numerical algorithms
            matrix_in[0][0] = 32'h3F800000; // 1.0     = 1/1
            matrix_in[0][1] = 32'h3F000000; // 0.5     = 1/2
            matrix_in[0][2] = 32'h3EAAAAAB; // 0.33333 = 1/3
            matrix_in[0][3] = 32'h3E800000; // 0.25    = 1/4
            matrix_in[0][4] = 32'h3E4CCCCD; // 0.2     = 1/5
            
            matrix_in[1][0] = 32'h3F000000; // 0.5     = 1/2
            matrix_in[1][1] = 32'h3EAAAAAB; // 0.33333 = 1/3
            matrix_in[1][2] = 32'h3E800000; // 0.25    = 1/4
            matrix_in[1][3] = 32'h3E4CCCCD; // 0.2     = 1/5
            matrix_in[1][4] = 32'h3E2AAAAB; // 0.16667 = 1/6
            
            matrix_in[2][0] = 32'h3EAAAAAB; // 0.33333 = 1/3
            matrix_in[2][1] = 32'h3E800000; // 0.25    = 1/4
            matrix_in[2][2] = 32'h3E4CCCCD; // 0.2     = 1/5
            matrix_in[2][3] = 32'h3E2AAAAB; // 0.16667 = 1/6
            matrix_in[2][4] = 32'h3E124925; // 0.14286 = 1/7
            
            matrix_in[3][0] = 32'h3E800000; // 0.25    = 1/4
            matrix_in[3][1] = 32'h3E4CCCCD; // 0.2     = 1/5
            matrix_in[3][2] = 32'h3E2AAAAB; // 0.16667 = 1/6
            matrix_in[3][3] = 32'h3E124925; // 0.14286 = 1/7
            matrix_in[3][4] = 32'h3E000000; // 0.125   = 1/8
            
            matrix_in[4][0] = 32'h3E4CCCCD; // 0.2     = 1/5
            matrix_in[4][1] = 32'h3E2AAAAB; // 0.16667 = 1/6
            matrix_in[4][2] = 32'h3E124925; // 0.14286 = 1/7
            matrix_in[4][3] = 32'h3E000000; // 0.125   = 1/8
            matrix_in[4][4] = 32'hEDDB6DB7; // 0.11111 = 1/9
        end
    endtask

    //--------------------------------------------------------------------------
    // Utility Functions
    //--------------------------------------------------------------------------
    
    // Convert IEEE 754 single precision to real for display
    function automatic shortreal f32_to_real(input [31:0] fp32);
        f32_to_real = $bitstoshortreal(fp32);
    endfunction
    
    // Pretty print matrix
    task print_matrix_A();
        begin
            $display("\n=== Input Matrix A (5x5) ===");
            for (int i = 0; i < M; i++) begin
                $write("A[%0d,:] = [", i);
                for (int j = 0; j < N; j++) begin
                    $write("%8.4f", f32_to_real(matrix_in[i][j]));
                    if (j < N-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask
    
    task print_matrix_R();
        begin
            $display("\n=== Output Matrix R (5x5) ===");
            for (int i = 0; i < M; i++) begin
                $write("R[%0d,:] = [", i);
                for (int j = 0; j < N; j++) begin
                    $write("%8.4f", f32_to_real(r_out[i][j]));
                    if (j < N-1) $write(", ");
                end
                $write("]\n");
            end
        end
    endtask
    
    task print_givens_params();
        begin
            $display("\n=== Givens Rotation Parameters ===");
            $write("c = [");
            for (int i = 0; i < M; i++) begin
                $write("%8.4f", f32_to_real(c_out[i]));
                if (i < M-1) $write(", ");
            end
            $write("]\n");
            
            $write("s = [");
            for (int i = 0; i < M; i++) begin
                $write("%8.4f", f32_to_real(s_out[i]));
                if (i < M-1) $write(", ");
            end
            $write("]\n");
        end
    endtask

    //--------------------------------------------------------------------------
    // Verification Tasks
    //--------------------------------------------------------------------------
    
    task check_upper_triangular();
        real tolerance = 1e-6;
        int violations = 0;
        real val;
        begin
            $display("\n=== Checking Upper Triangular Property ===");
            for (int i = 0; i < M; i++) begin
                for (int j = 0; j < N; j++) begin
                    if (i > j) begin
                        val = f32_to_real(r_out[i][j]);
                        if ((val > tolerance) || (val < -tolerance)) begin
                            violations++;
                            $display("VIOLATION: R[%0d][%0d] = %f (should be ~0)", i, j, val);
                        end
                    end
                end
            end
            
            if (violations == 0)
                $display("PASS: R matrix is upper triangular (tolerance = %e)", tolerance);
            else
                $display("FAIL: %0d violations of upper triangular property", violations);
        end
    endtask
    
    task check_r_diagonal_positive();
        int violations = 0;
        real val;
        begin
            $display("\n=== Checking R Diagonal Elements ===");
            for (int i = 0; i < M; i++) begin
                val = f32_to_real(r_out[i][i]);
                if (val <= 0.0) begin
                    violations++;
                    $display("VIOLATION: R[%0d][%0d] = %f (should be > 0)", i, i, val);
                end else begin
                    $display("R[%0d][%0d] = %f ✓", i, i, val);
                end
            end
            
            if (violations == 0)
                $display("PASS: All diagonal elements are positive");
            else
                $display("FAIL: %0d diagonal elements are not positive", violations);
        end
    endtask

    //--------------------------------------------------------------------------
    // Test Execution
    //--------------------------------------------------------------------------
    
    // Cycle counter for monitoring
    integer cycle_count;
    always @(posedge clk) begin
        if (rst)
            cycle_count <= 0;
        else if (start || !done)
            cycle_count <= cycle_count + 1;
    end

    // Main test sequence
    initial begin
        $display("========================================");
        $display("5x5 QR Decomposition Testbench Started");
        $display("True 2D Systolic Array Implementation");
        $display("========================================");
        
        // Initialize signals
        rst = 1'b1;
        start = 1'b0;
        cycle_count = 0;
        
        // Initialize test matrix (default: well-conditioned)
        init_test_matrix();
        
        // Reset release
        #100 rst = 1'b0;
        $display("[TB] Reset released at time %0t", $time);
        
        // Display input matrix
        print_matrix_A();
        
        // Start QR decomposition
        #50 start = 1'b1;
        $display("[TB] QR decomposition started at time %0t", $time);
        #10 start = 1'b0;
        
        // Wait for completion
        $display("[TB] Waiting for completion...");
        wait(done === 1'b1);
        $display("[TB] QR decomposition completed at time %0t after %0d cycles", 
                 $time, cycle_count);
        
        // Display results
        print_matrix_R();
        print_givens_params();
        
        // Verification
        check_upper_triangular();
        check_r_diagonal_positive();
        
        $display("\n========================================");
        $display("Test completed successfully!");
        $display("Total simulation cycles: %0d", cycle_count);
        $display("========================================");
        
        #100 $finish;
    end

    //--------------------------------------------------------------------------
    // Optional: Run multiple test cases
    //--------------------------------------------------------------------------
    
    // Uncomment to test additional matrices
    /*
    initial begin
        #5000; // Wait for first test to complete
        
        $display("\n*** Running Identity Matrix Test ***");
        rst = 1'b1;
        init_identity_matrix();
        #50 rst = 1'b0;
        print_matrix_A();
        #50 start = 1'b1;
        #10 start = 1'b0;
        wait(done === 1'b1);
        print_matrix_R();
        check_upper_triangular();
        
        #1000;
        $display("\n*** Running Hilbert Matrix Test (Challenging) ***");
        rst = 1'b1;
        init_hilbert_matrix();
        #50 rst = 1'b0;
        print_matrix_A();
        #50 start = 1'b1;
        #10 start = 1'b0;
        wait(done === 1'b1);
        print_matrix_R();
        check_upper_triangular();
    end
    */

    //--------------------------------------------------------------------------
    // Debug Monitoring (if enabled)
    //--------------------------------------------------------------------------
    generate if (DEBUG) begin : DEBUG_MONITOR
        always @(posedge clk) begin
            if (!rst && (start || !done)) begin
                // Monitor key internal signals
                if (dut.state == dut.INPUT_DATA)
                    $display("[DEBUG] Cycle %0d: Inputting row %0d", cycle_count, dut.input_row);
                else if (dut.state == dut.PROCESSING)
                    $display("[DEBUG] Cycle %0d: Processing (state=%0d)", cycle_count, dut.state);
            end
        end
    end endgenerate

    //--------------------------------------------------------------------------
    // Waveform Dump
    //--------------------------------------------------------------------------
    initial begin
        $fsdbDumpfile("baseline_5x5_tb.fsdb");
        $fsdbDumpvars(0, baseline_5x5_tb);
        $fsdbDumpMDA();
    end

endmodule
