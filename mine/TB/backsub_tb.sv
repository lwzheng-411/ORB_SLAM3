`timescale 1ns/1ps

module backsub_tb;
    localparam int N = 6;

    logic clk;
    logic rst;
    logic start;
    logic mode;

    logic [31:0] r_in [0:N-1][0:N-1];
    logic [31:0] b_in [0:N-1];
    wire  [31:0] x_out [0:N-1];
    wire         done;

    initial clk = 1'b0;
    always #5 clk = ~clk;

    backsub #(
        .N(N)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .mode(mode),
        .r_in(r_in),
        .b_in(b_in),
        .x_out(x_out),
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
    function automatic real from_fp32(input [31:0] bits);
        begin
            from_fp32 = $bitstoshortreal(bits);
        end
    endfunction

    task automatic init_inputs();
        // Initialize R matrix: upper triangular, diagonal = 10.0, off-diagonal = 1.0
        // Initialize b vector: [1.0, 2.0, 3.0, ...]
        for (int r = 0; r < N; r++) begin
            b_in[r] = to_fp32(r + 1.0);
            for (int c = 0; c < N; c++) begin
                if (r == c) begin
                    r_in[r][c] = to_fp32(10.0); // diagonal
                end else if (r < c) begin
                    r_in[r][c] = to_fp32(1.0); // upper triangle
                end else begin
                    r_in[r][c] = 32'h0; // lower triangle (should be 0)
                end
            end
        end
    endtask

    initial begin
        rst = 1'b1;
        start = 1'b0;
        mode  = 1'b1; // back substitution mode
        init_inputs();

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (done);
        repeat (2) @(posedge clk);

        $display("\n========================================");
        $display("backsub_tb (N=%0d): Back-Substitution Results", N);
        $display("========================================");
        $display("Input R matrix (upper triangular):");
        for (int r = 0; r < N; r++) begin
            $write("  R[%0d] = [", r);
            for (int c = 0; c < N; c++) begin
                $write("%8.4f", from_fp32(r_in[r][c]));
                if (c < N-1) $write(", ");
            end
            $display("]");
        end
        $display("\nInput b vector:");
        $write("  b = [");
        for (int r = 0; r < N; r++) begin
            $write("%8.4f", from_fp32(b_in[r]));
            if (r < N-1) $write(", ");
        end
        $display("]");
        $display("\nSolution x:");
        for (int r = 0; r < N; r++) begin
            $display("  x[%0d] = %8.4f", r, from_fp32(x_out[r]));
        end
        $display("\nbacksub_tb completed (done = %0b)", done);

        $finish;
    end

endmodule

