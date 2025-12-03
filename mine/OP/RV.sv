// RV: 3x3 Matrix-Vector Multiplication using 3x1 Systolic Array
// y = R * v, where R is 3x3, v is 3x1
// Uses 3 PEs (one per output element) with multiply-accumulate
module RV (
    input clk,
    input rstn,
    input start,
    input [31:0] R [0:2][0:2], // R matrix, R[row][col]
    input [31:0] v [0:2],      // v vector
    output reg done,
    output [31:0] y [0:2]
);

    // State machine for systolic array control
    localparam S_IDLE = 2'b00, S_COMPUTE = 2'b01, S_FLUSH = 2'b10, S_DONE = 2'b11;
    reg [1:0] state;
    reg [1:0] col_idx; // Current column being processed (0-2)
    reg [1:0] flush_cnt;

    // PE internal signals: 3 PEs for 3 output elements
    wire [31:0] mul_out [0:2];
    wire [31:0] add_out [0:2];
    reg [31:0] acc_reg [0:2]; // Accumulator for each PE

    // Instantiate 3 PEs (one per row of R)
    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : PE_INST
            // Multiplier: R[i][col_idx] * v[col_idx]
            DW_fp_mult_DG #(
                .sig_width(23),
                .exp_width(8),
                .ieee_compliance(1)
            ) u_mul (
                .a(R[i][col_idx]),
                .b(v[col_idx]),
                .rnd(3'b000),
                .DG_ctrl(1'b1),
                .z(mul_out[i]),
                .status()
            );

            // Adder: acc_reg[i] + mul_out[i]
            DW_fp_add #(
                .sig_width(23),
                .exp_width(8),
                .ieee_compliance(1)
            ) u_add (
                .a(acc_reg[i]),
                .b(mul_out[i]),
                .rnd(3'b000),
                .z(add_out[i]),
                .status()
            );
        end
    endgenerate

    // Control FSM
    integer j;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state <= S_IDLE;
            col_idx <= 2'b00;
            flush_cnt <= 2'b00;
            done <= 1'b0;
            for (j = 0; j < 3; j = j + 1) acc_reg[j] <= 32'h00000000; // 0.0
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state <= S_COMPUTE;
                        col_idx <= 2'b00;
                        flush_cnt <= 2'b00;
                        // Reset accumulators
                        for (j = 0; j < 3; j = j + 1) acc_reg[j] <= 32'h00000000;
                    end
                end

                S_COMPUTE: begin
                    if (col_idx < 3) begin
                        // Accumulate: acc_reg[i] += R[i][col_idx] * v[col_idx]
                        for (j = 0; j < 3; j = j + 1) acc_reg[j] <= add_out[j];
                        col_idx <= col_idx + 1;
                    end else begin
                        state <= S_FLUSH;
                        flush_cnt <= 2'b00;
                    end
                end

                S_FLUSH: begin
                    // Wait for pipeline flush (mul + add latency)
                    if (flush_cnt < 2) begin
                        flush_cnt <= flush_cnt + 1;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Output assignment
    assign y[0] = acc_reg[0];
    assign y[1] = acc_reg[1];
    assign y[2] = acc_reg[2];

endmodule
