// fp16 multiplier module
module fp16_mult (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [15:0] a,
    input   wire [15:0] b,
    output  wire [15:0] c,
    output  wire [7:0] status
);

    // c=a*b
    DW_fp_mult_DG #(
        .sig_width(10),        // fp16 mantissa width
        .exp_width(5),         // fp16 exponent width
        .ieee_compliance(1)
    ) u_mult (
        .a(a),
        .b(b),
        .rnd(3'b000),
        .DG_ctrl(enable),
        .z(c),
        .status(status)
    );
endmodule

// int8 multiplier module
module int8_mult (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [7:0] a,
    input   wire [7:0] b,
    output  wire [15:0] c,
    output  wire [7:0] status
);

    // c=a*b
    DW_mult_pipe #(
        .a_width(8),
        .b_width(8),
        .num_stages(2),
        .stall_mode(0),
        .rst_mode(0),
        .op_iso_mode(0)
    ) u_mult (
        .clk(clk),
        .rst_n(~rst),
        .en(enable),
        .tc(1'b1),
        .a(a),
        .b(b),
        .product(c)
    );
    
    assign status = 8'h0;  // int8 multiplier doesn't have status output
endmodule
