// fp16 adder module
module fp16_add (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [15:0] a,
    input   wire [15:0] b,
    output  wire [15:0] c,
    output  wire [7:0] status
);

    // c=a+b
    DW_fp_add #(
        .sig_width(10),        // fp16 mantissa width
        .exp_width(5),         // fp16 exponent width
        .ieee_compliance(1)
    ) u_add (
        .a(a),
        .b(b),
        .rnd(3'b000),
        .z(c),
        .status(status)
    );
endmodule

// int8 adder module
module int8_add (
    input   wire clk,
    input   wire rst,
    input   wire enable,
    input   wire [7:0] a,
    input   wire [7:0] b,
    output  wire [7:0] c,
    output  wire       co
);

    // c=a+b
    DW01_add #(
        .width(8)
    ) u_add (
        .A(a),
        .B(b),
        .CI(1'b0),
        .SUM(c),
        .CO(co)
    );
endmodule


