// Atan2Cordic: compute theta = atan2(y, x) in FP32 using iterative CORDIC-like pipeline
// Simplified placeholder: uses DW_fp_atan2 if available; otherwise expects integration later
module Atan2Cordic (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] y,
    input  wire [31:0] x,
    output wire        out_vld,
    output wire [31:0] theta
);
`ifdef HAS_DW_ATAN2
    wire [7:0] status;
    DW_fp_atan2 #(
        .sig_width(23), .exp_width(8), .ieee_compliance(1)
    ) u_atan2 (
        .a(y), .b(x), .z(theta), .status(status)
    );
    assign out_vld = in_vld;
`else
    // Temporary pass-through using DW_fp_div + DW_fp_atan for y/x
    wire [31:0] ratio; wire v1;
    PE_fp_div u_div(.clk(clk), .rstn(rstn), .in_vld(in_vld), .a(y), .b(x), .out_vld(v1), .z(ratio));
    // Use polynomial or DW_fp_atan if available. Here assume DW_fp_atan exists.
    wire [7:0] st_a; wire [31:0] at;
    DW_fp_atan #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u_at(.a(ratio), .z(at), .status(st_a));
    assign theta = at;
    assign out_vld = v1;
`endif
endmodule


