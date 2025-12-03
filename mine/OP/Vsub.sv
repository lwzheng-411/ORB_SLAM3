// Vector sub: o = a - b (FP32)
module Vsub (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] ax, ay, az,
    input  wire [31:0] bx, by, bz,
    output wire        out_vld,
    output wire [31:0] ox, oy, oz
);
    wire [7:0] st0, st1, st2;
    DW_fp_sub #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) subx (
        .a(ax), .b(bx), .rnd(3'b000), .z(ox), .status(st0)
    );
    DW_fp_sub #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) suby (
        .a(ay), .b(by), .rnd(3'b000), .z(oy), .status(st1)
    );
    DW_fp_sub #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) subz (
        .a(az), .b(bz), .rnd(3'b000), .z(oz), .status(st2)
    );
    assign out_vld = in_vld;
endmodule


