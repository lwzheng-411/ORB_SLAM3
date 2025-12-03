// VecScale: Scalar multiplication of 3D vector
// out = scalar * v (element-wise)
module VecScale (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] scalar,
    input  wire [31:0] vx, vy, vz,
    output wire        out_vld,
    output wire [31:0] ox, oy, oz
);
    PE_fp_mul mul_x (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(scalar), .b(vx), .out_vld(), .z(ox)
    );
    PE_fp_mul mul_y (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(scalar), .b(vy), .out_vld(), .z(oy)
    );
    PE_fp_mul mul_z (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(scalar), .b(vz), .out_vld(), .z(oz)
    );
    assign out_vld = in_vld;
endmodule

