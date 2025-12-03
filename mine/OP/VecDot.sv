// VecDot: 3D vector dot product
// dot = a·b = ax*bx + ay*by + az*bz
module VecDot (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] ax, ay, az,
    input  wire [31:0] bx, by, bz,
    output wire        out_vld,
    output wire [31:0] dot
);
    wire [31:0] px, py, pz, s01;
    
    // Three parallel multiplications
    PE_fp_mul mul_x (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(ax), .b(bx), .z(px), .out_vld()
    );
    PE_fp_mul mul_y (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(ay), .b(by), .z(py), .out_vld()
    );
    PE_fp_mul mul_z (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(az), .b(bz), .z(pz), .out_vld()
    );
    
    // Two-stage addition tree
    PE_fp_add add_01 (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(px), .b(py), .z(s01), .out_vld()
    );
    PE_fp_add add_2 (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(s01), .b(pz), .z(dot), .out_vld(out_vld)
    );
endmodule

