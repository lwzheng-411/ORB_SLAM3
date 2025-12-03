// VecNorm: 3D vector Euclidean norm
// norm = sqrt(x^2 + y^2 + z^2)
module VecNorm (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] x, y, z,
    output wire        out_vld,
    output wire [31:0] norm
);
    wire [31:0] norm_sq;
    
    // Compute squared norm via dot product with itself
    VecDot u_dot (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .ax(x), .ay(y), .az(z),
        .bx(x), .by(y), .bz(z),
        .dot(norm_sq), .out_vld()
    );
    
    // Take square root
    PE_fp_sqrt u_sqrt (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(norm_sq), .z(norm), .out_vld(out_vld)
    );
endmodule

