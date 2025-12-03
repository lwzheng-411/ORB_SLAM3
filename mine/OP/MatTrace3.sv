// MatTrace3: Trace of 3x3 matrix
// trace = R[0][0] + R[1][1] + R[2][2]
module MatTrace3 (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] R [0:2][0:2],
    output wire        out_vld,
    output wire [31:0] trace
);
    wire [31:0] s01;
    
    // Two-stage addition of diagonal elements
    PE_fp_add add_01 (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(R[0][0]), .b(R[1][1]), .z(s01), .out_vld()
    );
    PE_fp_add add_2 (
        .clk(clk), .rstn(rstn), .in_vld(in_vld),
        .a(s01), .b(R[2][2]), .z(trace), .out_vld(out_vld)
    );
endmodule

