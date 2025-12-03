// PE_fp_sub: FP32 subtractor PE with valid handshake
module PE_fp_sub (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_vld,
    output wire [31:0] z
);
    wire [7:0] status;
    DW_fp_sub #(
        .sig_width(23), 
        .exp_width(8), 
        .ieee_compliance(1)
        ) u_sub (
        .a(a), 
        .b(b), 
        .rnd(3'b000), 
        .z(z), 
        .status(status)
    );
    assign out_vld = in_vld;
endmodule


