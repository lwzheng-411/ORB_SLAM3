// PE_fp_mul: FP32 multiplier PE with valid handshake
module PE_fp_mul (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_vld,
    output wire [31:0] z
);
    wire [7:0] status;
    DW_fp_mult_DG #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u_mul (
        .a(a), .b(b), .rnd(3'b000), .DG_ctrl(1'b1), .z(z), .status(status)
    );
    assign out_vld = in_vld;
endmodule


