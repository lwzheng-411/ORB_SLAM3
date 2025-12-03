// PE_fp_sqrt: FP32 square-root PE with valid handshake
module PE_fp_sqrt (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] a,
    output wire        out_vld,
    output wire [31:0] z
);
    wire [7:0] status;
    DW_fp_sqrt #(.sig_width(23), .exp_width(8), .ieee_compliance(1)) u_sqrt (
        .a(a), .rnd(3'b000), .z(z), .status(status)
    );
    assign out_vld = in_vld;
endmodule


