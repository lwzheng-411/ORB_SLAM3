// PE_fp_add: FP32 adder PE with valid handshake
module PE_fp_add (
    input  wire        clk,
    input  wire        rstn,
    input  wire        in_vld,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire        out_vld,
    output wire [31:0] z
);
    wire [7:0] status;
    DW_fp_add #(
        .sig_width(23), 
        .exp_width(8), 
        .ieee_compliance(1)
        ) u_add (
        .a(a), 
        .b(b), 
        .rnd(3'b000), 
        .z(z), 
        .status(status)
    );
    assign out_vld = in_vld; // pipeline latency abstracted by scheduler
endmodule


