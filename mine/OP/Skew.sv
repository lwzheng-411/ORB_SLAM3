// Skew-symmetric matrix generator: K = [v]^x
// K = [[  0, -z,  y],
//      [  z,  0, -x],
//      [ -y,  x,  0]]
module Skew (
    input  wire [31:0] x,
    input  wire [31:0] y,
    input  wire [31:0] z,
    output wire [31:0] K [0:2][0:2]
);
    function automatic [31:0] negf(input [31:0] a);
        negf = {~a[31], a[30:0]};
    endfunction

    // Row 0
    assign K[0][0] = 32'h00000000;
    assign K[0][1] = negf(z);
    assign K[0][2] = y;
    // Row 1
    assign K[1][0] = z;
    assign K[1][1] = 32'h00000000;
    assign K[1][2] = negf(x);
    // Row 2
    assign K[2][0] = negf(y);
    assign K[2][1] = x;
    assign K[2][2] = 32'h00000000;
endmodule


