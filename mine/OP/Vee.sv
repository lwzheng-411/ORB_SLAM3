// Vee: Extract vector from skew-symmetric matrix
// Given K = [v]^x, recover v = [K[2][1], K[0][2], K[1][0]]
module Vee (
    input  wire [31:0] K [0:2][0:2],
    output wire [31:0] x, y, z
);
    // Combinational extraction (zero latency)
    assign x = K[2][1];
    assign y = K[0][2];
    assign z = K[1][0];
endmodule

