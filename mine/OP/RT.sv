// RT: Matrix Transpose for 3x3 Rotation Matrix
// R_T = R^T (simply rearrange elements)
module RT (
    input clk,
    input rstn,
    input in_vld,
    input [31:0] R [0:2][0:2], // Input rotation matrix R[row][col]
    output [31:0] RT_out [0:2][0:2], // Output transpose R_T[col][row]
    output out_vld
);

    // Transpose: R_T[i][j] = R[j][i]
    genvar i, j;
    generate
        for (i = 0; i < 3; i = i + 1) begin : RT_ROW
            for (j = 0; j < 3; j = j + 1) begin : RT_COL
                assign RT_out[i][j] = R[j][i];
            end
        end
    endgenerate

    // Combinational, no latency
    assign out_vld = in_vld;

endmodule

