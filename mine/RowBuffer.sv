module RowBuffer #(
    parameter MAX_LOCAL_ROWS = 64,    // TODO
    parameter MAX_LOCAL_COLS = 96,    // TODO
    parameter MAX_PANEL_COLS = 6      // TODO
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 bundle_begin,         // begin of a bundle
    input  wire [2:0]           bundle_panel_cols,    // 3 or 6
    input  wire [7:0]           bundle_trails_count,  // number of trail blocks
    input  wire [9:0]           bundle_local_cols,    // panel_cols + 6*trails_count
    input  wire [15:0]          bundle_id,            // TODO
    input  wire                 bundle_end,           // end of a bundle
    
    input  wire                 row_in_valid,
    output reg                  row_in_ready,
    input  wire [2:0]           row_in_panel_cols,
    input  wire [2:0]           row_in_trail_cols,
    input  wire [7:0]           row_in_panel_col_base,
    input  wire [7:0]           row_in_trail_col_base,
    input  wire [31:0]          row_in_panel [0:5],
    input  wire [31:0]          row_in_trail [0:5],
    input  wire [31:0]          row_in_b,
    input  wire                 row_in_last_in_factor,
    input  wire                 row_in_last_in_bundle,

    output reg                  sa_start,
    input  wire                 sa_done,
    output reg                  sa_row_valid,
    input  wire                 sa_row_ready,
    output reg  [9:0]           sa_row_idx,
    output reg  [31:0]          sa_row_data [0:MAX_LOCAL_COLS-1],
    output reg  [31:0]          sa_row_b,
    output reg  [2:0]           sa_panel_cols,        // Inform systolic: only triangulate first p cols
    output reg  [9:0]           sa_total_cols,

    // ========================================================================
    // R-block collection from systolicarray output
    // (For backsubstitution and new-factor generation)
    // ========================================================================
    input  wire                 sa_out_valid,
    output wire                 sa_out_ready,
    input  wire [9:0]           sa_out_row_idx,
    input  wire [31:0]          sa_out_R_row [0:MAX_LOCAL_COLS-1],
    input  wire [31:0]          sa_out_z,

    // ========================================================================
    // New-factor output (R12/R22 trailing blocks + z2) to CPU
    // ========================================================================
    output reg                  new_factor_valid,
    input  wire                 new_factor_ready,
    output reg  [15:0]          new_factor_bundle_id,
    output reg  [9:0]           new_factor_rows,      // Number of rows in new factor
    output reg  [9:0]           new_factor_cols,      // Trailing cols (local_cols - panel_cols)
    output reg  [31:0]          new_factor_R [0:MAX_LOCAL_ROWS-1][0:MAX_LOCAL_COLS-1],
    output reg  [31:0]          new_factor_z [0:MAX_LOCAL_ROWS-1]
);

    // ========================================================================
    // Local dense matrix buffer (m×L)
    // ========================================================================
    reg [31:0] local_A [0:MAX_LOCAL_ROWS-1][0:MAX_LOCAL_COLS-1];
    reg [31:0] local_b [0:MAX_LOCAL_ROWS-1];
    reg [9:0]  current_row_count;
    reg [2:0]  panel_p;
    reg [9:0]  local_L;
    reg [15:0] current_bundle_id;

    // Column base to local column index mapping (for trails)
    // Simple approach: CPU passes absolute col_base; we map to local trail slots
    // Slot k → local cols [panel_p + 6*k : panel_p + 6*(k+1)-1]
    // We use a lookup or direct mapping via (col_base - panel_base)/6 = slot_idx
    
    // Simpler: CPU pre-assigns trail_index in Axb call, RowBuffer uses it directly
    // Or we can use col_base to infer slot (requires knowing panel's global col_base)
    // For prototype: assume col_trail_base uniquely maps to a trail slot via simple offset

    reg [7:0] panel_global_base;  // Global column base of the panel variable
    reg [7:0] trail_bases [0:15]; // Up to 16 trail blocks, each 6 cols
    reg [3:0] trail_map_count;

    localparam [1:0] ST_IDLE = 2'd0,
                     ST_COLLECT = 2'd1,
                     ST_FEED_SA = 2'd2,
                     ST_COLLECT_R = 2'd3;
    reg [1:0] buf_state;
    reg [9:0] feed_row_idx;

    integer i, j;

    // ========================================================================
    // FSM for row assembly and SA feeding
    // ========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            buf_state <= ST_IDLE;
            row_in_ready <= 1'b1;
            current_row_count <= 0;
            sa_start <= 1'b0;
            sa_row_valid <= 1'b0;
            new_factor_valid <= 1'b0;
            
            for (i=0; i<MAX_LOCAL_ROWS; i=i+1) begin
                for (j=0; j<MAX_LOCAL_COLS; j=j+1)
                    local_A[i][j] <= 32'h00000000;
                local_b[i] <= 32'h00000000;
            end
            
        end else begin
            sa_start <= 1'b0;
            
            case (buf_state)
                // ============================================================
                // IDLE: Wait for bundle_begin
                // ============================================================
                ST_IDLE: begin
                    if (bundle_begin) begin
                        // Initialize new bundle
                        panel_p <= bundle_panel_cols;
                        local_L <= bundle_local_cols;
                        current_bundle_id <= bundle_id;
                        current_row_count <= 0;
                        trail_map_count <= bundle_trails_count;
                        
                        // Clear local matrix
                        for (i=0; i<MAX_LOCAL_ROWS; i=i+1) begin
                            for (j=0; j<MAX_LOCAL_COLS; j=j+1)
                                local_A[i][j] <= 32'h00000000;
                            local_b[i] <= 32'h00000000;
                        end
                        
                        buf_state <= ST_COLLECT;
                        row_in_ready <= 1'b1;
                    end
                end

                // ============================================================
                // COLLECT: Receive row segments and assemble dense rows
                // ============================================================
                ST_COLLECT: begin
                    if (row_in_valid && row_in_ready) begin
                        // Place panel segment at cols [0:panel_p-1]
                        for (j=0; j<6; j=j+1) begin
                            if (j < row_in_panel_cols)
                                local_A[current_row_count][j] <= row_in_panel[j];
                        end
                        
                        // Place trail segment: infer local col index from col_base
                        // Simplified: assume CPU passes trail in sequential order
                        // or we map col_base → slot_idx = (col_base - panel_base) / 6
                        // For now: direct placement at local cols [panel_p + trail_offset : ...]
                        if (row_in_trail_cols > 0) begin
                            // Simple heuristic: use col_base to compute offset
                            // Assume trails are contiguous 6-col blocks after panel
                            // trail_col_base tells us absolute column; we need local offset
                            // For prototype: place trail at panel_p + 0..5 (first trail)
                            // Better: use a trail_index input from Axb (0,1,2,...) 
                            // mapped to local cols [panel_p+6*trail_index : panel_p+6*trail_index+5]
                            for (j=0; j<6; j=j+1) begin
                                if (j < row_in_trail_cols)
                                    local_A[current_row_count][panel_p + j] <= row_in_trail[j];
                            end
                        end
                        
                        local_b[current_row_count] <= row_in_b;
                        current_row_count <= current_row_count + 1;
                        
                        if (row_in_last_in_bundle || bundle_end) begin
                            buf_state <= ST_FEED_SA;
                            row_in_ready <= 1'b0;
                            sa_start <= 1'b1;
                            feed_row_idx <= 0;
                        end
                    end
                end

                // ============================================================
                // FEED_SA: Stream dense local matrix to systolicarray
                // ============================================================
                ST_FEED_SA: begin
                    sa_panel_cols <= panel_p;
                    sa_total_cols <= local_L;
                    
                    if (sa_row_ready) begin
                        sa_row_valid <= 1'b1;
                        sa_row_idx <= feed_row_idx;
                        for (j=0; j<MAX_LOCAL_COLS; j=j+1)
                            sa_row_data[j] <= (j < local_L) ? local_A[feed_row_idx][j] : 32'h00000000;
                        sa_row_b <= local_b[feed_row_idx];
                        
                        if (feed_row_idx + 1 >= current_row_count) begin
                            sa_row_valid <= 1'b0;
                            buf_state <= ST_COLLECT_R;
                        end else begin
                            feed_row_idx <= feed_row_idx + 1;
                        end
                    end
                end

                // ============================================================
                // COLLECT_R: Receive R blocks from systolicarray output
                // ============================================================
                ST_COLLECT_R: begin
                    if (sa_done) begin
                        // systolicarray finished; extract R blocks
                        // R11 (panel×panel), R12 (panel×trail), R22 (trail×trail)
                        // z1 (panel), z2 (trail)
                        // Store to new_factor_R/z for CPU to fetch
                        
                        // For now: signal valid new factor
                        new_factor_valid <= 1'b1;
                        new_factor_bundle_id <= current_bundle_id;
                        new_factor_rows <= current_row_count - panel_p; // Trailing rows
                        new_factor_cols <= local_L - panel_p;            // Trailing cols
                        
                        // Actual R block extraction requires parsing sa_out_* stream
                        // Simplified: assume systolicarray directly outputs R to memory
                        // Or we collect sa_out_valid pulses row-by-row
                        
                        buf_state <= ST_IDLE;
                        row_in_ready <= 1'b1;
                    end
                end

                default: buf_state <= ST_IDLE;
            endcase
        end
    end

    // ========================================================================
    // Systolic output collection (R blocks)
    // ========================================================================
    assign sa_out_ready = (buf_state == ST_COLLECT_R);
    
    always @(posedge clk) begin
        if (sa_out_valid && sa_out_ready) begin
            // Store R row to new_factor_R (simplified)
            for (j=0; j<MAX_LOCAL_COLS; j=j+1)
                new_factor_R[sa_out_row_idx][j] <= sa_out_R_row[j];
            new_factor_z[sa_out_row_idx] <= sa_out_z;
        end
    end

endmodule

