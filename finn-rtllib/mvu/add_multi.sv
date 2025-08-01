/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Pipelined multi-input adder tree.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

module add_multi import mvu_pkg::*; #(
	int unsigned  N,     // Number of Addends
	int unsigned  DEPTH, // Pipeline Depth
	int unsigned  ARG_WIDTH,
	int  ARG_LO = 0,
	int  ARG_HI = 0,
	bit  RESET_ZERO = 1,

	localparam int unsigned  SUM_WIDTH = sumwidth(N, ARG_WIDTH, ARG_LO, ARG_HI)
)(
	input	logic  clk,
	input	logic  rst,
	input	logic  en,

	input	logic [ARG_WIDTH-1:0]  arg[N],
	output	logic [SUM_WIDTH-1:0]  sum
);

//---------------------------------------------------------------------------
// Implementation Choice
typedef enum {
	LOOP,	// Combinatorial addition in loop followed by delay regs
	TREE,	// Binary adder tree with balanced pipelining as allowed by DEPTH
	COMP	// Use fitting pre-built compressor if available, fallback to TREE otherwise
} impl_e;
localparam impl_e  IMPL =
`ifdef ADD_MULTI_IMPL
	1? `ADD_MULTI_IMPL :
`endif
	TREE;

//===========================================================================
// Catch all Cases with matching pre-built Compressors if IMPL == COMP

`define CATCH_COMP(n,w,d) \
else if((IMPL == COMP) && !RESET_ZERO && (N == n) && (ARG_WIDTH == w) && (DEPTH >= d) && (0 <= ARG_LO)) begin : gen``n``u``w``_d``d \
	initial $info("Building add_multi(N=%0d, D=%0d, W=%0d) as COMP.", N, DEPTH, ARG_WIDTH); \
\
	uwire [N*ARG_WIDTH-1:0]  in; \
	uwire [SUM_WIDTH  -1:0]  out; \
	for(genvar  i = 0; i < N; i++) begin \
		for(genvar  j = 0; j < ARG_WIDTH; j++) begin \
			assign	in[j*N+i] = arg[i][j]; \
		end \
	end \
	comp``n``u``w``_d``d comp ( \
		.clk, .en, \
		.in, .out \
	); \
\
	localparam int unsigned  SUM_DELAY = DEPTH - d; \
	if(SUM_DELAY == 0)  assign  sum = out; \
	else begin : genDelay \
		logic [SUM_WIDTH-1:0]  SumZ[SUM_DELAY] = '{ default: 'x }; \
		always_ff @(posedge clk) begin \
			if(rst)  SumZ <= '{ default: 'x }; \
			else begin \
				for(int unsigned  i = 0; i < SUM_DELAY-1; i++)  SumZ[i] <= SumZ[i+1]; \
				SumZ[SUM_DELAY-1] <= out; \
			end \
		end \
		assign	sum = SumZ[0]; \
	end : genDelay \
end : gen``n``u``w``_d``d

if(0) begin end
//- Pre-built Compressor Matching -------
// N = 32, D = 4+
`CATCH_COMP(32,6,4)
`CATCH_COMP(32,16,4)
// N = 47, D = 4+
`CATCH_COMP(47,5,4)
`CATCH_COMP(47,6,4)
`CATCH_COMP(47,15,4)
// N = 56, D = 5+
`CATCH_COMP(56,7,5)
`CATCH_COMP(56,8,5)
`CATCH_COMP(56,17,5)
// N = 60, D = 5+
`CATCH_COMP(60,8,5)
`CATCH_COMP(60,18,5)
// N = 72, D = 5+
`CATCH_COMP(72,7,5)
`CATCH_COMP(72,8,5)
`CATCH_COMP(72,17,5)
`undef CATCH_COMP

//- Generic Behavioral Addition ---------
else begin : genGeneric

	// Number of tree levels with reductions
	localparam int unsigned  L = (IMPL == LOOP)? 0 : $clog2(N);

	logic [SUM_WIDTH-1:0]  sum0;
	if(L < 1) begin : genLoop
		initial $info("Building add_multi(N=%0d, D=%0d, W=%0d) as LOOP.", N, DEPTH, ARG_WIDTH);

		always_comb begin
			sum0 = 0;
			foreach(arg[i])  sum0 += arg[i];
		end
	end : genLoop
	else begin : genTree
		initial $info("Building add_multi(N=%0d, D=%0d, W=%0d) as TREE.", N, DEPTH, ARG_WIDTH);

		localparam int unsigned  D = L < DEPTH? L : DEPTH;  // Pipeline stages absorbed by tree

		// Compute the count of decendents for all nodes in the reduction trees.
		typedef int unsigned  leaf_load_t[2*N-1];
		function leaf_load_t init_leaf_loads();
			automatic leaf_load_t  res;
			for(int unsigned  i = 2*N-1; i-- > N-1;)  res[i] = 1;
			for(int unsigned  i =   N-1; i-- >   0;)  res[i] = res[2*i+1] + res[2*i+2];
			return  res;
		endfunction : init_leaf_loads
		localparam leaf_load_t   LEAF_LOAD = init_leaf_loads();

		// Adder Tree
		localparam bit  SIGNED = ARG_LO < 0;
		uwire signed [SUM_WIDTH-1:0]  tree[2*N-1];

		for(genvar  i = 0; i < 2*N-1; i++) begin
			localparam int unsigned  SW = sumwidth(LEAF_LOAD[i], ARG_WIDTH, ARG_LO, ARG_HI);
			uwire [SW-1:0]  s;

			if(N-1 <= i) begin : genLeave
				assign	s = arg[i - (N-1)];
			end : genLeave
			else begin : genReduce
				localparam int unsigned  LIDX = 2*i+1;
				localparam int unsigned  RIDX = 2*i+2;
				localparam int unsigned  LW = sumwidth(LEAF_LOAD[LIDX], ARG_WIDTH, ARG_LO, ARG_HI);
				localparam int unsigned  RW = sumwidth(LEAF_LOAD[RIDX], ARG_WIDTH, ARG_LO, ARG_HI);

				uwire [LW-1:0]  l = tree[LIDX];
				uwire [RW-1:0]  r = tree[RIDX];
				if(!SIGNED)  assign  s = l + r;
				else begin
					uwire signed [SW-1:0]  s0 = $signed(l) + $signed(r);
					assign	s = s0;
				end
			end : genReduce

			localparam int unsigned  TREE_LEVEL = $clog2(i+2) - 1;
			localparam int unsigned  STEP = L - TREE_LEVEL;
			localparam bit  REG = (TREE_LEVEL < L) && ((STEP*D / L) < ((STEP+1)*D / L));
			if(REG) begin : genReg
				localparam logic [SW-1:0]  S_RESET = {(SW){RESET_ZERO? 1'b0 : 1'bx}};
				logic [SW-1:0]  S = S_RESET;
				always_ff @(posedge clk) begin
					if(rst)      S <= S_RESET;
					else if(en)  S <= s;
				end
				assign  tree[i] = $signed({ SIGNED && S[SW-1], S} );
			end : genReg
			else  assign  tree[i] = $signed({ SIGNED && s[SW-1], s} );
		end

		assign	sum0 = tree[0];

	end : genTree

	// Delay Output if requested DEPTH exceeds Tree Height
	if(DEPTH <= L)  assign  sum = sum0;
	else begin : genDelay
		localparam logic [SUM_WIDTH-1:0]  SUM_RESET = {(SUM_WIDTH){RESET_ZERO? 1'b0 : 1'bx}};
		logic [SUM_WIDTH-1:0]  SumZ[DEPTH - L] = '{ default: SUM_RESET };
		always_ff @(posedge clk) begin
			if(rst)  SumZ <= '{ default: SUM_RESET };
			else begin
				for(int unsigned  i = 0; i < DEPTH-L-1; i++)  SumZ[i] <= SumZ[i+1];
				SumZ[DEPTH-L-1] <= sum0;
			end
		end
		assign	sum = SumZ[0];
	end : genDelay

end : genGeneric

endmodule : add_multi
