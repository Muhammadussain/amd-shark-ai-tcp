// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Data-tiled Q5_0 matmul with transposed RHS: out[b,m,n] = sum_k a[b,m,k] * w[n,k].
//
// Template parameters: n, k, group0 (= k / 32) and the tile sizes m0, n0, k0
// (k0 must be a multiple of 32, so a K tile holds kb = k0 / 32 whole blocks).
//
// Inputs are the planes exactly as stored in the IRPA:
//   a  : [B, M, K]    activations (fp16/fp32)
//   d  : [N, G, 1]    scales, one per block of 32
//   qh : [N, G, 4]    u8, bit j of the LE uint32 = 5th bit of element j
//   qs : [N, G, 16]   u8, byte j = element j (lo nibble) | element j+16 (hi)
//
// Steps:
//   1. collapse to 2-D:  a -> [B*M, K], d -> [N, G], qh -> [N, G*4], qs -> [N, G*16]
//   2. linalg.pack all four inputs (data movement only, zero padded):
//        a  -> [M1, K1, M0, K0]
//        d  -> [N1, K1, N0, KB]
//        qh -> [N1, K1, N0, KB*4]
//        qs -> [N1, K1, N0, KB*16]
//   3. one linalg.generic over the packed operands. Dequantization
//      (w = d * ((hibit << 4 | nibble) - 16)) happens inline in the MAC.
//   4. linalg.unpack the packed [M1, N1, M0, N0] result, cast, expand to [B, M, N].
// Zero padding is harmless: padded activations are 0 and padded scales are 0.

{% set accum_type = "f32" %}
{% set kb = k0 // 32 %}
{% set n1 = (n + n0 - 1) // n0 %}
{% set k1 = (group0 + kb - 1) // kb %}

!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<?x?x{{k}}x!a_type>
!d_tensor_type = tensor<{{n}}x{{group0}}x1x!scale_type>
!qh_tensor_type = tensor<{{n}}x{{group0}}x4xi8>
!qs_tensor_type = tensor<{{n}}x{{group0}}x16xi8>
!c_tensor_type = tensor<?x?x{{n}}x!a_type>

module {

util.func private @amdsharktank_mmt_block_scaled_q5_0_tiled_3d_{{n}}_{{k}}_{{m0}}x{{n0}}x{{k0}}_{{a_type}}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qh: !qh_tensor_type, %qs: !qs_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.0 : !accum_type
  %zero_a = arith.constant 0.0 : !a_type
  %zero_s = arith.constant 0.0 : !scale_type
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c_m0 = arith.constant {{m0}} : index
  %batch = tensor.dim %a, %c0 : !a_tensor_type
  %m = tensor.dim %a, %c1 : !a_tensor_type
  %bm = arith.muli %batch, %m : index
  %m1 = arith.ceildivui %bm, %c_m0 : index

  // 1. Collapse to 2-D (pure reshapes, no data change).
  %a2 = tensor.collapse_shape %a [[0, 1], [2]] : !a_tensor_type into tensor<?x{{k}}x!a_type>
  %d2 = tensor.collapse_shape %d [[0], [1, 2]] : !d_tensor_type into tensor<{{n}}x{{group0}}x!scale_type>
  %qh2 = tensor.collapse_shape %qh [[0], [1, 2]] : !qh_tensor_type into tensor<{{n}}x{{group0 * 4}}xi8>
  %qs2 = tensor.collapse_shape %qs [[0], [1, 2]] : !qs_tensor_type into tensor<{{n}}x{{group0 * 16}}xi8>

  // 2. Pack all four inputs.
  %a_dest = tensor.empty(%m1) : tensor<?x{{k1}}x{{m0}}x{{k0}}x!a_type>
  %a_pack = linalg.pack %a2 padding_value(%zero_a : !a_type)
      inner_dims_pos = [0, 1] inner_tiles = [{{m0}}, {{k0}}] into %a_dest
      : tensor<?x{{k}}x!a_type> -> tensor<?x{{k1}}x{{m0}}x{{k0}}x!a_type>

  %d_dest = tensor.empty() : tensor<{{n1}}x{{k1}}x{{n0}}x{{kb}}x!scale_type>
  %d_pack = linalg.pack %d2 padding_value(%zero_s : !scale_type)
      inner_dims_pos = [0, 1] inner_tiles = [{{n0}}, {{kb}}] into %d_dest
      : tensor<{{n}}x{{group0}}x!scale_type> -> tensor<{{n1}}x{{k1}}x{{n0}}x{{kb}}x!scale_type>

  %qh_dest = tensor.empty() : tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 4}}xi8>
  %qh_pack = linalg.pack %qh2 padding_value(%zero_i8 : i8)
      inner_dims_pos = [0, 1] inner_tiles = [{{n0}}, {{kb * 4}}] into %qh_dest
      : tensor<{{n}}x{{group0 * 4}}xi8> -> tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 4}}xi8>

  %qs_dest = tensor.empty() : tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 16}}xi8>
  %qs_pack = linalg.pack %qs2 padding_value(%zero_i8 : i8)
      inner_dims_pos = [0, 1] inner_tiles = [{{n0}}, {{kb * 16}}] into %qs_dest
      : tensor<{{n}}x{{group0 * 16}}xi8> -> tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 16}}xi8>

  // 3. GEMM on the packed operands, dequantizing inline.
  %c_empty = tensor.empty(%m1) : tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type>
  %c_fill = linalg.fill ins(%zero : !accum_type)
      outs(%c_empty : tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type>) -> tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type>
  %c_pack = linalg.generic {
      indexing_maps = [
          // d0 = m1, d1 = n1, d2 = k1 (r), d3 = m0, d4 = n0, d5 = kk in [0, K0) (r)
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5 floordiv 32)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5 floordiv 8)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, (d5 floordiv 32) * 16 + d5 mod 16)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>],
      iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"] }
      ins(%a_pack, %d_pack, %qh_pack, %qs_pack
          : tensor<?x{{k1}}x{{m0}}x{{k0}}x!a_type>,
            tensor<{{n1}}x{{k1}}x{{n0}}x{{kb}}x!scale_type>,
            tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 4}}xi8>,
            tensor<{{n1}}x{{k1}}x{{n0}}x{{kb * 16}}xi8>)
      outs(%c_fill : tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type>) {
  ^bb0(%a_element: !a_type, %d_element: !scale_type, %qh_element: i8, %qs_element: i8, %out: !accum_type):
      %kk = linalg.index 5 : index
      %kk_i32 = arith.index_cast %kk : index to i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %c7_i32 = arith.constant 7 : i32
      %c15_i32 = arith.constant 15 : i32
      %c16_i32 = arith.constant 16 : i32

      // Low nibble: element j = kk mod 32; j < 16 -> low nibble, j >= 16 -> high.
      %hsel = arith.shrui %kk_i32, %c4_i32 : i32
      %h = arith.andi %hsel, %c1_i32 : i32
      %nib_shift = arith.muli %h, %c4_i32 : i32
      %qs_i32 = arith.extui %qs_element : i8 to i32
      %qs_shifted = arith.shrui %qs_i32, %nib_shift : i32
      %low4 = arith.andi %qs_shifted, %c15_i32 : i32

      // 5th bit: bit (kk mod 8) of the qh byte.
      %bit_pos = arith.andi %kk_i32, %c7_i32 : i32
      %qh_i32 = arith.extui %qh_element : i8 to i32
      %qh_shifted = arith.shrui %qh_i32, %bit_pos : i32
      %hbit = arith.andi %qh_shifted, %c1_i32 : i32
      %hbit_up = arith.shli %hbit, %c4_i32 : i32

      %q = arith.ori %low4, %hbit_up : i32
      %q_signed = arith.subi %q, %c16_i32 : i32
      %q_fp = arith.sitofp %q_signed : i32 to !accum_type
    {% if scale_type == accum_type %}
      %d_f32 = arith.addf %d_element, %zero : !accum_type
    {% else %}
      %d_f32 = arith.extf %d_element : !scale_type to !accum_type
    {% endif %}
      %w = arith.mulf %q_fp, %d_f32 : !accum_type

    {% if a_type == accum_type %}
      %a_f32 = arith.addf %a_element, %zero : !accum_type
    {% else %}
      %a_f32 = arith.extf %a_element : !a_type to !accum_type
    {% endif %}
      %prod = arith.mulf %a_f32, %w : !accum_type
      %acc = arith.addf %prod, %out : !accum_type
      linalg.yield %acc : !accum_type
  } -> tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type>

  // 4. Unpack, cast to the activation type, restore [B, M, N].
  %c_un_empty = tensor.empty(%bm) : tensor<?x{{n}}x!accum_type>
  %c_un = linalg.unpack %c_pack
      inner_dims_pos = [0, 1] inner_tiles = [{{m0}}, {{n0}}] into %c_un_empty
      : tensor<?x{{n1}}x{{m0}}x{{n0}}x!accum_type> -> tensor<?x{{n}}x!accum_type>
  %c_cast_empty = tensor.empty(%bm) : tensor<?x{{n}}x!a_type>
  %c_cast = linalg.copy
      ins(%c_un : tensor<?x{{n}}x!accum_type>)
      outs(%c_cast_empty : tensor<?x{{n}}x!a_type>) -> tensor<?x{{n}}x!a_type>
  %result = tensor.expand_shape %c_cast [[0, 1], [2]] output_shape [%batch, %m, {{n}}]
      : tensor<?x{{n}}x!a_type> into !c_tensor_type
  util.return %result : !c_tensor_type
}

}
