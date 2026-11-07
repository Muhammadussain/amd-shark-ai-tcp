// // Copyright 2025 Advanced Micro Devices, Inc.
// //
// // Licensed under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// // Untiled Q5_0 matmul with transposed RHS: out[b,m,n] = sum_k a[b,m,k] * w[n,k].
// //
// // The Q5_0 weight is given as its three GGML planes (exactly as stored in IRPA):
// //   d  : [N, G, 1]  f16  scale per block of 32 (G = K / 32)
// //   qh : [N, G, 4]  u8   32 high bits, bit j of the LE uint32 = element j
// //   qs : [N, G, 16] u8   low nibbles, byte j = elem j (lo) | elem j+16 (hi)
// //
// // Element j of a block is split as j = h*16 + s*8 + k (h in 0..1, s in 0..1,
// // k in 0..7). Then:
// //   low nibble = (qs[s*8 + k] >> (4*h)) & 0xF
// //   high bit   = (qh[h*2 + s] >> k) & 1
// //   w          = d * (((high bit << 4) | low nibble) - 16)
// // Dequantization happens inside the multiply-accumulate generic below, so no
// // dequantized [N, K] weight tensor is ever created.

// {% set accum_type = "f32" %}

// !a_type = {{a_type}}
// !scale_type = {{scale_type}}
// !accum_type = {{accum_type}}
// !a_tensor_type = tensor<?x?x{{k}}x!a_type>
// !aexp_tensor_type = tensor<?x?x{{group0}}x2x2x8x!a_type>
// !d_tensor_type = tensor<{{n}}x{{group0}}x1x!scale_type>
// !qh_tensor_type = tensor<{{n}}x{{group0}}x4xi8>
// !qhexp_tensor_type = tensor<{{n}}x{{group0}}x2x2xi8>
// !qs_tensor_type = tensor<{{n}}x{{group0}}x16xi8>
// !qsexp_tensor_type = tensor<{{n}}x{{group0}}x2x8xi8>
// !accum_tensor_type = tensor<?x?x{{n}}x!accum_type>
// !c_tensor_type = tensor<?x?x{{n}}x!a_type>

// module {

// util.func private @amdsharktank_mmt_block_scaled_q5_0_3d_{{n}}_{{k}}_{{a_type}}(
//     %a: !a_tensor_type, %d: !d_tensor_type, %qh: !qh_tensor_type, %qs: !qs_tensor_type)
//     -> !c_tensor_type {
//   %zero = arith.constant 0.0: !accum_type
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %batch0 = tensor.dim %a, %c0 : !a_tensor_type
//   %m = tensor.dim %a, %c1 : !a_tensor_type

//   // Expand activations / planes to the blocked reduction structure.
//   %aexp = tensor.expand_shape %a [[0], [1], [2, 3, 4, 5]] output_shape [%batch0, %m, {{group0}}, 2, 2, 8] : !a_tensor_type into !aexp_tensor_type
//   %qhexp = tensor.expand_shape %qh [[0], [1], [2, 3]] output_shape [{{n}}, {{group0}}, 2, 2] : !qh_tensor_type into !qhexp_tensor_type
//   %qsexp = tensor.expand_shape %qs [[0], [1], [2, 3]] output_shape [{{n}}, {{group0}}, 2, 8] : !qs_tensor_type into !qsexp_tensor_type

//   %result_empty = tensor.empty(%batch0, %m) : !accum_tensor_type
//   %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
//   %result = linalg.generic {
//       indexing_maps = [
//           // d0 = b, d1 = m, d2 = n, d3 = group (r), d4 = h (r), d5 = s (r), d6 = k (r)
//           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d5, d6)>,
//           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, 0)>,
//           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d4, d5)>,
//           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d5, d6)>,
//           affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>],
//       iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"] }
//       ins(%aexp, %d, %qhexp, %qsexp : !aexp_tensor_type, !d_tensor_type, !qhexp_tensor_type, !qsexp_tensor_type)
//       outs(%result_fill : !accum_tensor_type) {
//   ^bb0(%a_element: !a_type, %d_element: !scale_type, %qh_element: i8, %qs_element: i8, %out: !accum_type):
//       %h = linalg.index 4 : index
//       %kk = linalg.index 6 : index
//       %h_i32 = arith.index_cast %h : index to i32
//       %k_i32 = arith.index_cast %kk : index to i32
//       %c2_i32 = arith.constant 2 : i32
//       %c1_i32 = arith.constant 1 : i32
//       %c4_i32 = arith.constant 4 : i32
//       %c15_i32 = arith.constant 15 : i32
//       %c16_i32 = arith.constant 16 : i32

//       // Low nibble: shift by 4*h (h = 0 -> low nibble, h = 1 -> high nibble).
//       %qs_i32 = arith.extui %qs_element : i8 to i32
//       %nib_shift = arith.shli %h_i32, %c2_i32 : i32
//       %qs_shifted = arith.shrui %qs_i32, %nib_shift : i32
//       %low4 = arith.andi %qs_shifted, %c15_i32 : i32

//       // High (5th) bit: bit k of the qh byte.
//       %qh_i32 = arith.extui %qh_element : i8 to i32
//       %qh_shifted = arith.shrui %qh_i32, %k_i32 : i32
//       %hbit = arith.andi %qh_shifted, %c1_i32 : i32
//       %hbit_up = arith.shli %hbit, %c4_i32 : i32

//       // q in 0..31, then w = d * (q - 16) in f32.
//       %q = arith.ori %low4, %hbit_up : i32
//       %q_signed = arith.subi %q, %c16_i32 : i32
//       %q_fp = arith.sitofp %q_signed : i32 to !accum_type
//     {% if scale_type == accum_type %}
//       %d_f32 = arith.addf %d_element, %zero : !accum_type
//     {% else %}
//       %d_f32 = arith.extf %d_element : !scale_type to !accum_type
//     {% endif %}
//       %w = arith.mulf %q_fp, %d_f32 : !accum_type

//     {% if a_type == accum_type %}
//       %a_f32 = arith.addf %a_element, %zero : !accum_type
//     {% else %}
//       %a_f32 = arith.extf %a_element : !a_type to !accum_type
//     {% endif %}
//       %prod = arith.mulf %a_f32, %w : !accum_type
//       %acc = arith.addf %prod, %out : !accum_type
//       linalg.yield %acc : !accum_type
//   } -> !accum_tensor_type

//   // Cast.
//   %result_cast_empty = tensor.empty(%batch0, %m) : !c_tensor_type
//   %result_cast = linalg.copy
//     ins(%result : !accum_tensor_type)
//     outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
//   util.return %result_cast : !c_tensor_type
// }

// }
// Untiled Q5_0 matmul with transposed RHS: out[b,m,n] = sum_k a[b,m,k] * w[n,k].
//
// Planes (as stored in IRPA):
//   d  : [N, G, 1]  scale per block of 32 (G = K / 32)
//   qh : [N, G, 4]  32 high bits, bit j of the LE uint32 = element j
//   qs : [N, G, 16] byte j = elem j (low nibble) | elem j+16 (high nibble)
//
// Flat 4-D loop nest (b, m, n, kk), kk over the whole K. The block structure
// comes from floordiv / mod in the weight indexing maps:
//   d  : block = kk floordiv 32
//   qh : byte  = (kk mod 32) floordiv 8, bit = kk mod 8
//   qs : byte  = kk mod 16,              nibble = (kk mod 32) floordiv 16
// w = d * (q - 16) is computed inline; no dequantized [N, K] tensor exists.

{% set accum_type = "f32" %}

!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<?x?x{{k}}x!a_type>
!d_tensor_type = tensor<{{n}}x{{group0}}x1x!scale_type>
!qh_tensor_type = tensor<{{n}}x{{group0}}x4xi8>
!qs_tensor_type = tensor<{{n}}x{{group0}}x16xi8>
!accum_tensor_type = tensor<?x?x{{n}}x!accum_type>
!c_tensor_type = tensor<?x?x{{n}}x!a_type>

module {

util.func private @amdsharktank_mmt_block_scaled_q5_0_3d_{{n}}_{{k}}_{{a_type}}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qh: !qh_tensor_type, %qs: !qs_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.0: !accum_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch0 = tensor.dim %a, %c0 : !a_tensor_type
  %m = tensor.dim %a, %c1 : !a_tensor_type

  %result_empty = tensor.empty(%batch0, %m) : !accum_tensor_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          // d0 = b, d1 = m, d2 = n, d3 = kk (reduction over K) k =45 /32 = 1  13 1
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d2, d3 floordiv 32, 0)>,
          affine_map<(d0, d1, d2, d3) -> (d2, d3 floordiv 32, (d3 mod 32) floordiv 8)>,
          affine_map<(d0, d1, d2, d3) -> (d2, d3 floordiv 32, d3 mod 16)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"] }
      ins(%a, %d, %qh, %qs : !a_tensor_type, !d_tensor_type, !qh_tensor_type, !qs_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
  ^bb0(%a_element: !a_type, %d_element: !scale_type, %qh_element: i8, %qs_element: i8, %out: !accum_type):
      %kk = linalg.index 3 : index
      %kk_i32 = arith.index_cast %kk : index to i32
      %c1_i32 = arith.constant 1 : i32
      %c4_i32 = arith.constant 4 : i32
      %c7_i32 = arith.constant 7 : i32
      %c15_i32 = arith.constant 15 : i32
      %c16_i32 = arith.constant 16 : i32
                                    
      // Nibble select: j = kk mod 32; j < 16 -> low nibble, j >= 16 -> high.
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
  } -> !accum_tensor_type

  %result_cast_empty = tensor.empty(%batch0, %m) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
  util.return %result_cast : !c_tensor_type
}

}