// Copyright 2024 Advanced Micro Devices, Inc.
// Licensed under the Apache License v2.0 with LLVM Exceptions.

{% set accum_type = "f32" %}
!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}

!a_tensor_type = tensor<?x?x{{k}}x!a_type>
!aexp_tensor_type = tensor<?x?x{{group0}}x{{bs}}x!a_type>
!d_tensor_type = tensor<{{n}}x{{group0}}x1x!scale_type>
!qh_tensor_type = tensor<{{n}}x{{group0}}x4xi8>
!qs_tensor_type = tensor<{{n}}x{{group0}}x16xi8>
!accum_tensor_type = tensor<?x?x{{n}}x!accum_type>
!c_tensor_type = tensor<?x?x{{n}}x!a_type>

module {
  util.func private @amdsharktank_mmt_block_scaled_q5_0_3d_{{n}}_{{k}}_{{bs}}_{{a_type}}(
      %a: !a_tensor_type,
      %d: !d_tensor_type,
      %qh: !qh_tensor_type,
      %qs: !qs_tensor_type) -> !c_tensor_type {

    %zero = arith.constant 0.0 : !accum_type
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i8 = arith.constant 2 : i8
    %c4_i8 = arith.constant 4 : i8
    %c8_i8 = arith.constant 8 : i8
    %c0F_i8 = arith.constant 15 : i8
    %c1_i8 = arith.constant 1 : i8
    %c16_f16 = arith.constant 16.0 : !a_type

    %batch0 = tensor.dim %a, %c0 : !a_tensor_type
    %m = tensor.dim %a, %c1 : !a_tensor_type

    // Expand a: [B, M, K] -> [B, M, G, 32]
    %aexp = tensor.expand_shape %a [[0], [1], [2, 3]]
        output_shape [%batch0, %m, {{group0}}, {{bs}}]
        : !a_tensor_type into !aexp_tensor_type

    // Accumulator [B, M, N] filled with 0.0
    %result_empty = tensor.empty(%batch0, %m) : !accum_tensor_type
    %result_fill = linalg.fill ins(%zero: !accum_type)
        outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type

    // 🔑 COMBINED dequantize + matmul in ONE linalg.generic
    // No full [N, K] weight tensor is ever created.
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(b, m, n, g, k) -> (b, m, g, k)>,          // a
        affine_map<(b, m, n, g, k) -> (n, g, 0)>,            // d
        affine_map<(b, m, n, g, k) -> (n, g, k floordiv 8)>, // qh
        affine_map<(b, m, n, g, k) -> (n, g, k floordiv 2)>, // qs
        affine_map<(b, m, n, g, k) -> (b, m, n)>              // out
      ],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%aexp, %d, %qh, %qs
        : !aexp_tensor_type, !d_tensor_type, !qh_tensor_type, !qs_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
    ^bb0(%a_elem: !a_type, %d_elem: !scale_type,
         %qh_byte: i8, %qs_byte: i8, %out: !accum_type):

      %k_idx = linalg.index 4 : index
      %k_i8 = arith.index_cast %k_idx : index to i8

      // 5th bit from qh: bit = (qh_byte >> (k % 8)) & 1
      %bit_pos = arith.remui %k_i8, %c8_i8 : i8
      %qh_shifted = arith.shrui %qh_byte, %bit_pos : i8
      %bit = arith.andi %qh_shifted, %c1_i8 : i8

      // Low nibble from qs: nibble = (qs_byte >> (4 * (k % 2))) & 0xF
      %nibble_idx = arith.remui %k_i8, %c2_i8 : i8
      %nibble_shift = arith.muli %nibble_idx, %c4_i8 : i8
      %qs_shifted = arith.shrui %qs_byte, %nibble_shift : i8
      %nibble = arith.andi %qs_shifted, %c0F_i8 : i8

      // Assemble 5-bit q = nibble | (bit << 4)
      %bit_shifted = arith.shli %bit, %c4_i8 : i8
      %q = arith.ori %nibble, %bit_shifted : i8

      // Center: q_centered = q - 16
      %q_ext = arith.extsi %q : i8 to i32
      %q_fp = arith.sitofp %q_ext : i32 to !a_type
      %q_centered = arith.subf %q_fp, %c16_f16 : !a_type

      // w = d * (q - 16)
      %d_ext = arith.extf %d_elem : !scale_type to !a_type
      %w = arith.mulf %q_centered, %d_ext : !a_type

      // a[m,k] * w[n,k]
      %prod = arith.mulf %a_elem, %w : !a_type

      // Accumulate in f32
      %prod_ext = arith.extf %prod : !a_type to !accum_type
      %acc = arith.addf %prod_ext, %out : !accum_type
      linalg.yield %acc : !accum_type
    } -> !accum_tensor_type

    // Cast f32 accumulator -> f16 output
    %result_cast_empty = tensor.empty(%batch0, %m) : !c_tensor_type
    %result_cast = linalg.copy ins(%result : !accum_tensor_type)
        outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
    util.return %result_cast : !c_tensor_type
  }
}
