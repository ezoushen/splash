#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"

// The gated output of a full-attention layer at element `element` of a
// lane's rows of every query head: the head's attention row, staged `stride`
// rows per query group ([group][row][head in group][dimension]), times
// the sigmoid of the head's gate in the packed row. QueryGroup may split
// one physical KV head's queries into smaller groups. Lane `lane`'s rows start
// at packed row lane x lane_rows and its staging at group lane x Groups; a
// prefill chunk is lane 0.
template <uint QHeads, uint KHeads, uint QueryGroup = QHeads / KHeads>
inline bfloat full_attention_gate_value(device const bfloat *packed_qkv,
                                        device const bfloat *attention,
                                        uint lane, uint lane_rows, uint stride,
                                        uint element) {
  constexpr uint HeadDim = 256, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint Groups = QHeads / QueryGroup;
  uint row = element / (QHeads * HeadDim);
  uint remainder = element % (QHeads * HeadDim);
  uint query_head = remainder / HeadDim;
  uint dim = remainder % HeadDim;
  float gate = float(packed_qkv[(ulong(lane) * lane_rows + row) * PackedStride +
                                query_head * QStride + HeadDim + dim]);
  float gate_scale = splash_sigmoid(gate);
  uint query_group = query_head / QueryGroup;
  uint local_head = query_head % QueryGroup;
  ulong attention_index =
      (((ulong(lane) * Groups + query_group) * stride + row) * QueryGroup +
       local_head) *
          HeadDim +
      dim;
  return bfloat(float(attention[attention_index]) * gate_scale);
}
