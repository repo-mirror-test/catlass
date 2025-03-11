# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import sys
import math
import numpy as np

from dataclasses import dataclass

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def gen_seq_len(batch, max_seqlen, variate_seq=False):
    seqlen = np.ones((batch, )) * max_seqlen
    seqlen = seqlen.astype(np.int32)

    ntokens = seqlen.sum()
    return seqlen, ntokens


def fa_matmul(q_head, group_num, lhs, rhs):
    group_head = q_head // group_num
    score = None
    for i in range(group_num):
        group_score = np.matmul(lhs[i * group_head : (i + 1) * group_head, :, :].astype(np.float32),
            rhs[i : (i + 1), :, :].astype(np.float32)).astype(np.float32)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score


@dataclass
class FAParams:
    batch: int
    seqlen: int
    q_head: int
    group_num: int
    embed: int
    max_seqlen: int
    is_mask: bool = True
    variate_seq: bool = False


def calc_expect_func(params: FAParams):
    batch = params.batch
    seqlen = params.seqlen
    q_head = params.q_head
    group_num = params.group_num
    embed = params.embed
    max_seqlen = params.max_seqlen
    is_mask = params.is_mask
    variate_seq = params.variate_seq

    q_seqlen, q_ntokens = gen_seq_len(batch, seqlen, variate_seq)
    kv_seqlen, kv_ntokens = gen_seq_len(batch, seqlen, variate_seq)

    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, q_head * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(batch * max_seqlen, group_num * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(batch * max_seqlen, group_num * embed)).astype(np.float16)

    mask = np.ones(shape=(1, max_seqlen, max_seqlen)).astype(np.float16)
    mask = np.triu(mask, 1)
    mask *= -10000.0

    tor = np.float16(1.0 / math.sqrt(1.0 * embed))

    q_offset = 0
    k_offset = 0
    v_offset = 0

    s = None
    p = None
    out = None

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]
        q_slice = q[q_offset:q_offset + q_s][:]
        q_slice = q_slice.reshape(q_s, q_head, embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))  # (q_head, q_s, embed)
        k_slice = k[k_offset:k_offset + kv_s][:]
        k_slice = k_slice.reshape(kv_s, group_num, embed)
        k_slice_t = np.transpose(k_slice, (1, 2, 0))  # get k^T (group_num, embed, kv_s)
        v_slice = v[v_offset:v_offset + kv_s][:]
        v_slice = v_slice.reshape(kv_s, group_num, embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))  # (group_num, kv_s, embed)

        score = fa_matmul(q_head, group_num, q_slice, k_slice_t).astype(np.float16)
        if s is None:
            s = score.reshape([-1, ])
        else:
            s = np.concatenate((s, score.reshape([-1, ])), 0)

        score = score * tor
        if is_mask:
            score = score + mask[:, :q_s, :kv_s]
        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((q_head, q_s, 1))
        score_exp = np.exp(score.astype(np.float32))
        if p is None:
            p = score_exp.astype(np.float16).reshape([-1, ])
        else:
            p = np.concatenate((p, score_exp.astype(np.float16).reshape([-1, ])), 0)
        score_sum = np.sum(score_exp, axis=-1)
        o_tmp = fa_matmul(q_head, group_num, score_exp.astype(np.float16), v_slice)
        o_tmp = o_tmp.astype(np.float16) / score_sum.reshape((q_head, q_s, 1)).astype(np.float16)

        o_tmp = o_tmp.reshape(q_head, q_s, embed)
        o_tmp = np.transpose(o_tmp, (1, 0, 2))
        o_tmp = np.ascontiguousarray(o_tmp)
        if out is None:
            out = o_tmp
        else:
            out = np.concatenate((out, o_tmp), 0)

        q_offset += q_s
        k_offset += max_seqlen
        v_offset += max_seqlen

    q.astype(np.float16).tofile(os.path.join(WORKSPACE, "data", "q.bin"))
    k.astype(np.float16).tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    v.astype(np.float16).tofile(os.path.join(WORKSPACE, "data", "v.bin"))
    mask.astype(np.float16).tofile(os.path.join(WORKSPACE, "data", "mask.bin"))
    q_seqlen.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_seqlen.bin"))
    q_ntokens.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "q_ntokens.bin"))
    kv_seqlen.astype(np.int32).tofile(os.path.join(WORKSPACE, "data", "kv_seqlen.bin"))
    out.astype(np.float32).tofile(os.path.join(WORKSPACE, "data", "golden.bin"))


if __name__ == "__main__":
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)

    batch = int(sys.argv[1])
    seqlen = int(sys.argv[2])
    q_head = int(sys.argv[3])
    group_num = int(sys.argv[4])
    embed = int(sys.argv[5])
    max_seqlen = int(sys.argv[6])
    calc_expect_func(FAParams(batch, seqlen, q_head, group_num, embed, max_seqlen))

