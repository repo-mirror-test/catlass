/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_HPP
#define ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_HPP

#include "actlass/actlass.hpp"
#include "actlass/matmul/tile/tile_copy.hpp"
#include "actlass/matmul/tile/tile_mmad.hpp"

namespace actlass::matmul::block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = matmul::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = matmul::tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace actlass::matmul::block

#include "actlass/matmul/block/block_mmad_pingpong.hpp"
#include "actlass/matmul/block/block_mmad_fa_qk.hpp"
#include "actlass/matmul/block/block_mmad_fa_pv.hpp"
#include "actlass/matmul/block/block_mmad_preload.hpp"

#endif // ACTLASS_MATMUL_BLOCK_BLOCK_MMAD_HPP
