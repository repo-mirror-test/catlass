/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTLASS_MATMUL_TILE_TILE_COPY_HPP
#define ACTLASS_MATMUL_TILE_TILE_COPY_HPP

#include "actlass/matmul/tile/copy_gm_to_l1.hpp"
#include "actlass/matmul/tile/copy_l0c_to_gm.hpp"
#include "actlass/matmul/tile/copy_l1_to_l0.hpp"
#include "actlass/matmul/helper.hpp"


namespace actlass::matmul::tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for B matrix operand
    class BType,
    /// MatmulType type for C matrix operand
    class CType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename matmul::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = matmul::tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = matmul::tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = matmul::tile::CopyL1ToL0A<ArchTag, AType>;
    using CopyL1ToL0B = matmul::tile::CopyL1ToL0B<ArchTag, BType>;
    using CopyL0CToGm = matmul::tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};

} // namespace actlass::matmul::tile

#endif // ACTLASS_MATMUL_TILE_TILE_COPY_HPP
