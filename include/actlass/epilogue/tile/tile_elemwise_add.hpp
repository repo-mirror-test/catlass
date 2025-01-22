/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTLASS_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP
#define ACTLASS_EPILOGUE_TILE_TILE_ELEMWISE_ADD_HPP

#include "actlass/actlass.hpp"

namespace actlass::epilogue::tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    uint32_t COMPUTE_LENGTH_
>
struct TileElemWiseAdd {
    using ArchTag = ArchTag_;
    using ElmentCompute = typename ComputeType_::Element;

    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    ACTLASS_DEVICE
    TileElemWiseAdd() {}

    ACTLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElmentCompute> const &ubOut,
        AscendC::LocalTensor<ElmentCompute> const &ubIn0,
        AscendC::LocalTensor<ElmentCompute> const &ubIn1
    )
    {
        // Do the calculation
        AscendC::Add(ubOut, ubIn0, ubIn1, COMPUTE_LENGTH);
    }
};

} // namespace actlass::epilogue::tile

#endif
