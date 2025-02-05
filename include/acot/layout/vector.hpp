/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_LAYOUT_VECTOR_HPP
#define ACOT_LAYOUT_VECTOR_HPP

#include "acot/acot.hpp"
#include "acot/coord.hpp"

namespace acot::layout {

struct VectorLayout {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    ACOT_HOST_DEVICE
    VectorLayout() : stride_(MakeCoord(LongIndex(1))) {}

    /// Returns the stride of the layout
    ACOT_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

private:
    /// Stride data member
    Stride stride_;
};

} // namespace acot::layout

#endif  // ACOT_LAYOUT_VECTOR_HPP