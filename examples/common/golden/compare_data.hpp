/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP

#include <cmath>
#include <vector>

#include "acot/matmul_coord.hpp"

namespace acot::golden {

constexpr uint32_t COMPUTE_NUM_THRESHOLD = 2048;
constexpr float RTOL_GENERAL = 1.0f / 256;
constexpr float RTOL_OVER_THRESHOLD = 1.0f / 128;

template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum)
{
    float rtol = computeNum < COMPUTE_NUM_THRESHOLD ? RTOL_GENERAL : RTOL_OVER_THRESHOLD;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = static_cast<ElementCompare>(expect[i]);
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

}  // namespace acot::golden

#endif  // EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
