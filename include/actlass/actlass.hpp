/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTLASS_ACTLASS_HPP
#define ACTLASS_ACTLASS_HPP

#include <kernel_operator.h>

#include "actlass/detail/alignment.hpp"
#include "actlass/detail/dependent_false.hpp"
#include "actlass/detail/macros.hpp"

namespace actlass {

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_FRACTAL = 512;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;

}  // namespace actlass

#endif  // ACTLASS_ACTLASS_HPP