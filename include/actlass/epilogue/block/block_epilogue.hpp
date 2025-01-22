/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACTLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define ACTLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "actlass/actlass.hpp"

namespace actlass::epilogue::block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace actlass::epilogue::block

#include "actlass/epilogue/block/block_epilogue_elemwise_one_source.hpp"
#include "actlass/epilogue/block/block_epilogue_fa_softmax.hpp"
#include "actlass/epilogue/block/block_epilogue_fa_rescal_o.hpp"

#endif  // ACTLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
