/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

-- Compute

#version 450

layout(local_size_x = 1) in;

struct VkDrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct VkDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

layout(binding = 0) uniform TriangleCounterBuffer {
    uint numTriangles;
};
layout(binding = 1, std430) readonly buffer DrawIndirectCommandBuffer {
    VkDrawIndirectCommand drawIndirectCommandData;
};
layout(binding = 2, std430) readonly buffer DispatchIndirectCommandBuffer {
    VkDispatchIndirectCommand dispatchIndirectCommandData;
};

uint uiceil(uint x, uint y) {
    return x > 0u ? (x - 1u) / y + 1u : 0u;
}

void main() {
    drawIndirectCommandData.vertexCount = numTriangles * 3u;
    drawIndirectCommandData.instanceCount = 1u;
    drawIndirectCommandData.firstVertex = 0u;
    drawIndirectCommandData.firstInstance = 0u;

    dispatchIndirectCommandData.x = uiceil(numTriangles, BLOCK_SIZE);
    dispatchIndirectCommandData.y = 1u;
    dispatchIndirectCommandData.z = 1u;
}
