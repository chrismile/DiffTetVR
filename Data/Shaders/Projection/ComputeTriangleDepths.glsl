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

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform UniformDataBuffer {
    uint numTriangles;
};

layout(binding = 1, std430) readonly buffer VertexPositionBuffer {
    vec4 vertexPositions[];
};

struct TriangleKeyValue {
    uint index;
    float depth;
};
layout(binding = 2, std430) writeonly buffer TriangleKeyValueBuffer {
    TriangleKeyValue triangleKeyValues[];
};

void main() {
    const uint globalThreadIdx = gl_GlobalInvocationID.x;
    if (globalThreadIdx >= numTriangles) {
        return;
    }

    const uint triangleOffset = globalThreadIdx * 3u;
    vec4 p0 = vertexPositions[triangleOffset];
    vec4 p1 = vertexPositions[triangleOffset + 1u];
    vec4 p2 = vertexPositions[triangleOffset + 2u];
    vec4 centerPoint = (p0 + p1 + p2) / 3.0;

    TriangleKeyValue triKeyVal;
    triKeyVal.index = globalThreadIdx;
    triKeyVal.depth = centerPoint.z;
    triangleKeyValues[globalThreadIdx] = triKeyVal;
}
