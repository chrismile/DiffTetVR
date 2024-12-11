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

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = BLOCK_SIZE) in;

#undef INTERSECTION_RENDERER

#ifdef INTERSECTION_RENDERER
#include "IntersectUniform.glsl"
#endif

layout(binding = 1) uniform TriangleCounterBuffer {
    uint numTriangles;
};

layout(binding = 2, std430) readonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};

#ifdef INTERSECTION_RENDERER
layout(binding = 3, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};
layout(binding = 4, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 5, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};
#endif

struct TriangleKeyValue {
    uint index;
    float depth;
};
layout(binding = 6, std430) writeonly buffer TriangleKeyValueBuffer {
    TriangleKeyValue triangleKeyValues[];
};

void main() {
    const uint globalThreadIdx = gl_GlobalInvocationID.x;
    if (globalThreadIdx >= numTriangles) {
        return;
    }

#ifdef INTERSECTION_RENDERER
    /*uint tetOffset = triangleTetIndices[globalThreadIdx] * 4u;
    vec3 centerPoint = vec3(0.0);
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        centerPoint += tetsVertexPositions[tetsIndices[tetOffset + tetVertIdx]];
    }
    centerPoint /= 4.0;
    vec4 centerPointHom = viewProjMat * vec4(centerPoint, 1.0);
    centerPoint = centerPointHom.xyz / centerPointHom.w;
    centerPoint.z = centerPoint.z;*/

    uint tetOffset = triangleTetIndices[globalThreadIdx] * 4u;
    vec3 centerPoint = vec3(0.0);
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        centerPoint += tetsVertexPositions[tetsIndices[tetOffset + tetVertIdx]];
    }
    centerPoint /= 4.0;
    vec4 centerPointHom = viewMat * vec4(centerPoint, 1.0);
    centerPoint = centerPointHom.xyz / centerPointHom.w;
    centerPoint.z = -centerPoint.z;
#else
    const uint triangleOffset = globalThreadIdx * 3u;
    vec4 p0 = vertexPositions[triangleOffset];
    vec4 p1 = vertexPositions[triangleOffset + 1u];
    vec4 p2 = vertexPositions[triangleOffset + 2u];
    vec4 centerPoint = (p0 + p1 + p2) / 3.0;
#endif

    TriangleKeyValue triKeyVal;
    triKeyVal.index = globalThreadIdx;
    triKeyVal.depth = centerPoint.z;
    triangleKeyValues[globalThreadIdx] = triKeyVal;
}
