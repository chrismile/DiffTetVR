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

#include "TetFaceTable.glsl"
#include "IntersectUniform.glsl"
#ifdef USE_CLIP_PLANE
#include "ClipPlane.glsl"
#endif

// Atomically increased linear append index.
layout(binding = 1, std430) buffer TriangleCounterBuffer {
    uint globalTriangleCounter;
};

// Input tet data.
layout(binding = 2, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 3, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};

// Output triangle data.
layout(binding = 4, scalar) writeonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 5, std430) writeonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};

void pushTri(inout uint triOffset, vec4 pA, vec4 pB, vec4 pC, uint tetIdx) {
    uint idx0 = triOffset * 3u;
    vertexPositions[idx0] = pA;
    vertexPositions[idx0 + 1] = pB;
    vertexPositions[idx0 + 2] = pC;
    triangleTetIndices[triOffset] = tetIdx;
    triOffset++;
}

void main() {
    const uint tetIdx = gl_GlobalInvocationID.x;
    if (tetIdx >= numTets) {
        return;
    }

    // Read the tet vertex positions from the global buffer.
    vec3 tetVertexPositions[4];
    vec4 tetVertexPositionsNdc[4];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
        tetVertexPositions[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
        vertexPosNdc.xyz /= vertexPosNdc.w;
        vertexPosNdc.w = 1.0;
        tetVertexPositionsNdc[tetVertIdx] = vertexPosNdc;
    }

#ifdef USE_CLIP_PLANE
    bool allOutside = true;
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        if (!checkIsPointOutsideClipPlane(tetVertexPositions[tetVertIdx])) {
            allOutside = false;
        }
    }
    if (allOutside) {
        return;
    }
#endif

    // Compute the signs of the faces.
    uint numGeneratedTris = 0;
    uint visibleFaceIndices[3];
    [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
        vec3 p0 = tetVertexPositions[tetFaceTable[tetFaceIdx][0]];
        vec3 p1 = tetVertexPositions[tetFaceTable[tetFaceIdx][1]];
        vec3 p2 = tetVertexPositions[tetFaceTable[tetFaceIdx][2]];
        vec3 p3 = tetVertexPositions[tetFaceTable[tetFaceIdx][3]];
        vec3 n = cross(p1 - p0, p2 - p0);
        float d = -dot(n, p0);
        float signValCam = sign(dot(cameraPosition, n) + d);
        float signValOpposite = sign(dot(p3, n) + d);
        int signVal = 0;
        if (signValCam != 0 && signValCam != signValOpposite) {
            visibleFaceIndices[numGeneratedTris] = tetFaceIdx;
            numGeneratedTris++;
        }
    }

    // Generate the projected triangles.
    uint triOffset = atomicAdd(globalTriangleCounter, numGeneratedTris);
    for (uint visibleFaceIdx = 0; visibleFaceIdx < numGeneratedTris; visibleFaceIdx++) {
        uint tetFaceIdx = visibleFaceIndices[visibleFaceIdx];
        vec4 p0 = tetVertexPositionsNdc[tetFaceTable[tetFaceIdx][0]];
        vec4 p1 = tetVertexPositionsNdc[tetFaceTable[tetFaceIdx][1]];
        vec4 p2 = tetVertexPositionsNdc[tetFaceTable[tetFaceIdx][2]];
        pushTri(triOffset, p0, p1, p2, tetIdx);
    }
}
