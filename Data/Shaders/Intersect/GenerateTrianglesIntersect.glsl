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

layout(binding = 0) uniform UniformDataBuffer {
    mat4 viewProjMat;
    mat4 invProjMat;
    vec3 cameraPosition;
    float attenuationCoefficient;
    uint numTets;
};

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
layout(binding = 4, scalar) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};

// Output triangle data.
layout(binding = 5, scalar) writeonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 6, std430) writeonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};

const int tetFaceTable[4][4] = {
        { 0, 1, 2, 3 }, // Last index is point opposite to face.
        { 1, 0, 3, 2 }, // Last index is point opposite to face.
        { 0, 2, 3, 1 }, // Last index is point opposite to face.
        { 2, 1, 3, 0 }, // Last index is point opposite to face.
};

void pushTri(inout uint triOffset, vec3 pA vec3 pB, vec3 pC, uint tetIdx) {
    uint idx0 = triOffset * 3u;
    vertexPositions[idx0] = vec4(pA, 1.0);
    vertexPositions[idx0 + 1] = vec4(pB, 1.0);
    vertexPositions[idx0 + 2] = vec4(pC, 1.0);
    triangleTetIndices[triOffset] = tetIdx;
    triOffset++;
}

float solveLineT(vec2 p0, vec2 p1, vec2 pt) {
    vec2 num = pt - p0;
    vec2 denom = p1 - p0;
    return abs(denom.x) > abs(denom.y) ? num.x / denom.x : num.y / denom.y;
}

void main() {
    const uint tetIdx = gl_GlobalInvocationID.x;
    if (tetIdx >= numTets) {
        return;
    }

    // Read the tet vertex positions and colors from the global buffer.
    vec3 tetVertexPositions[4];
    vec4 tetVertexColors[4];
    vec4 tetVertexPositionsNdc[4];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
        tetVertexPositions[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
        vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
        vertexPosNdc.xyz /= vertexPosNdc.w;
        vertexPosNdc.w = 1.0;
        tetVertexPositionsNdc[tetVertIdx] = vertexPosNdc;
    }

    // Compute the signs of the faces.
    int tetFaceSigns[4];
    int numPositiveSigns = 0;
    int numNegativeSigns = 0;
    int numZeroSigns = 0;
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
        if (signValCam == 0) {
            signVal = 0;
            numZeroSigns++;
        } else if (signValCam != signValOpposite) {
            signVal = 1;
            numPositiveSigns++;
        } else if (signValCam == signValOpposite) {
            signVal = -1;
            numNegativeSigns++;
        }
        tetFaceSigns[tetFaceIdx] = signVal;
    }

    // Get the case index and decide on number of triangles.
    uint caseIdx = 4;
    uint numGeneratedTris = 1u;
    if (numZeroSigns == 0) {
        if (numPositiveSigns != numNegativeSigns) {
            caseIdx = 1;
            numGeneratedTris = 3u;
        } else {
            caseIdx = 2;
            numGeneratedTris = 4u;
        }
    } else if (numZeroSigns == 1) {
        caseIdx = 3;
        numGeneratedTris = 2u;
    }

    // Generate the projected triangles.
    uint triOffset = atomicAdd(globalTriangleCounter, numGeneratedTris);

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
        if (signValCam == 0) {
            signVal = 0;
            numZeroSigns++;
        } else if (signValCam != signValOpposite) {
            signVal = 1;
            numPositiveSigns++;
        } else if (signValCam == signValOpposite) {
            signVal = -1;
            numNegativeSigns++;
        }
        tetFaceSigns[tetFaceIdx] = signVal;
        pushTri(triOffset, p0, p1, p2, tetIdx);
    }

}
