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
layout(binding = 6, std430) writeonly buffer TriangleVertexColorBuffer {
    vec4 vertexColors[];
};

const int tetFaceTable[4][4] = {
        { 0, 1, 2, 3 }, // Last index is point opposite to face.
        { 1, 0, 3, 2 }, // Last index is point opposite to face.
        { 0, 2, 3, 1 }, // Last index is point opposite to face.
        { 2, 1, 3, 0 }, // Last index is point opposite to face.
};

float computeAlpha(float thickness) {
    float alpha = 1.0 - exp(-thickness * attenuationCoefficient);
    return alpha;
}

void pushTri(inout uint triOffset, vec3 pF, vec3 pB, vec3 pC, float alpha) {
    uint idx0 = triOffset * 3u;
    vertexPositions[idx0] = vec4(pF, 1.0);
    vertexPositions[idx0 + 1] = vec4(pB, 1.0);
    vertexPositions[idx0 + 2] = vec4(pC, 1.0);
    vertexColors[idx0] = vec4(1.0, 1.0, 1.0, alpha);
    vertexColors[idx0 + 1] = vec4(0.0, 0.0, 0.0, 0.0);
    vertexColors[idx0 + 2] = vec4(0.0, 0.0, 0.0, 0.0);
    triOffset++;
}

void main() {
    const uint tetIdx = gl_GlobalInvocationID.x;
    if (tetIdx >= numTets) {
        return;
    }

    // Read the tet vertex positions and colors from the global buffer.
    vec3 tetVertexPosition[4];
    vec4 tetVertexColors[4];
    vec4 tetVertexPositionNdc[4];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
        tetVertexPosition[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
        vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
        vertexPosNdc.xyz /= vertexPosNdc.w;
        tetVertexPositionNdc[tetVertIdx] = vertexPosNdc;
    }

    // Compute the signs of the faces.
    int tetFaceSigns[4];
    int numPositiveSigns = 0;
    int numNegativeSigns = 0;
    int numZeroSigns = 0;
    [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
        vec3 p0 = tetVertexPosition[tetFaceTable[tetFaceIdx][0]];
        vec3 p1 = tetVertexPosition[tetFaceTable[tetFaceIdx][1]];
        vec3 p2 = tetVertexPosition[tetFaceTable[tetFaceIdx][2]];
        vec3 p3 = tetVertexPosition[tetFaceTable[tetFaceIdx][3]];
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

    if (caseIdx == 1) {
        // 3 triangles; find vertex shared by all three (pI), and 3x not shared by all three (pA, pB, pC).
        // Find index of face with sign different from all others.
        int outlierSign = numPositiveSigns < numNegativeSigns ? 1 : -1;
        uint oppositeFaceIdx = 0;
        [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
            if (outlierSign == tetFaceSigns[tetFaceIdx]) {
                oppositeFaceIdx = tetFaceIdx;
                break;
            }
        }
        vec3 pA = tetVertexPositionNdc[tetFaceTable[oppositeFaceIdx][0]].xyz;
        vec3 pB = tetVertexPositionNdc[tetFaceTable[oppositeFaceIdx][1]].xyz;
        vec3 pC = tetVertexPositionNdc[tetFaceTable[oppositeFaceIdx][2]].xyz;
        vec3 pI = tetVertexPositionNdc[tetFaceTable[oppositeFaceIdx][3]].xyz;

        // Solve Barycentric interpolation equation described in Sec. 2.4 of paper by Shirley and Tuchmann.
        // Solve for pT.z, pT.{x,y} == pI.{x,y}: pT = pA + u(pB - pA) + v(pC - pA)
        vec2 uv = inverse(mat2(pB - pA, pC - pA)) * (pI.xy - pA.xy);
        vec3 pT = vec3(pI.x, pI.y, pA.z + uv.x * (pB.z - pA.z) + uv.y * (pC.z - pA.z));
        vec4 pTW = invProjMat * vec4(pT, 1.0);
        vec4 pIW = invProjMat * vec4(pI, 1.0);
        float thickness = distance(pTW.xyz / pTW.w, pIW.xyz / pIW.w);
        float alpha = computeAlpha(thickness);

        pushTri(triOffset, pT, pA, pB, alpha);
        pushTri(triOffset, pT, pB, pC, alpha);
        pushTri(triOffset, pT, pC, pA, alpha);
    } else if (caseIdx == 2) {
        // Find vertices forming lines lp and lm, which are formed by faces with sign (1, 1) and sign (-1, -1).
        uint ff0 = 4, ff1 = 4; // faces plus
        uint fb0 = 4, fb1 = 4; // faces minus
        [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
            if (tetFaceSigns[tetFaceIdx] == 1) {
                if (ff0 == 4) {
                    ff0 = tetFaceIdx;
                } else {
                    ff1 = tetFaceIdx;
                }
            } else {
                if (fb0 == 4) {
                    fb0 = tetFaceIdx;
                } else {
                    fb1 = tetFaceIdx;
                }
            }
        }
        uint bvf = 0x7u; // vertices bits plus/front
        uint bvb = 0x7u; // vertices bits minus/back
        [[unroll]] for (uint vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
            bvf &= 1u << tetFaceTable[ff0][vertexIdx];
            bvf &= 1u << tetFaceTable[ff1][vertexIdx];
            bvb &= 1u << tetFaceTable[fb0][vertexIdx];
            bvb &= 1u << tetFaceTable[fb1][vertexIdx];
        }
        vec3 pf0 = vec3(tetVertexPositionNdc[findLSB(bvf)].xyz);
        vec3 pf1 = vec3(tetVertexPositionNdc[findMSB(bvf)].xyz);
        vec3 pb0 = vec3(tetVertexPositionNdc[findLSB(bvb)].xyz);
        vec3 pb1 = vec3(tetVertexPositionNdc[findMSB(bvb)].xyz);

        // Compute the perspective formula for the lines and compute their intersection in screen coordinates.
        vec3 lf = cross(pf0, pf1);
        vec3 lb = cross(pb0, pb1);
        vec3 pIntersectScreen = cross(lf, lb);
        pIntersectScreen.xy /= pIntersectScreen.z;

        // TODO: Get pF (point on lf in world space) and pB (point on lb in world space)
        vec3 pF = vec3(0.0);
        vec3 pB = vec3(0.0);
        vec4 pFW = invProjMat * vec4(pF, 1.0);
        vec4 pBW = invProjMat * vec4(pB, 1.0);
        float thickness = distance(pFW.xyz / pFW.w, pBW.xyz / pBW.w);

        float alpha = computeAlpha(thickness);
        pushTri(triOffset, pF, pf0, pb0, alpha);
        pushTri(triOffset, pF, pf0, pb1, alpha);
        pushTri(triOffset, pF, pf1, pb0, alpha);
        pushTri(triOffset, pF, pf1, pb1, alpha);
    } else if (caseIdx == 3) {
        // 2 triangles; find the two tris formed by 2x negative or 2x positive faces.
        // TODO
        //pushTri(triOffset, vec3(0.0), vec3(0.0), vec3(0.0), 0.0);
        //pushTri(triOffset, vec3(0.0), vec3(0.0), vec3(0.0), 0.0);
    } else if (caseIdx == 4) {
        // The protruding vertex is not shared by front and back face.
        uint ff = 4, fb = 4; // face front, back
        [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
            if (tetFaceSigns[tetFaceIdx] == 1) {
                ff = tetFaceIdx;
            } else if (tetFaceSigns[tetFaceIdx] == -1) {
                fb = tetFaceIdx;
            }
        }

        // Get index not shared by front and back face.
        uint vertexBits = 0x7u; // vertices bits front/back
        uint vertexBitsFront = 0x0u; // vertices bits front/back
        uint vertexBitsBack = 0x0u; // vertices bits front/back
        [[unroll]] for (uint vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
            vertexBits &= 1u << tetFaceTable[ff][vertexIdx];
            vertexBits &= 1u << tetFaceTable[fb][vertexIdx];
            vertexBitsFront |= 1u << tetFaceTable[ff][vertexIdx];
            vertexBitsBack |= 1u << tetFaceTable[fb][vertexIdx];
        }
        uint ivShared0 = findLSB(vertexBits);
        uint ivShared1 = findMSB(vertexBits);
        vertexBits = vertexBits ^ 0x7u; // invert bit mask.
        uint ivUnique0 = findLSB(vertexBits);
        uint ivUnique1 = findMSB(vertexBits);

        vec3 pF = vec3(tetVertexPositionNdc[(vertexBitsFront & ivUnique0) != 0 ? ivUnique0 : ivUnique1].xyz);
        vec3 pB = vec3(tetVertexPositionNdc[ivShared0].xyz);
        vec3 pC = vec3(tetVertexPositionNdc[ivShared1].xyz);
        float thickness = distance(tetVertexPosition[ivUnique0], tetVertexPosition[ivUnique1]);

        float alpha = computeAlpha(thickness);
        pushTri(triOffset, pF, pB, pC, alpha);
    }
}
