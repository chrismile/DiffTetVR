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
#include "ForwardCommon.glsl"
#include "RayCommon.glsl"

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

float computeAlpha(float thickness) {
    float alpha = 1.0 - exp(-thickness * attenuationCoefficient);
    return alpha;
}

vec4 integrateColor(vec4 cF, vec4 cB, float t) {
    const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
    float tSeg = t / float(NUM_SUBDIVS);
    vec4 rayColor = vec4(0.0);
    for (int s = 0; s < NUM_SUBDIVS; s++) {
        float fbegin = (float(s)) * INV_N_SUB;
        float fmid = (float(s) + 0.5) * INV_N_SUB;
        float fend = (float(s) + 1.0) * INV_N_SUB;
        vec3 c0 = mix(cF.rgb, cB.rgb, fbegin);
        vec3 c1 = mix(cF.rgb, cB.rgb, fend);
        float alpha = mix(cF.a, cB.a, fmid);
        vec4 currentColor = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient);
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
    }
    return rayColor;
}

void pushTri(inout uint triOffset, vec3 pF, vec3 pB, vec3 pC, vec4 cF, vec4 cB, vec4 cC) {
    uint idx0 = triOffset * 3u;
    vertexPositions[idx0] = vec4(pF, 1.0);
    vertexPositions[idx0 + 1] = vec4(pB, 1.0);
    vertexPositions[idx0 + 2] = vec4(pC, 1.0);
    //vertexColors[idx0] = vec4(alpha, alpha, alpha, alpha);
    //vertexColors[idx0 + 1] = vec4(0.0, 0.0, 0.0, 0.0);
    //vertexColors[idx0 + 2] = vec4(0.0, 0.0, 0.0, 0.0);
    //float alpha = computeAlpha(cF.a * thickness);
    //vertexColors[idx0] = vec4(cF.rgb, alpha);
    //vertexColors[idx0 + 1] = vec4(cB.rgb, 0.0);
    //vertexColors[idx0 + 2] = vec4(cC.rgb, 0.0);
    vertexColors[idx0] = vec4(cF.rgb / cF.a, cF.a);
    vertexColors[idx0 + 1] = vec4(cB.rgb, 0.0);
    vertexColors[idx0 + 2] = vec4(cC.rgb, 0.0);
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
    vec3 tetVertexPosition[4];
    vec4 tetVertexColors[4];
    vec4 tetVertexPositionNdc[4];
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetIdx * 4 + tetVertIdx];
        tetVertexPosition[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
        vec4 vertexPosNdc = viewProjMat * vec4(tetsVertexPositions[tetGlobalVertIdx], 1.0);
        vertexPosNdc.xyz /= vertexPosNdc.w;
        vertexPosNdc.w = 1.0;
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
        vec4 cA = tetVertexColors[tetFaceTable[oppositeFaceIdx][0]];
        vec4 cB = tetVertexColors[tetFaceTable[oppositeFaceIdx][1]];
        vec4 cC = tetVertexColors[tetFaceTable[oppositeFaceIdx][2]];
        vec4 cI = tetVertexColors[tetFaceTable[oppositeFaceIdx][3]];

        // Solve Barycentric interpolation equation described in Sec. 2.4 of paper by Shirley and Tuchmann.
        // Solve for pT.z, pT.{x,y} == pI.{x,y}: pT = pA + u(pB - pA) + v(pC - pA)
        vec2 uv = inverse(mat2(pB.xy - pA.xy, pC.xy - pA.xy)) * (pI.xy - pA.xy);
        vec3 pT = vec3(pI.xy, pA.z + uv.x * (pB.z - pA.z) + uv.y * (pC.z - pA.z));
        vec4 cT = cA + uv.x * (cB - cA) + uv.y * (cC - cA);
        vec4 pTW = invProjMat * vec4(pT, 1.0);
        vec4 pIW = invProjMat * vec4(pI, 1.0);
        float thickness = distance(pTW.xyz / pTW.w, pIW.xyz / pIW.w);
        vec4 colorFront, colorBack;
        if (numPositiveSigns > numNegativeSigns) {
            colorFront = cI;
            colorBack = cT;
        } else {
            colorFront = cT;
            colorBack = cI;
        }
        vec4 colorIntegrated = integrateColor(colorFront, colorBack, thickness);

        pushTri(triOffset, pT, pA, pB, colorIntegrated, cA, cB);
        pushTri(triOffset, pT, pB, pC, colorIntegrated, cB, cC);
        pushTri(triOffset, pT, pC, pA, colorIntegrated, cC, cA);
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
        uint bvf0 = 0, bvf1 = 0; // vertices bits plus/front
        uint bvb0 = 0, bvb1 = 0; // vertices bits minus/back
        [[unroll]] for (uint vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
            bvf0 |= 1u << tetFaceTable[ff0][vertexIdx];
            bvf1 |= 1u << tetFaceTable[ff1][vertexIdx];
            bvb0 |= 1u << tetFaceTable[fb0][vertexIdx];
            bvb1 |= 1u << tetFaceTable[fb1][vertexIdx];
        }
        uint bvf = bvf0 & bvf1; // vertices bits plus/front
        uint bvb = bvb0 & bvb1; // vertices bits minus/back
        uint idx = findLSB(bvf);
        vec3 pf0 = tetVertexPositionNdc[idx].xyz;
        vec4 cf0 = tetVertexColors[idx];
        idx = findMSB(bvf);
        vec3 pf1 = tetVertexPositionNdc[idx].xyz;
        vec4 cf1 = tetVertexColors[idx];
        idx = findLSB(bvb);
        vec3 pb0 = tetVertexPositionNdc[idx].xyz;
        vec4 cb0 = tetVertexColors[idx];
        idx = findMSB(bvb);
        vec3 pb1 = tetVertexPositionNdc[idx].xyz;
        vec4 cb1 = tetVertexColors[idx];

        // Compute the perspective formula for the lines and compute their intersection in screen coordinates.
        vec3 lf = cross(pf0, pf1);
        vec3 lb = cross(pb0, pb1);
        vec3 pIntersectScreen = cross(lf, lb);
        pIntersectScreen.xy /= pIntersectScreen.z;

        // Get pF (point on lf in clip space) and pB (point on lb in clip space)
        float tf = solveLineT(pf0.xy, pf1.xy, pIntersectScreen.xy);
        float tb = solveLineT(pb0.xy, pb1.xy, pIntersectScreen.xy);
        vec3 pF = vec3(pIntersectScreen.xy, pf0.z + tf * (pf1.z - pf0.z));
        vec4 cF = cf0 + tf * (cf1 - cf0);
        vec3 pB = vec3(pIntersectScreen.xy, pb0.z + tf * (pb1.z - pb0.z));
        vec4 cB = cb0 + tf * (cb1 - cb0);
        vec4 pFW = invProjMat * vec4(pF, 1.0);
        vec4 pBW = invProjMat * vec4(pB, 1.0);
        float thickness = distance(pFW.xyz / pFW.w, pBW.xyz / pBW.w);
        vec4 colorIntegrated = integrateColor(cF, cB, thickness);

        pushTri(triOffset, pF, pf0, pb0, colorIntegrated, cf0, cb0);
        pushTri(triOffset, pF, pf0, pb1, colorIntegrated, cf0, cb1);
        pushTri(triOffset, pF, pf1, pb0, colorIntegrated, cf1, cb0);
        pushTri(triOffset, pF, pf1, pb1, colorIntegrated, cf1, cb1);
    } else if (caseIdx == 3) {
        // 2 triangles; find the two tris formed by 2x negative or 2x positive faces.

        int twoSign = numPositiveSigns > numNegativeSigns ? 1 : -1;
        int singleSign = numPositiveSigns > numNegativeSigns ? -1 : 1;
        uint zeroFace = 4, f20 = 4, f21 = 4, f10 = 4; // face with sign 0, faces with double sign, faces with single sign
        [[unroll]] for (uint tetFaceIdx = 0; tetFaceIdx < 4; tetFaceIdx++) {
            if (tetFaceSigns[tetFaceIdx] == 0) {
                zeroFace = tetFaceIdx;
            } else if (tetFaceSigns[tetFaceIdx] == singleSign) {
                f10 = tetFaceIdx;
            } else if (f20 == 4u) {
                f20 = tetFaceIdx;
            } else {
                f21 = tetFaceIdx;
            }
        }

        // Protruding: Shared by zero and two sign faces.
        // Line: Shared by zero and single sign face.
        uint vertexBitsZero = 0x0u;
        //uint vertexBitsTwoSign0 = 0x0u;
        //uint vertexBitsTwoSign1 = 0x0u;
        uint vertexBitsSingleSign = 0x0u;
        [[unroll]] for (uint vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
            vertexBitsZero |= 1u << tetFaceTable[zeroFace][vertexIdx];
            //vertexBitsTwoSign0 |= 1u << tetFaceTable[f20][vertexIdx];
            //vertexBitsTwoSign1 |= 1u << tetFaceTable[f21][vertexIdx];
            vertexBitsSingleSign |= 1u << tetFaceTable[f10][vertexIdx];
        }
        uint ivProtruding = findLSB(vertexBitsSingleSign ^ 0xFu);
        uint ivThin = findLSB(vertexBitsZero ^ 0xFu);
        uint indicesLine = vertexBitsSingleSign & vertexBitsZero;
        uint ivLine0 = findLSB(indicesLine);
        uint ivLine1 = findMSB(indicesLine);

        vec3 pF = vec3(tetVertexPositionNdc[ivProtruding].xyz);
        vec3 pA = vec3(tetVertexPositionNdc[ivThin].xyz);
        vec3 pB = vec3(tetVertexPositionNdc[ivLine0].xyz);
        vec3 pC = vec3(tetVertexPositionNdc[ivLine1].xyz);
        vec4 cF = tetVertexColors[ivProtruding];
        vec4 cA = tetVertexColors[ivThin];
        vec4 cB = tetVertexColors[ivLine0];
        vec4 cC = tetVertexColors[ivLine1];

        // Compute intersection of line (eye, pF) with line (pB, pC) to get thickness.
        float t = solveLineT(pB.xy, pC.xy, pF.xy);
        vec3 pT = pB + t * (pC - pB);
        vec4 cT = cB + t * (cC - cB);
        vec4 pFW = invProjMat * vec4(pF, 1.0);
        vec4 pTW = invProjMat * vec4(pT, 1.0);
        float thickness = distance(pFW, pTW);
        vec4 colorFront, colorBack;
        if (numPositiveSigns > numNegativeSigns) {
            colorFront = cF;
            colorBack = cT;
        } else {
            colorFront = cT;
            colorBack = cF;
        }
        vec4 colorIntegrated = integrateColor(colorFront, colorBack, thickness);

        pushTri(triOffset, pF, pA, pB, colorIntegrated, cA, cB);
        pushTri(triOffset, pF, pA, pC, colorIntegrated, cA, cC);
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
        //uint vertexBits = 0xFu; // vertices bits front/back
        uint vertexBitsFront = 0x0u; // vertices bits front/back
        uint vertexBitsBack = 0x0u; // vertices bits front/back
        [[unroll]] for (uint vertexIdx = 0; vertexIdx < 3; vertexIdx++) {
            //vertexBits &= 1u << tetFaceTable[ff][vertexIdx];
            //vertexBits &= 1u << tetFaceTable[fb][vertexIdx];
            vertexBitsFront |= 1u << tetFaceTable[ff][vertexIdx];
            vertexBitsBack |= 1u << tetFaceTable[fb][vertexIdx];
        }
        uint vertexBits = vertexBitsFront & vertexBitsBack; // vertices bits front/back
        uint ivShared0 = findLSB(vertexBits);
        uint ivShared1 = findMSB(vertexBits);
        vertexBits = vertexBits ^ 0xFu; // invert bit mask.
        uint ivUnique0 = findLSB(vertexBits);
        uint ivUnique1 = findMSB(vertexBits);

        uint idxf = (vertexBitsFront & ivUnique0) != 0 ? ivUnique0 : ivUnique1;
        vec3 pF = vec3(tetVertexPositionNdc[idxf].xyz);
        vec3 pB = vec3(tetVertexPositionNdc[ivShared0].xyz);
        vec3 pC = vec3(tetVertexPositionNdc[ivShared1].xyz);
        vec4 cF = tetVertexColors[idxf];
        vec4 cB = tetVertexColors[ivShared0];
        vec4 cC = tetVertexColors[ivShared0];
        float thickness = distance(tetVertexPosition[ivUnique0], tetVertexPosition[ivUnique1]);
        vec4 colorIntegrated = integrateColor(cF, cB, thickness);

        pushTri(triOffset, pF, pB, pC, colorIntegrated, cB, cC);
    }
}
