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

-- Vertex

#version 450

struct TriangleKeyValue {
    uint index;
    float depth;
};
layout(binding = 1, std430) readonly buffer TriangleKeyValueBuffer {
    TriangleKeyValue triangleKeyValues[];
};
layout(binding = 2, std430) readonly buffer TriangleVertexPositionBuffer {
    vec4 vertexPositions[];
};
layout(binding = 3, std430) readonly buffer TriangleTetIndexBuffer {
    uint triangleTetIndices[];
};
#ifdef BACK_TO_FRONT_BLENDING
layout(binding = 4) uniform TriangleCounterBuffer {
    uint numTriangles;
};
#endif

layout(location = 0) flat out uint tetIdx;

void main() {
#ifdef BACK_TO_FRONT_BLENDING
    // Rendering is done in back-to-front order, so we reverse the index using "numTriangles - i - 1".
    uint triangleIdx = triangleKeyValues[numTriangles - gl_VertexIndex / 3u - 1u].index;
#else
    uint triangleIdx = triangleKeyValues[gl_VertexIndex / 3u].index;
#endif
    uint vertexIdx = triangleIdx * 3u + (gl_VertexIndex % 3u);
    tetIdx = triangleTetIndices[triangleIdx];
    gl_Position = vertexPositions[vertexIdx];
}


-- Fragment

#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_terminate_invocation : require
//#extension GL_EXT_debug_printf : enable

#include "TetFaceTable.glsl"
#include "RayCommon.glsl"
#include "IntersectUniform.glsl"

#ifdef USE_SHADING
#include "Lighting.glsl"
#endif

// Tet data.
layout(binding = 5, std430) readonly buffer TetIndexBuffer {
    uint tetsIndices[];
};
layout(binding = 6, scalar) readonly buffer TetVertexPositionBuffer {
    vec3 tetsVertexPositions[];
};

#ifndef SHOW_TET_QUALITY
#ifdef PER_VERTEX_COLORS
layout(binding = 7, scalar) readonly buffer TetVertexColorBuffer {
    vec4 tetsVertexColors[];
};
#else
layout(binding = 7, scalar) readonly buffer TetCellColorBuffer {
    vec4 tetsCellColors[];
};
#endif
#else
#define INVALID_TET 0xFFFFFFFFu
layout(binding = 7, scalar) readonly buffer TetQualityBuffer {
    float tetQualityArray[];
};
layout (binding = 8) uniform MinMaxUniformBuffer {
    float minAttributeValue;
    float maxAttributeValue;
};
layout(binding = 9) uniform sampler1D transferFunctionTexture;
vec4 transferFunction(float attr) {
    // Transfer to range [0, 1].
    float posFloat = clamp((attr - minAttributeValue) / (maxAttributeValue - minAttributeValue), 0.0, 1.0);
    // Look up the color value.
    return texture(transferFunctionTexture, posFloat);
}
#endif

#ifdef SHOW_DEPTH_COMPLEXITY
// Stores the number of fragments using atomic operations.
layout(binding = 10) coherent buffer DepthComplexityCounterBuffer {
    uint depthComplexityCounterBuffer[];
};

uint addrGenLinear(uvec2 addr2D) {
    return addr2D.x + viewportLinearW * addr2D.y;
}
#endif

in vec4 gl_FragCoord;
layout(location = 0) flat in uint tetIdx;
layout(location = 0) out vec4 outputColor;

#include "RayIntersectionTests.glsl"
#include "BarycentricInterpolation.glsl"
#include "ForwardCommon.glsl"

void main() {
    // Compute camera ray direction.
    vec2 fragNdc = 2.0 * (gl_FragCoord.xy / vec2(viewportSize)) - 1.0;
    vec3 rayTarget = (invProjMat * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (invViewMat * vec4(normalizedTarget, 0.0)).xyz;

    // Fetch the tet vertex positions and colors.
    vec3 tetVertexPositions[4];
#if !defined(SHOW_TET_QUALITY) && defined(PER_VERTEX_COLORS)
    vec4 tetVertexColors[4];
#endif
    const uint tetOffset = tetIdx * 4u;
    [[unroll]] for (uint tetVertIdx = 0; tetVertIdx < 4; tetVertIdx++) {
        uint tetGlobalVertIdx = tetsIndices[tetOffset + tetVertIdx];
        tetVertexPositions[tetVertIdx] = tetsVertexPositions[tetGlobalVertIdx];
#if !defined(SHOW_TET_QUALITY) && defined(PER_VERTEX_COLORS)
        tetVertexColors[tetVertIdx] = tetsVertexColors[tetGlobalVertIdx];
#endif
    }

    uint f0, f1; //< Face index hit 0, 1.
    float t0, t1; //< Ray distance hit 0, 1.
    if (!intersectRayTet(cameraPosition, rayDirection, tetVertexPositions, f0, f1, t0, t1)) {
        //outputColor = vec4(1.0, 0.0, 0.0, 1.0);
        //return;
        terminateInvocation; // Should never happen...
    }

    vec3 intersectPos0 = cameraPosition + t0 * rayDirection;
    vec3 intersectPos1 = cameraPosition + t1 * rayDirection;

#if !defined(SHOW_TET_QUALITY) && defined(PER_VERTEX_COLORS)
    // Barycentric interpolation (face 0 and 1).
    uint i0, i1, i2;
    vec3 p0, p1, p2;
    vec4 c0, c1, c2;
    i0 = tetFaceTable[f0][0];
    i1 = tetFaceTable[f0][1];
    i2 = tetFaceTable[f0][2];
    p0 = tetVertexPositions[i0];
    p1 = tetVertexPositions[i1];
    p2 = tetVertexPositions[i2];
    c0 = tetVertexColors[i0];
    c1 = tetVertexColors[i1];
    c2 = tetVertexColors[i2];
    vec3 barycentricCoordinates = barycentricInterpolation(p0, p1, p2, intersectPos0);
    vec4 fragment0Color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;

    i0 = tetFaceTable[f1][0];
    i1 = tetFaceTable[f1][1];
    i2 = tetFaceTable[f1][2];
    p0 = tetVertexPositions[i0];
    p1 = tetVertexPositions[i1];
    p2 = tetVertexPositions[i2];
    c0 = tetVertexColors[i0];
    c1 = tetVertexColors[i1];
    c2 = tetVertexColors[i2];
    barycentricCoordinates = barycentricInterpolation(p0, p1, p2, intersectPos1);
    vec4 fragment1Color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;
#endif
#if defined(SHOW_TET_QUALITY) && defined(USE_SHADING)
    uint i0 = tetFaceTable[f0][0];
    uint i1 = tetFaceTable[f0][1];
    uint i2 = tetFaceTable[f0][2];
    vec3 p0 = tetVertexPositions[i0];
    vec3 p1 = tetVertexPositions[i1];
    vec3 p2 = tetVertexPositions[i2];
    vec3 faceNormal0 = normalize(cross(p1 - p0, p2 - p0));
#endif

    vec4 rayColor = vec4(0.0);
    float t = t1 - t0;

#ifdef SHOW_TET_QUALITY

    float tetQuality = tetQualityArray[tetIdx];
    rayColor = transferFunction(tetQuality);
#ifdef USE_SHADING
    rayColor = blinnPhongShadingSurface(rayColor, intersectPos0, faceNormal0);
#endif
#ifndef ALPHA_MODE_STRAIGHT
    rayColor.rgb *= rayColor.a;
#endif

#else // !defined(SHOW_TET_QUALITY)

#ifdef PER_VERTEX_COLORS

#ifdef USE_SUBDIVS
    float tSeg = t / float(NUM_SUBDIVS);
    const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
    for (int s = 0; s < NUM_SUBDIVS; s++) {
        float fbegin = (float(s)) * INV_N_SUB;
        float fmid = (float(s) + 0.5) * INV_N_SUB;
        float fend = (float(s) + 1.0) * INV_N_SUB;
        vec3 c0 = mix(fragment0Color.rgb, fragment1Color.rgb, fbegin);
        vec3 c1 = mix(fragment0Color.rgb, fragment1Color.rgb, fend);
        float alpha = mix(fragment0Color.a, fragment1Color.a, fmid);
        vec4 currentColor = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient);
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
    }
#else
    vec4 currentColor = accumulateLinear(
            t, fragment0Color.rgb, fragment1Color.rgb,
            fragment0Color.a * attenuationCoefficient, fragment1Color.a * attenuationCoefficient);
    rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
    rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
#endif

#else // !defined(PER_VERTEX_COLORS)

    vec4 fragmentColor = tetsCellColors[tetIdx];
    vec4 currentColor = accumulateConst(t, fragmentColor.rgb, fragmentColor.a * attenuationCoefficient);
    rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
    rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;

#endif // PER_VERTEX_COLORS

#ifdef ALPHA_MODE_STRAIGHT
    if (rayColor.a > 1e-5) {
        rayColor.rgb = rayColor.rgb / rayColor.a; // Correct rgb with alpha
    }
#endif

#endif // SHOW_TET_QUALITY

#ifdef SHOW_DEPTH_COMPLEXITY
    uvec2 fragCoordUvec = uvec2(gl_FragCoord.xy);
    atomicAdd(depthComplexityCounterBuffer[addrGenLinear(fragCoordUvec)], 1u);
#endif

    outputColor = rayColor;
}
