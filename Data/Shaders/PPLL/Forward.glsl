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

#include "ForwardCommon.glsl"

#ifdef USE_SHADING
#include "Lighting.glsl"
#endif

void getNextFragment(
        in uint i, in uint fragsCount,
#if defined(SHOW_TET_QUALITY) || !defined(PER_VERTEX_COLORS)
        out uvec2 tetIds,
#else
        out vec4 color,
#endif
#if defined(SHOW_TET_QUALITY) && defined(USE_SHADING)
        out vec3 fragmentPosition,
        out vec3 faceNormal,
#endif
        out float depthLinear, out bool boundary, out bool frontFace) {
    minHeapSink4(0, fragsCount - i);
    uint faceBits = colorList[0];
    float depthBufferValue = depthList[0];
    colorList[0] = colorList[fragsCount - i - 1];
    depthList[0] = depthList[fragsCount - i - 1];

    depthLinear = convertDepthBufferValueToLinearDepth(depthBufferValue);
    frontFace = (faceBits & 1u) == 1u ? true : false;
    boundary = ((faceBits >> 1u) & 1u) == 1u ? true : false;

#if defined(SHOW_TET_QUALITY) || !defined(PER_VERTEX_COLORS)
    uint faceIndexTri = faceBits >> 2u;
    tetIds = faceToTetMap[faceIndexTri];
#if defined(SHOW_TET_QUALITY) && defined(USE_SHADING)
    uint faceIndex = (faceBits >> 2u) * 3u;
    vec4 fragPosNdc = vec4(2.0 * gl_FragCoord.xy / vec2(viewportSize) - vec2(1.0), depthBufferValue, 1.0);
    vec4 fragPosWorld = inverseViewProjectionMatrix * fragPosNdc;
    vec3 fragmentPositionWorld = fragPosWorld.xyz / fragPosWorld.w;
    uint i0 = triangleIndices[faceIndex];
    uint i1 = triangleIndices[faceIndex + 1];
    uint i2 = triangleIndices[faceIndex + 2];
    vec3 p0 = vertexPositions[i0];
    vec3 p1 = vertexPositions[i1];
    vec3 p2 = vertexPositions[i2];
    fragmentPosition = fragmentPositionWorld;
    faceNormal = normalize(cross(p1 - p0, p2 - p0));
#endif
#else // !defined(SHOW_TET_QUALITY) && defined(PER_VERTEX_COLORS)
    uint faceIndex = (faceBits >> 2u) * 3u;
    // Compute world space position from depth.
#ifndef COMPUTE_SHADER // TODO
    vec2 fragCoord = gl_FragCoord.xy;
#endif
    vec4 fragPosNdc = vec4(2.0 * gl_FragCoord.xy / vec2(viewportSize) - vec2(1.0), depthBufferValue, 1.0);
    vec4 fragPosWorld = inverseViewProjectionMatrix * fragPosNdc;
    vec3 fragmentPositionWorld = fragPosWorld.xyz / fragPosWorld.w;

    uint i0 = triangleIndices[faceIndex];
    uint i1 = triangleIndices[faceIndex + 1];
    uint i2 = triangleIndices[faceIndex + 2];
    vec3 p0 = vertexPositions[i0];
    vec3 p1 = vertexPositions[i1];
    vec3 p2 = vertexPositions[i2];
#ifdef PER_VERTEX_COLORS
    vec4 c0 = vertexColors[i0];
    vec4 c1 = vertexColors[i1];
    vec4 c2 = vertexColors[i2];
#endif

    // Barycentric interpolation.
    vec3 d20 = p2 - p0;
    vec3 d21 = p2 - p1;
    float totalArea = max(length(cross(d20, d21)), 1e-5);
    float u = length(cross(d21, fragmentPositionWorld - p1)) / totalArea;
    float v = length(cross(fragmentPositionWorld - p0, d20)) / totalArea;
    const vec3 barycentricCoordinates = vec3(u, v, 1.0 - u - v);
    color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;
#endif
}

vec4 frontToBackPQ(uint fragsCount) {
    uint i;

    // Bring it to heap structure
    for (i = fragsCount/4; i > 0; --i) {
        // First is not one right place - will be done in for
        minHeapSink4(i, fragsCount); // Sink all inner nodes
    }

    // Start with transparent Ray
    vec4 rayColor = vec4(0.0);
    vec4 currentColor;

#ifdef SHOW_TET_QUALITY

    uvec2 fragmentTetIds;
    float fragmentDepth;
    bool fragmentBoundary;
    bool fragmentFrontFace;
#ifdef USE_SHADING
    vec3 fragmentPositionWorld, lastFragmentPositionWorld;
    vec3 fragmentNormal, lastFragmentNormal;
#endif

    uint openTetId = INVALID_TET;
    for (i = 0; i < fragsCount; i++) {
        getNextFragment(
                i, fragsCount, fragmentTetIds,
#ifdef USE_SHADING
                fragmentPositionWorld, fragmentNormal,
#endif
                fragmentDepth, fragmentBoundary, fragmentFrontFace);
        bool eqA = fragmentTetIds.x != INVALID_TET && fragmentTetIds.x == openTetId;
        bool eqB = fragmentTetIds.y != INVALID_TET && fragmentTetIds.y == openTetId;
        if (eqA || eqB) {
            float tetQuality = tetQualityArray[openTetId];
            currentColor = transferFunction(tetQuality);
#ifdef USE_SHADING
            currentColor = blinnPhongShadingSurface(currentColor, lastFragmentPositionWorld, lastFragmentNormal);
#endif
            rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.a * currentColor.rgb;
            rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
        } else {
            eqA = fragmentTetIds.y != INVALID_TET;
        }
        openTetId = eqA ? fragmentTetIds.y : fragmentTetIds.x;
#ifdef USE_SHADING
        lastFragmentPositionWorld = fragmentPositionWorld;
        lastFragmentNormal = fragmentNormal;
#endif
    }

#elif defined(PER_VERTEX_COLORS) // !defined(SHOW_TET_QUALITY)

    vec4 fragment0Color, fragment1Color;
    float fragment0Depth, fragment1Depth;
    bool fragment0Boundary, fragment1Boundary;
    bool fragment0FrontFace, fragment1FrontFace;
    float t, tSeg;
    getNextFragment(0, fragsCount, fragment1Color, fragment1Depth, fragment1Boundary, fragment1FrontFace);

    for (i = 1; i < fragsCount; i++) {
#ifdef USE_TERMINATION_INDEX
        if (rayColor.a > earlyRayTerminationAlpha) {
            break;
        }
#endif

        // Load the new fragment.
        fragment0Color = fragment1Color;
        fragment0Depth = fragment1Depth;
        fragment0Boundary = fragment1Boundary;
        fragment0FrontFace = fragment1FrontFace;
        getNextFragment(i, fragsCount, fragment1Color, fragment1Depth, fragment1Boundary, fragment1FrontFace);

        // Skip if the closest fragment is a boundary face.
        if ((fragment0Boundary && !fragment0FrontFace) && (fragment1Boundary && fragment1FrontFace)) {
            continue;
        }

        // Compute the accumulated color of the fragments.
        t = fragment1Depth - fragment0Depth;

#ifdef USE_SUBDIVS
        tSeg = t / float(NUM_SUBDIVS);
        const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
        for (int s = 0; s < NUM_SUBDIVS; s++) {
            float fbegin = (float(s)) * INV_N_SUB;
            float fmid = (float(s) + 0.5) * INV_N_SUB;
            float fend = (float(s) + 1.0) * INV_N_SUB;
            vec3 c0 = mix(fragment0Color.rgb, fragment1Color.rgb, fbegin);
            vec3 c1 = mix(fragment0Color.rgb, fragment1Color.rgb, fend);
            float alpha = mix(fragment0Color.a, fragment1Color.a, fmid);
            currentColor = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient);
            rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
            rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
        }
#else
        currentColor = accumulateLinear(
                t, fragment0Color.rgb, fragment1Color.rgb,
                fragment0Color.a * attenuationCoefficient, fragment1Color.a * attenuationCoefficient);
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
#endif
    }

#else // !defined(SHOW_TET_QUALITY) && !defined(SHOW_TET_QUALITY)

    uvec2 fragmentTetIds;
    float fragmentDepth, lastFragmentDepth;
    bool fragmentBoundary;
    bool fragmentFrontFace;

    uint openTetId = INVALID_TET;
    for (i = 0; i < fragsCount; i++) {
        getNextFragment(
                i, fragsCount, fragmentTetIds, fragmentDepth, fragmentBoundary, fragmentFrontFace);
        bool eqA = fragmentTetIds.x != INVALID_TET && fragmentTetIds.x == openTetId;
        bool eqB = fragmentTetIds.y != INVALID_TET && fragmentTetIds.y == openTetId;
        if (eqA || eqB) {
            float t = fragmentDepth - lastFragmentDepth;
            vec4 tetColor = cellColors[openTetId];
            vec4 currentColor = accumulateConst(t, tetColor.rgb, tetColor.a * attenuationCoefficient);
            rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
            rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
        } else {
            eqA = fragmentTetIds.y != INVALID_TET;
        }
        openTetId = eqA ? fragmentTetIds.y : fragmentTetIds.x;
        lastFragmentDepth = fragmentDepth;
    }

#endif // SHOW_TET_QUALITY

#ifdef ALPHA_MODE_STRAIGHT
    if (rayColor.a > 1e-5) {
        rayColor.rgb = rayColor.rgb / rayColor.a; // Correct rgb with alpha
    }
#endif

#ifdef USE_TERMINATION_INDEX
    ivec2 workIdx = ivec2(gl_FragCoord.xy);
    imageStore(terminationIndexImage, workIdx, uvec4(i));
#endif

    return rayColor;
}
