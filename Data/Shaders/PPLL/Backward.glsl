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

#include "BackwardCommon.glsl"
layout(binding = 10, rgba32f) uniform readonly image2D colorImageOpt;
layout(binding = 11, rgba32f) uniform readonly image2D adjointColors;

layout(push_constant) uniform PushConstants {
    uint useAbsGrad;
};

void getNextFragment(
        in uint i, in uint fragsCount,
#ifndef PER_VERTEX_COLORS
        out uvec2 tetIds,
#else
        out vec4 color,
#endif
        out float depthLinear, out bool boundary, out bool frontFace,
        out uint i0, out uint i1, out uint i2, out vec3 p, out vec3 p0, out vec3 p1, out vec3 p2,
#ifdef PER_VERTEX_COLORS
        out vec4 c0, out vec4 c1, out vec4 c2,
#endif
        out float u, out float v
#ifdef USE_TERMINATION_INDEX
        , bool skipLayer
#endif
) {
    minHeapSink4(0, fragsCount - i);
    uint faceBits = colorList[0];
    float depthBufferValue = depthList[0];
    colorList[0] = colorList[fragsCount - i - 1];
    depthList[0] = depthList[fragsCount - i - 1];
#ifdef USE_TERMINATION_INDEX
    if (skipLayer) {
        return;
    }
#endif

    depthLinear = convertDepthBufferValueToLinearDepth(depthBufferValue);
    frontFace = (faceBits & 1u) == 1u ? true : false;
    boundary = ((faceBits >> 1u) & 1u) == 1u ? true : false;
    uint faceIndex = (faceBits >> 2u) * 3u;

#ifndef PER_VERTEX_COLORS
    uint faceIndexTri = faceBits >> 2u;
    tetIds = faceToTetMap[faceIndexTri];
#endif

    // Compute world space position from depth.
#ifndef COMPUTE_SHADER // TODO
    vec2 fragCoord = gl_FragCoord.xy;
#endif
    vec4 fragPosNdc = vec4(2.0 * gl_FragCoord.xy / vec2(viewportSize) - vec2(1.0), depthBufferValue, 1.0);
    vec4 fragPosWorld = inverseViewProjectionMatrix * fragPosNdc;
    p = fragPosWorld.xyz / fragPosWorld.w;

    i0 = triangleIndices[faceIndex];
    i1 = triangleIndices[faceIndex + 1];
    i2 = triangleIndices[faceIndex + 2];
    p0 = vertexPositions[i0];
    p1 = vertexPositions[i1];
    p2 = vertexPositions[i2];
#ifdef PER_VERTEX_COLORS
    c0 = vertexColors[i0];
    c1 = vertexColors[i1];
    c2 = vertexColors[i2];
#endif

    // Barycentric interpolation.
    vec3 d20 = p2 - p0;
    vec3 d21 = p2 - p1;
    float totalArea = max(length(cross(d20, d21)), 1e-6);
    u = length(cross(d21, p - p1)) / totalArea;
    v = length(cross(p - p0, d20)) / totalArea;
    const vec3 barycentricCoordinates = vec3(u, v, 1.0 - u - v);
#ifdef PER_VERTEX_COLORS
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

    ivec2 workIdx = ivec2(gl_FragCoord.xy);

#ifdef USE_TERMINATION_INDEX
    uint terminationIndex = fragsCount - imageLoad(terminationIndexImage, workIdx).x;
#endif
    vec4 colorRayOut = imageLoad(colorImageOpt, workIdx);
    vec4 dOut_dColorRayOut = imageLoad(adjointColors, workIdx);

    uint if00, if01, if02, if10, if11, if12;
    vec3 pf0, pf00, pf01, pf02, pf1, pf10, pf11, pf12;
    float uf0, vf0, uf1, vf1;

#ifdef PER_VERTEX_COLORS

    vec4 fragment0Color, fragment1Color;
    float fragment0Depth, fragment1Depth;
    bool fragment0Boundary, fragment1Boundary;
    bool fragment0FrontFace, fragment1FrontFace;
    vec4 cf00, cf01, cf02, cf10, cf11, cf12;
    getNextFragment(
            0, fragsCount, fragment0Color, fragment0Depth, fragment0Boundary, fragment0FrontFace,
            if00, if01, if02, pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0
#ifdef USE_TERMINATION_INDEX
            , terminationIndex != 0u
#endif
    );

#else // !defined(PER_VERTEX_COLORS)

    uvec2 fragmentTetIds, lastFragmentTetIds = uvec2(INVALID_TET, INVALID_TET);
    float fragmentDepth, lastFragmentDepth;
    bool fragmentBoundary;
    bool fragmentFrontFace;
    uint openTetId = INVALID_TET;

#endif // PER_VERTEX_COLORS

    /*if (isnan(dOut_dColorRayOut.x) || isnan(dOut_dColorRayOut.y) || isnan(dOut_dColorRayOut.z) || isnan(dOut_dColorRayOut.w)
            //|| dOut_dColorRayOut.x != 0.0 || dOut_dColorRayOut.y != 0.0 || dOut_dColorRayOut.z != 0.0 || dOut_dColorRayOut.w != 0.0
    ) {
        debugPrintfEXT("n %i %i %f %f %f %f", workIdx.x, workIdx.y, dOut_dColorRayOut.x, dOut_dColorRayOut.y, dOut_dColorRayOut.z, dOut_dColorRayOut.w);
        dOut_dColorRayOut = vec4(0.0);
    }*/

    // Start with transparent Ray
    vec4 colorAcc;
    float A;
    float t;
#ifdef PER_VERTEX_COLORS
    float tSeg;
    for (i = 1; i < fragsCount; i++)
#else
    for (i = 0; i < fragsCount; i++)
#endif
    {
        // Load the new fragment.
#ifdef PER_VERTEX_COLORS
#ifdef USE_TERMINATION_INDEX
        if (i > terminationIndex) {
#endif
        fragment1Color = fragment0Color;
        fragment1Depth = fragment0Depth;
        fragment1Boundary = fragment0Boundary;
        fragment1FrontFace = fragment0FrontFace;
        if10 = if00;
        if11 = if01;
        if12 = if02;
        pf1 = pf0;
        pf10 = pf00;
        pf11 = pf01;
        pf12 = pf02;
        cf10 = cf00;
        cf11 = cf01;
        cf12 = cf02;
        uf1 = uf0;
        vf1 = vf0;
#ifdef USE_TERMINATION_INDEX
        }
#endif
        getNextFragment(
                i, fragsCount, fragment0Color, fragment0Depth, fragment0Boundary, fragment0FrontFace,
                if00, if01, if02, pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0
#ifdef USE_TERMINATION_INDEX
                , i < terminationIndex
#endif
        );

#ifdef USE_TERMINATION_INDEX
        if (i <= terminationIndex) {
            continue;
        }
#endif

        // Skip if the closest fragment is a boundary face.
        if ((fragment0Boundary && !fragment0FrontFace) && (fragment1Boundary && fragment1FrontFace)) {
            continue;
        }

        vec4 dOut_dcf0 = vec4(0.0);
        vec4 dOut_dcf1 = vec4(0.0);
        float dOut_dt = 0.0;

        // Compute the accumulated color of the fragments.
        t = fragment1Depth - fragment0Depth;
        tSeg = t / float(NUM_SUBDIVS);
        const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
        for (int s = NUM_SUBDIVS - 1; s > 0; s--) {
            float fbegin = (float(s)) * INV_N_SUB;
            float fmid = (float(s) + 0.5) * INV_N_SUB;
            float fend = (float(s) + 1.0) * INV_N_SUB;
            vec3 c0 = mix(fragment0Color.rgb, fragment1Color.rgb, fbegin);
            vec3 c1 = mix(fragment0Color.rgb, fragment1Color.rgb, fend);
            float alpha = mix(fragment0Color.a, fragment1Color.a, fmid);
            colorAcc = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient, A);

            // Inversion trick from "Differentiable Direct Volume Rendering", Weiß et al. 2021.
            float alphaRayIn = (colorAcc.a - colorRayOut.a) / (colorAcc.a - 1.0);
            vec3 colorRayIn = colorRayOut.rgb - (1.0 - alphaRayIn) * colorAcc.rgb;
            //colorRayOut.a = colorRayIn.a + (1.0 - colorRayIn.a) * colorAcc.a;
            //colorRayOut.rgb = colorRayIn.rgb + (1.0 - colorRayIn.a) * colorAcc.rgb;

            // Compute adjoint for accumulated color/opacity.
            vec4 dOut_dColorAcc;
            dOut_dColorAcc.rgb = (1.0 - alphaRayIn) * dOut_dColorRayOut.rgb;
            dOut_dColorAcc.a = (1.0 - alphaRayIn) * dOut_dColorRayOut.a;

            // Backpropagation for the accumulated color.
            // colorCurrAdjoint.rgb stays the same (see paper cited above, Chat^(i) = Chat^(i+1)).
            float alphaNewAdjoint = dOut_dColorRayOut.a * (1.0 - colorAcc.a) - dot(dOut_dColorRayOut.rgb, colorAcc.rgb);
            dOut_dColorRayOut.a = alphaNewAdjoint;
            colorRayOut = vec4(colorRayIn, alphaRayIn);

            /*if (isnan(dOut_dColorRayOut.x) || isnan(dOut_dColorRayOut.y) || isnan(dOut_dColorRayOut.z) || isnan(dOut_dColorRayOut.w)
                    //|| dOut_dColorRayOut.x != 0.0 || dOut_dColorRayOut.y != 0.0 || dOut_dColorRayOut.z != 0.0 || dOut_dColorRayOut.w != 0.0
            ) {
                debugPrintfEXT("k %i %f %f %f %f", s, dOut_dColorRayOut.x, dOut_dColorRayOut.y, dOut_dColorRayOut.z, dOut_dColorRayOut.w);
                dOut_dColorRayOut = vec4(0.0);
            }*/

            // Compute adjoint for the pre-accumulation colors and opacity.
            vec3 dOut_dc0;
            vec3 dOut_dc1;
            float dOut_da;
            accumulateLinearConstAdjoint(
                    tSeg, INV_N_SUB, c0, c1, alpha * attenuationCoefficient, A, dOut_dColorAcc,
                    dOut_dt, dOut_dc0, dOut_dc1, dOut_da);
            dOut_da *= attenuationCoefficient;

            // Backpropagate gradients wrt. segmented color/opacity to intersection point color/opacity.
            // dc0_dcf0 = (1.0 - fbegin), dc1_dcf0 = (1.0 - fend), dc0_dcf1 = fbegin, dc1_dcf1 = fend.
            dOut_dcf0 += vec4((1.0 - fbegin) * dOut_dc0 + (1.0 - fend) * dOut_dc1, (1.0 - fmid) * dOut_da);
            dOut_dcf1 += vec4(fbegin * dOut_dc0 + fend * dOut_dc1, fmid * dOut_da);

            /*if (isnan(dOut_dcf0.x) || isnan(dOut_dcf0.y) || isnan(dOut_dcf0.z) || isnan(dOut_dcf0.w)
                    //|| dOut_dcf0.x != 0.0 || dOut_dcf0.y != 0.0 || dOut_dcf0.z != 0.0 || dOut_dcf0.w != 0.0
            ) {
                debugPrintfEXT("p %u %f %f %f %f", if00, dOut_dcf0.x, dOut_dcf0.y, dOut_dcf0.z, dOut_dcf0.w);
                dOut_dcf0 = vec4(0.0);
            }
            if (isnan(dOut_dcf1.x) || isnan(dOut_dcf1.y) || isnan(dOut_dcf1.z) || isnan(dOut_dcf1.w)
                    //|| dOut_dcf1.x != 0.0 || dOut_dcf1.y != 0.0 || dOut_dcf1.z != 0.0 || dOut_dcf1.w != 0.0
            ) {
                debugPrintfEXT("q %u %f %f %f %f", if00, dOut_dcf1.x, dOut_dcf1.y, dOut_dcf1.z, dOut_dcf1.w);
                dOut_dcf1 = vec4(0.0);
            }*/
        }

#else // !defined(PER_VERTEX_COLORS)

        getNextFragment(
                i, fragsCount, fragmentTetIds, fragmentDepth, fragmentBoundary, fragmentFrontFace,
                if00, if01, if02, pf0, pf00, pf01, pf02, uf0, vf0
#ifdef USE_TERMINATION_INDEX
                , i < terminationIndex
#endif
        );
#ifdef USE_TERMINATION_INDEX
        if (i < terminationIndex) {
            continue;
        }
#endif
        openTetId = tetIdUnion(fragmentTetIds, lastFragmentTetIds);
        // tetIdUnion is more stable when missing some fragments, and is the only solution for USE_TERMINATION_INDEX.
        //bool eqA = fragmentTetIds.x != INVALID_TET && fragmentTetIds.x == openTetId;
        //bool eqB = fragmentTetIds.y != INVALID_TET && fragmentTetIds.y == openTetId;
        //if (eqA || eqB) {
        if (openTetId != INVALID_TET) {
            float t = lastFragmentDepth - fragmentDepth;
            vec4 c = cellColors[openTetId];
            colorAcc = accumulateConst(t, c.rgb, c.a * attenuationCoefficient, A);

            // Inversion trick from "Differentiable Direct Volume Rendering", Weiß et al. 2021.
            float alphaRayIn = (colorAcc.a - colorRayOut.a) / (colorAcc.a - 1.0);
            vec3 colorRayIn = colorRayOut.rgb - (1.0 - alphaRayIn) * colorAcc.rgb;
            //colorRayOut.a = colorRayIn.a + (1.0 - colorRayIn.a) * colorAcc.a;
            //colorRayOut.rgb = colorRayIn.rgb + (1.0 - colorRayIn.a) * colorAcc.rgb;

            // Compute adjoint for accumulated color/opacity.
            vec4 dOut_dColorAcc;
            dOut_dColorAcc.rgb = (1.0 - alphaRayIn) * dOut_dColorRayOut.rgb;
            dOut_dColorAcc.a = (1.0 - alphaRayIn) * dOut_dColorRayOut.a;

            // Backpropagation for the accumulated color.
            // colorCurrAdjoint.rgb stays the same (see paper cited above, Chat^(i) = Chat^(i+1)).
            float alphaNewAdjoint = dOut_dColorRayOut.a * (1.0 - colorAcc.a) - dot(dOut_dColorRayOut.rgb, colorAcc.rgb);
            dOut_dColorRayOut.a = alphaNewAdjoint;
            colorRayOut = vec4(colorRayIn, alphaRayIn);

            // Compute adjoint for the pre-accumulation colors and opacity.
            float dOut_dt = 0.0;
            vec4 dOut_dc = vec4(0.0);
            accumulateConstAdjoint(t, c.rgb, c.a * attenuationCoefficient, A, dOut_dColorAcc, dOut_dt, dOut_dc);
            dOut_dc.a *= attenuationCoefficient;

#endif // PER_VERTEX_COLORS

        vec3 dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12;
        vec4 dOut_dcf00, dOut_dcf01, dOut_dcf02, dOut_dcf10, dOut_dcf11, dOut_dcf12;
        float dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1;
        segmentLengthAdjoint(
                pf00, pf01, pf02, pf10, pf11, pf12, uf0, vf0, uf1, vf1, dOut_dt,
                dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1,
                dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12);
        //vec3 dOut_dpf00 = vec3(0.0), dOut_dpf01 = vec3(0.0), dOut_dpf02 = vec3(0.0), dOut_dpf10 = vec3(0.0), dOut_dpf11 = vec3(0.0), dOut_dpf12 = vec3(0.0);
        //vec4 dOut_dcf00, dOut_dcf01, dOut_dcf02, dOut_dcf10, dOut_dcf11, dOut_dcf12;
        //float dOut_duf0 = 0.0, dOut_dvf0 = 0.0, dOut_duf1 = 0.0, dOut_dvf1 = 0.0;
        baryAdjoint(
                pf0, pf00, pf01, pf02,
#ifdef PER_VERTEX_COLORS
                cf00, cf01, cf02,
#endif
                uf0, vf0,
#ifdef PER_VERTEX_COLORS
                dOut_dcf0,
#endif
                dOut_duf0, dOut_dvf0,
                dOut_dpf00, dOut_dpf01, dOut_dpf02
#ifdef PER_VERTEX_COLORS
                , dOut_dcf00, dOut_dcf01, dOut_dcf02
#endif
        );
        baryAdjoint(
                pf1, pf10, pf11, pf12,
#ifdef PER_VERTEX_COLORS
                cf10, cf11, cf12,
#endif
                uf1, vf1,
#ifdef PER_VERTEX_COLORS
                dOut_dcf1,
#endif
                dOut_duf1, dOut_dvf1,
                dOut_dpf10, dOut_dpf11, dOut_dpf12
#ifdef PER_VERTEX_COLORS
                , dOut_dcf10, dOut_dcf11, dOut_dcf12
#endif
        );

        /*if (isnan(dOut_dcf00.x) || isnan(dOut_dcf00.y) || isnan(dOut_dcf00.z) || isnan(dOut_dcf00.w)
                //|| dOut_dcf00.x != 0.0 || dOut_dcf00.y != 0.0 || dOut_dcf00.z != 0.0 || dOut_dcf00.w != 0.0
        ) {
            debugPrintfEXT("a %u %f %f %f %f", if00, dOut_dcf00.x, dOut_dcf00.y, dOut_dcf00.z, dOut_dcf00.w);
            dOut_dcf00 = vec4(0.0);
        }
        if (isnan(dOut_dcf01.x) || isnan(dOut_dcf01.y) || isnan(dOut_dcf01.z) || isnan(dOut_dcf01.w)
                //|| dOut_dcf01.x != 0.0 || dOut_dcf01.y != 0.0 || dOut_dcf01.z != 0.0 || dOut_dcf01.w != 0.0
        ) {
            debugPrintfEXT("b %u %f %f %f %f", if01, dOut_dcf01.x, dOut_dcf01.y, dOut_dcf01.z, dOut_dcf01.w);
            dOut_dcf01 = vec4(0.0);
        }
        if (isnan(dOut_dcf02.x) || isnan(dOut_dcf02.y) || isnan(dOut_dcf02.z) || isnan(dOut_dcf02.w)
        //|| dOut_dcf02.x != 0.0 || dOut_dcf02.y != 0.0 || dOut_dcf02.z != 0.0 || dOut_dcf02.w != 0.0
        ) {
            debugPrintfEXT("c %u %f %f %f %f", if02, dOut_dcf02.x, dOut_dcf02.y, dOut_dcf02.z, dOut_dcf02.w);
            dOut_dcf02 = vec4(0.0);
        }
        if (isnan(dOut_dcf10.x) || isnan(dOut_dcf10.y) || isnan(dOut_dcf10.z) || isnan(dOut_dcf10.w)
        //|| dOut_dcf10.x != 0.0 || dOut_dcf10.y != 0.0 || dOut_dcf10.z != 0.0 || dOut_dcf10.w != 0.0
        ) {
            debugPrintfEXT("d %u %f %f %f %f", if10, dOut_dcf10.x, dOut_dcf10.y, dOut_dcf10.z, dOut_dcf10.w);
            dOut_dcf10 = vec4(0.0);
        }
        if (isnan(dOut_dcf11.x) || isnan(dOut_dcf11.y) || isnan(dOut_dcf11.z) || isnan(dOut_dcf11.w)
        //|| dOut_dcf11.x != 0.0 || dOut_dcf11.y != 0.0 || dOut_dcf11.z != 0.0 || dOut_dcf11.w != 0.0
        ) {
            debugPrintfEXT("e %u %f %f %f %f", if11, dOut_dcf11.x, dOut_dcf11.y, dOut_dcf11.z, dOut_dcf11.w);
            dOut_dcf11 = vec4(0.0);
        }
        if (isnan(dOut_dcf12.x) || isnan(dOut_dcf12.y) || isnan(dOut_dcf12.z) || isnan(dOut_dcf12.w)
        //|| dOut_dcf12.x != 0.0 || dOut_dcf12.y != 0.0 || dOut_dcf12.z != 0.0 || dOut_dcf12.w != 0.0
        ) {
            debugPrintfEXT("f %u %f %f %f %f", if12, dOut_dcf12.x, dOut_dcf12.y, dOut_dcf12.z, dOut_dcf12.w);
            dOut_dcf12 = vec4(0.0);
        }

        if (isnan(dOut_dpf00.x) || isnan(dOut_dpf00.y) || isnan(dOut_dpf00.z)
        //|| dOut_dpf00.x != 0.0 || dOut_dpf00.y != 0.0 || dOut_dpf00.z != 0.0
        ) {
            debugPrintfEXT("g %u %f %f %f", if00, dOut_dpf00.x, dOut_dpf00.y, dOut_dpf00.z);
            dOut_dpf00 = vec3(0.0);
        }
        if (isnan(dOut_dpf01.x) || isnan(dOut_dpf01.y) || isnan(dOut_dpf01.z)
        //|| dOut_dpf01.x != 0.0 || dOut_dpf01.y != 0.0 || dOut_dpf01.z != 0.0
        ) {
            debugPrintfEXT("h %u %f %f %f", if01, dOut_dpf01.x, dOut_dpf01.y, dOut_dpf01.z);
            dOut_dpf01 = vec3(0.0);
        }
        if (isnan(dOut_dpf02.x) || isnan(dOut_dpf02.y) || isnan(dOut_dpf02.z)
        //|| dOut_dpf02.x != 0.0 || dOut_dpf02.y != 0.0 || dOut_dpf02.z != 0.0
        ) {
            debugPrintfEXT("i %u %f %f %f", if02, dOut_dpf02.x, dOut_dpf02.y, dOut_dpf02.z);
            dOut_dpf02 = vec3(0.0);
        }
        if (isnan(dOut_dpf10.x) || isnan(dOut_dpf10.y) || isnan(dOut_dpf10.z)
        //|| dOut_dpf10.x != 0.0 || dOut_dpf10.y != 0.0 || dOut_dpf10.z != 0.0
        ) {
            debugPrintfEXT("j %u %f %f %f", if10, dOut_dpf10.x, dOut_dpf10.y, dOut_dpf10.z);
            dOut_dpf10 = vec3(0.0);
        }
        if (isnan(dOut_dpf11.x) || isnan(dOut_dpf11.y) || isnan(dOut_dpf11.z)
        //|| dOut_dpf11.x != 0.0 || dOut_dpf11.y != 0.0 || dOut_dpf11.z != 0.0
        ) {
            debugPrintfEXT("k %u %f %f %f", if11, dOut_dpf11.x, dOut_dpf11.y, dOut_dpf11.z);
            dOut_dpf11 = vec3(0.0);
        }
        if (isnan(dOut_dpf12.x) || isnan(dOut_dpf12.y) || isnan(dOut_dpf12.z)
        //|| dOut_dpf12.x != 0.0 || dOut_dpf12.y != 0.0 || dOut_dpf12.z != 0.0
        ) {
            debugPrintfEXT("l %u %f %f %f", if12, dOut_dpf12.x, dOut_dpf12.y, dOut_dpf12.z);
            dOut_dpf12 = vec3(0.0);
        }*/

        /*
         * For testing an idea similar to the one from the following 3DGS paper:
         * "AbsGS: Recovering Fine Details for 3D Gaussian Splatting". 2024.
         * Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, Yong Dou.
         */
        if (useAbsGrad != 0u) {
#ifdef PER_VERTEX_COLORS
            dOut_dcf00 = abs(dOut_dcf00);
            dOut_dcf01 = abs(dOut_dcf01);
            dOut_dcf02 = abs(dOut_dcf02);
            dOut_dcf10 = abs(dOut_dcf10);
            dOut_dcf11 = abs(dOut_dcf11);
            dOut_dcf12 = abs(dOut_dcf12);
#else
            dOut_dc = abs(dOut_dc);
#endif
            dOut_dpf00 = abs(dOut_dpf00);
            dOut_dpf01 = abs(dOut_dpf01);
            dOut_dpf02 = abs(dOut_dpf02);
            dOut_dpf10 = abs(dOut_dpf10);
            dOut_dpf11 = abs(dOut_dpf11);
            dOut_dpf12 = abs(dOut_dpf12);
        }

        // Accumulate gradients wrt. tet properties.
#ifdef PER_VERTEX_COLORS
        atomicAddGradCol(if00, dOut_dcf00);
        atomicAddGradCol(if01, dOut_dcf01);
        atomicAddGradCol(if02, dOut_dcf02);

        atomicAddGradCol(if10, dOut_dcf10);
        atomicAddGradCol(if11, dOut_dcf11);
        atomicAddGradCol(if12, dOut_dcf12);
#else
        atomicAddGradCol(openTetId, dOut_dc);
#endif

        atomicAddGradPos(if00, dOut_dpf00);
        atomicAddGradPos(if01, dOut_dpf01);
        atomicAddGradPos(if02, dOut_dpf02);

        atomicAddGradPos(if10, dOut_dpf10);
        atomicAddGradPos(if11, dOut_dpf11);
        atomicAddGradPos(if12, dOut_dpf12);

#ifndef PER_VERTEX_COLORS
        } /*else {
            eqA = fragmentTetIds.y != INVALID_TET;
        }*/
        //openTetId = eqA ? fragmentTetIds.y : fragmentTetIds.x;
        lastFragmentDepth = fragmentDepth;
        lastFragmentTetIds = fragmentTetIds;
        if10 = if00;
        if11 = if01;
        if12 = if02;
        pf1 = pf0;
        pf10 = pf00;
        pf11 = pf01;
        pf12 = pf02;
        uf1 = uf0;
        vf1 = vf0;
#endif
    }

    return vec4(0.0);
}
