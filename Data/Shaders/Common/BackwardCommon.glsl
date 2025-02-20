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

layout(binding = 8, scalar) coherent buffer VertexPositionGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexPositionGradients[]; // stride: vec3
#else
    uint vertexPositionGradients[]; // stride: vec3
#endif
};

#ifdef PER_VERTEX_COLORS
layout(binding = 9, scalar) coherent buffer VertexColorGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexColorGradients[]; // stride: vec4
#else
    uint vertexColorGradients[]; // stride: vec4
#endif
};
#else
layout(binding = 9, scalar) coherent buffer CellColorGradientBuffer {
    // We also use the name "vertexColorGradients" here, so we can use the same name in the atomic functions.
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexColorGradients[]; // stride: vec4
#else
    uint vertexColorGradients[]; // stride: vec4
#endif
};
#endif

void atomicAddGradCol(uint idx, vec4 value) {
    const uint stride = idx * 4;
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(vertexColorGradients[stride], value.r);
    atomicAdd(vertexColorGradients[stride + 1], value.g);
    atomicAdd(vertexColorGradients[stride + 2], value.b);
    atomicAdd(vertexColorGradients[stride + 3], value.a);
#else
    [[unroll]] for (uint i = 0; i < 4; i++) {
        uint oldValue = vertexColorGradients[stride + i];
        uint expectedValue, newValue;
        do {
            expectedValue = oldValue;
            newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value[i]);
            oldValue = atomicCompSwap(vertexColorGradients[stride + i], expectedValue, newValue);
        } while (oldValue != expectedValue);
    }
#endif
}

void atomicAddGradPos(uint idx, vec3 value) {
    const uint stride = idx * 3;
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(vertexPositionGradients[stride], value.x);
    atomicAdd(vertexPositionGradients[stride + 1], value.y);
    atomicAdd(vertexPositionGradients[stride + 2], value.z);
#else
    [[unroll]] for (uint i = 0; i < 3; i++) {
        uint oldValue = vertexPositionGradients[stride + i];
        uint expectedValue, newValue;
        do {
            expectedValue = oldValue;
            newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value[i]);
            oldValue = atomicCompSwap(vertexPositionGradients[stride + i], expectedValue, newValue);
        } while (oldValue != expectedValue);
    }
#endif
}

vec4 accumulateConst(float t, vec3 c, float a, out float A) {
    A = exp(-a * t);
    return vec4((1.0 - A) * c, 1.0 - A);
}

void accumulateConstAdjoint(
        float t, vec3 c, float a, float A, vec4 dOut_dC,
        inout float dOut_dt, out vec4 dOut_dc) {
    dOut_dt += (a * A) * (dot(dOut_dC.rgb, c) + dOut_dC.a);
    dOut_dc = vec4((1.0 - A) * dOut_dC.rgb, (t * A) * (dot(dOut_dC.rgb, c) + dOut_dC.a));
}

vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a, out float A) {
    A = exp(-a * t);
    if (a < 1e-6) {
        // lim a->0 (t + 1.0 / a) * A - 1.0 / a) = 0
        return vec4(0.0);
    }
    return vec4((1.0 - A) * c0 + ((t + 1.0 / a) * A - 1.0 / a) * (c0 - c1), 1.0 - A);
}

void accumulateLinearConstAdjoint(
        float t, float tf, vec3 c0, vec3 c1, float a, float A, vec4 dOut_dC,
        inout float dOut_dt, out vec3 dOut_dc0, out vec3 dOut_dc1, out float dOut_da) {
    dOut_dt += tf * (dot(dOut_dC.rgb, (a * c0 + (a * c1 - a * c0) * t) * A) + dOut_dC.a * a * A);
    if (a < 1e-6) {
        // For c0 and c1: lim a->0 (t + 1.0 / a) * A - 1.0 / a) = 0
        dOut_dc0 = vec3(0.0);
        dOut_dc1 = vec3(0.0);
        dOut_da = dot(dOut_dC.rgb, t * c0 + 0.5 * t * t * (c1 - c0)) + dOut_dC.a * t;
    } else {
        const float inva = 1.0 / a;
        dOut_dc0 = ((1.0 - A) + ((t + inva) * A - inva)) * dOut_dC.rgb;
        dOut_dc1 = -((t + inva) * A - inva) * dOut_dC.rgb;
        dOut_da =
                dot(dOut_dC.rgb, t * A * c0 + (inva * inva - (t * t + inva * inva + t * inva) * A) * (c0 - c1))
                + dOut_dC.a * t * A;
    }
}

float pow2(float x) {
    return x * x;
}

void segmentLengthAdjoint(
        vec3 pf00, vec3 pf01, vec3 pf02, vec3 pf10, vec3 pf11, vec3 pf12, float uf0, float vf0, float uf1, float vf1,
        float dOut_dt, out float dOut_duf0, out float dOut_dvf0, out float dOut_duf1, out float dOut_dvf1,
        out vec3 dOut_dpf00, out vec3 dOut_dpf01, out vec3 dOut_dpf02,
        out vec3 dOut_dpf10, out vec3 dOut_dpf11, out vec3 dOut_dpf12) {
    float wf0 = 1.0 - uf0 - vf0;
    float wf1 = 1.0 - uf1 - vf1;
    float denomSq = pow2(pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) + pow2(pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) + pow2(pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1);
    float invDenom = 1.0 / sqrt(max(denomSq, 1e-6));
    float dt_duf0 = ((pf00.x - pf02.x)*(pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) + (pf00.y - pf02.y)*(pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) + (pf00.z - pf02.z)*(pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1)) * invDenom;
    float dt_dvf0 = ((pf01.x - pf02.x)*(pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) + (pf01.y - pf02.y)*(pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) + (pf01.z - pf02.z)*(pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1)) * invDenom;
    float dt_duf1 = (-(pf10.x - pf12.x)*(pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) - (pf10.y - pf12.y)*(pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) - (pf10.z - pf12.z)*(pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1)) * invDenom;
    float dt_dvf1 = (-(pf11.x - pf12.x)*(pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) - (pf11.y - pf12.y)*(pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) - (pf11.z - pf12.z)*(pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1)) * invDenom;
    float dt_dpf00x = uf0 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf00y = uf0 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf00z = uf0 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;
    float dt_dpf01x = vf0 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf01y = vf0 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf01z = vf0 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;
    float dt_dpf02x = wf0 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf02y = wf0 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf02z = wf0 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;
    float dt_dpf10x = -uf1 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf10y = -uf1 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf10z = -uf1 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;
    float dt_dpf11x = -vf1 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf11y = -vf1 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf11z = -vf1 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;
    float dt_dpf12x = -wf1 * (pf00.x*uf0 + pf01.x*vf0 + pf02.x*wf0 - pf10.x*uf1 - pf11.x*vf1 - pf12.x*wf1) * invDenom;
    float dt_dpf12y = -wf1 * (pf00.y*uf0 + pf01.y*vf0 + pf02.y*wf0 - pf10.y*uf1 - pf11.y*vf1 - pf12.y*wf1) * invDenom;
    float dt_dpf12z = -wf1 * (pf00.z*uf0 + pf01.z*vf0 + pf02.z*wf0 - pf10.z*uf1 - pf11.z*vf1 - pf12.z*wf1) * invDenom;

    dOut_duf0 = dOut_dt * dt_duf0;
    dOut_dvf0 = dOut_dt * dt_dvf0;
    dOut_duf1 = dOut_dt * dt_duf1;
    dOut_dvf1 = dOut_dt * dt_dvf1;
    dOut_dpf00 = dOut_dt * vec3(dt_dpf00x, dt_dpf00y, dt_dpf00z);
    dOut_dpf01 = dOut_dt * vec3(dt_dpf01x, dt_dpf01y, dt_dpf01z);
    dOut_dpf02 = dOut_dt * vec3(dt_dpf02x, dt_dpf02y, dt_dpf02z);
    dOut_dpf10 = dOut_dt * vec3(dt_dpf10x, dt_dpf10y, dt_dpf10z);
    dOut_dpf11 = dOut_dt * vec3(dt_dpf11x, dt_dpf11y, dt_dpf11z);
    dOut_dpf12 = dOut_dt * vec3(dt_dpf12x, dt_dpf12y, dt_dpf12z);
}

void baryAdjoint(
        vec3 p, vec3 p0, vec3 p1, vec3 p2,
#ifdef PER_VERTEX_COLORS
        vec4 c0, vec4 c1, vec4 c2,
#endif
#ifdef PROJECTED_RASTER
        float d0, float d1, float d2,
#endif
        float u, float v,
#ifdef PER_VERTEX_COLORS
        vec4 dOut_dc,
#endif
#ifdef PROJECTED_RASTER
        float dOut_dd,
#endif
        float dOut_du, float dOut_dv, // forwarded from segmentLengthAdjoint
        inout vec3 dOut_dp0, inout vec3 dOut_dp1, inout vec3 dOut_dp2
#ifdef PER_VERTEX_COLORS
        , out vec4 dOut_dc0, out vec4 dOut_dc1, out vec4 dOut_dc2
#endif
#ifdef PROJECTED_RASTER
        , float dOut_dd0, float dOut_dd1, float dOut_dd2
#endif
) {
    float f0 = pow2((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + pow2((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)) + pow2((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y));
    float f1 = pow2((p.x - p1.x)*(p1.y - p2.y) - (p.y - p1.y)*(p1.x - p2.x)) + pow2((p.x - p1.x)*(p1.z - p2.z) - (p.z - p1.z)*(p1.x - p2.x)) + pow2((p.y - p1.y)*(p1.z - p2.z) - (p.z - p1.z)*(p1.y - p2.y));
    float f2 = pow2((p.x - p0.x)*(p0.y - p2.y) - (p.y - p0.y)*(p0.x - p2.x)) + pow2((p.x - p0.x)*(p0.z - p2.z) - (p.z - p0.z)*(p0.x - p2.x)) + pow2((p.y - p0.y)*(p0.z - p2.z) - (p.z - p0.z)*(p0.y - p2.y));
    float denom0 = max(pow(f0, 3.0/2.0), 1e-6);
    float denom1 = max(sqrt(f1), 1e-6);
    float denom2 = max(sqrt(f2), 1e-6);
    float du_dp0x = (-(p1.y - p2.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p1.z - p2.z)*(-(p0.x - p2.x)*(p1.z - p2.z) + (p0.z - p2.z)*(p1.x - p2.x)))*denom1/denom0;
    float du_dp0y = ((p1.x - p2.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p1.z - p2.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*denom1/denom0;
    float du_dp0z = (-(p1.x - p2.x)*(-(p0.x - p2.x)*(p1.z - p2.z) + (p0.z - p2.z)*(p1.x - p2.x)) + (p1.y - p2.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*denom1/denom0;
    float du_dp1x = (-((p.y - p2.y)*((p.x - p1.x)*(p1.y - p2.y) - (p.y - p1.y)*(p1.x - p2.x)) + (p.z - p2.z)*((p.x - p1.x)*(p1.z - p2.z) - (p.z - p1.z)*(p1.x - p2.x)))*f0 + ((p0.y - p2.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p0.z - p2.z)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)))*(f1))/(denom1*denom0);
    float du_dp1y = (((p.x - p2.x)*((p.x - p1.x)*(p1.y - p2.y) - (p.y - p1.y)*(p1.x - p2.x)) - (p.z - p2.z)*((p.y - p1.y)*(p1.z - p2.z) - (p.z - p1.z)*(p1.y - p2.y)))*f0 - ((p0.x - p2.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p0.z - p2.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f1))/(denom1*denom0);
    float du_dp1z = (((p.x - p2.x)*((p.x - p1.x)*(p1.z - p2.z) - (p.z - p1.z)*(p1.x - p2.x)) + (p.y - p2.y)*((p.y - p1.y)*(p1.z - p2.z) - (p.z - p1.z)*(p1.y - p2.y)))*f0 - ((p0.x - p2.x)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)) + (p0.y - p2.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f1))/(denom1*denom0);
    float du_dp2x = (((p.y - p1.y)*((p.x - p1.x)*(p1.y - p2.y) - (p.y - p1.y)*(p1.x - p2.x)) + (p.z - p1.z)*((p.x - p1.x)*(p1.z - p2.z) - (p.z - p1.z)*(p1.x - p2.x)))*f0 - ((p0.y - p1.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p0.z - p1.z)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)))*(f1))/(denom1*denom0);
    float du_dp2y = ((-(p.x - p1.x)*((p.x - p1.x)*(p1.y - p2.y) - (p.y - p1.y)*(p1.x - p2.x)) + (p.z - p1.z)*((p.y - p1.y)*(p1.z - p2.z) - (p.z - p1.z)*(p1.y - p2.y)))*f0 + ((p0.x - p1.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p0.z - p1.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f1))/(denom1*denom0);
    float du_dp2z = (-((p.x - p1.x)*((p.x - p1.x)*(p1.z - p2.z) - (p.z - p1.z)*(p1.x - p2.x)) + (p.y - p1.y)*((p.y - p1.y)*(p1.z - p2.z) - (p.z - p1.z)*(p1.y - p2.y)))*f0 + ((p0.x - p1.x)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)) + (p0.y - p1.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f1))/(denom1*denom0);
    float dv_dp0x = ((-(p.y - p2.y)*((p.x - p0.x)*(p0.y - p2.y) - (p.y - p0.y)*(p0.x - p2.x)) - (p.z - p2.z)*((p.x - p0.x)*(p0.z - p2.z) - (p.z - p0.z)*(p0.x - p2.x)))*f0 - ((p1.y - p2.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p1.z - p2.z)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)))*(f2))/(denom2*denom0);
    float dv_dp0y = (((p.x - p2.x)*((p.x - p0.x)*(p0.y - p2.y) - (p.y - p0.y)*(p0.x - p2.x)) - (p.z - p2.z)*((p.y - p0.y)*(p0.z - p2.z) - (p.z - p0.z)*(p0.y - p2.y)))*f0 + ((p1.x - p2.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p1.z - p2.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f2))/(denom2*denom0);
    float dv_dp0z = (((p.x - p2.x)*((p.x - p0.x)*(p0.z - p2.z) - (p.z - p0.z)*(p0.x - p2.x)) + (p.y - p2.y)*((p.y - p0.y)*(p0.z - p2.z) - (p.z - p0.z)*(p0.y - p2.y)))*f0 + ((p1.x - p2.x)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)) + (p1.y - p2.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f2))/(denom2*denom0);
    float dv_dp1x = ((p0.y - p2.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p0.z - p2.z)*(-(p0.x - p2.x)*(p1.z - p2.z) + (p0.z - p2.z)*(p1.x - p2.x)))*denom2/denom0;
    float dv_dp1y = (-(p0.x - p2.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p0.z - p2.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*denom2/denom0;
    float dv_dp1z = ((p0.x - p2.x)*(-(p0.x - p2.x)*(p1.z - p2.z) + (p0.z - p2.z)*(p1.x - p2.x)) - (p0.y - p2.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*denom2/denom0;
    float dv_dp2x = (((p.y - p0.y)*((p.x - p0.x)*(p0.y - p2.y) - (p.y - p0.y)*(p0.x - p2.x)) + (p.z - p0.z)*((p.x - p0.x)*(p0.z - p2.z) - (p.z - p0.z)*(p0.x - p2.x)))*f0 - ((p0.y - p1.y)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) + (p0.z - p1.z)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)))*(f2))/(denom2*denom0);
    float dv_dp2y = ((-(p.x - p0.x)*((p.x - p0.x)*(p0.y - p2.y) - (p.y - p0.y)*(p0.x - p2.x)) + (p.z - p0.z)*((p.y - p0.y)*(p0.z - p2.z) - (p.z - p0.z)*(p0.y - p2.y)))*f0 + ((p0.x - p1.x)*((p0.x - p2.x)*(p1.y - p2.y) - (p0.y - p2.y)*(p1.x - p2.x)) - (p0.z - p1.z)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f2))/(denom2*denom0);
    float dv_dp2z = (-((p.x - p0.x)*((p.x - p0.x)*(p0.z - p2.z) - (p.z - p0.z)*(p0.x - p2.x)) + (p.y - p0.y)*((p.y - p0.y)*(p0.z - p2.z) - (p.z - p0.z)*(p0.y - p2.y)))*f0 + ((p0.x - p1.x)*((p0.x - p2.x)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.x - p2.x)) + (p0.y - p1.y)*((p0.y - p2.y)*(p1.z - p2.z) - (p0.z - p2.z)*(p1.y - p2.y)))*(f2))/(denom2*denom0);

    // p = u * p0 + v * p1 + (1 - u - v) * p2;
#ifdef PER_VERTEX_COLORS
    // c = u * c0 + v * c1 + (1 - u - v) * c2;
    dOut_du += dot(dOut_dc, c0 - c2 /* dc_du */);
    dOut_dv += dot(dOut_dc, c1 - c2 /* dc_dv */);
    dOut_dc0 = dOut_dc * u /* dc_c0 */;
    dOut_dc1 = dOut_dc * v /* dc_c1 */;
    dOut_dc2 = dOut_dc * (1.0 - u - v) /* dc_c2 */;
#endif
#ifdef PROJECTED_RASTER
    dOut_du += dot(dOut_dd, d0 - d2 /* dd_du */);
    dOut_dv += dot(dOut_dd, d1 - d2 /* dd_dv */);
    dOut_dd0 = dOut_dd * u /* dc_c0 */;
    dOut_dd1 = dOut_dd * v /* dc_c1 */;
    dOut_dd2 = dOut_dd * (1.0 - u - v) /* dc_c2 */;
#endif
    dOut_dp0 += dOut_du * vec3(du_dp0x, du_dp0y, du_dp0z) + dOut_dv * vec3(dv_dp0x, dv_dp0y, dv_dp0z);
    dOut_dp1 += dOut_du * vec3(du_dp1x, du_dp1y, du_dp1z) + dOut_dv * vec3(dv_dp1x, dv_dp1y, dv_dp1z);
    dOut_dp2 += dOut_du * vec3(du_dp2x, du_dp2y, du_dp2z) + dOut_dv * vec3(dv_dp2x, dv_dp2y, dv_dp2z);
}
