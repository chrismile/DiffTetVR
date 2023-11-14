layout(binding = 7, scalar) coherent buffer VertexPositionGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexPositionGradients[]; // stride: vec3
#else
    uint vertexPositionGradients[]; // stride: vec3
#endif
};
layout(binding = 8, scalar) coherent buffer VertexColorGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float vertexColorGradients[]; // stride: vec4
#else
    uint vertexColorGradients[]; // stride: vec4
#endif
};
layout(binding = 9, rgba32f) uniform readonly image2D colorImageOpt;
layout(binding = 10, rgba32f) uniform readonly image2D adjointColors;

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

vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a, out float A) {
    A = exp(-a * t);
    return vec4((1.0 - A) * c0 + ((t + 1.0 / a) * A - 1.0 / a) * (c0 - c1), 1.0 - A);
}

void accumulateLinearConstAdjoint(
        float t, float tf, vec3 c0, vec3 c1, float a, float A, vec4 dOut_dC,
        inout float dOut_dt, out vec3 dOut_dc0, out vec3 dOut_dc1, out float dOut_da) {
    const float inva = 1.0 / a;
    dOut_dt += tf * dot(dOut_dC.rgb, (a * c0 + (a * c1 - a * c0) * t) * A);
    dOut_dc0 = ((1.0 - A) + ((t + inva) * A - inva)) * dOut_dC.rgb;
    dOut_dc1 = -((t + inva) * A - inva) * dOut_dC.rgb;
    dOut_da =
            dot(dOut_dC.rgb, t * A * c0 + ((t - inva * inva) * A - (t + inva) * t * A + inva * inva) * (c0 - c1))
            + dOut_dC.a * t * A;
}

void getNextFragment(
        in uint i, in uint fragsCount, out vec4 color, out float depthLinear, out bool boundary, out bool frontFace,
        out uint i0, out uint i1, out uint i2, out vec3 p, out vec3 p0, out vec3 p1, out vec3 p2,
        out vec4 c0, out vec4 c1, out vec4 c2, out float u, out float v) {
    minHeapSink4(0, fragsCount - i);
    uint faceBits = colorList[0];
    float depthBufferValue = depthList[0];
    colorList[0] = colorList[fragsCount - i - 1];
    depthList[0] = depthList[fragsCount - i - 1];

    depthLinear = convertDepthBufferValueToLinearDepth(depthBufferValue);
    frontFace = (faceBits & 1u) == 1u ? true : false;
    boundary = ((faceBits >> 1u) & 1u) == 1u ? true : false;
    uint faceIndex = (faceBits >> 2u) * 3u;

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
    c0 = vertexColors[i0];
    c1 = vertexColors[i1];
    c2 = vertexColors[i2];

    // Barycentric interpolation.
    vec3 d20 = p2 - p0;
    vec3 d21 = p2 - p1;
    float totalArea = length(cross(d20, d21));
    u = length(cross(d21, p - p1)) / totalArea;
    v = length(cross(p - p0, d20)) / totalArea;
    const vec3 barycentricCoordinates = vec3(u, v, 1.0 - u - v);
    color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;
}

float pow2(float x) {
    return x * x;
}

void segmentLengthAdjoint(
        vec3 pf00, vec3 pf01, vec3 pf02, vec3 pf10, vec3 pf11, vec3 pf12, float uf0, float vf0, float uf1, float vf1,
        float dOut_dt, out float dOut_duf0, out float dOut_dvf0, out float dOut_duf1, out float dOut_dvf1,
        out vec3 dOut_dpf00, out vec3 dOut_dpf01, out vec3 dOut_dpf02,
        out vec3 dOut_dpf10, out vec3 dOut_dpf11, out vec3 dOut_dpf12) {
    float dt_duf0 = ((pf00.x - pf02.x) * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + (pf00.y - pf02.y) * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0))))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dvf0 = ((pf01.x - pf02.x) * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + (pf01.y - pf02.y) * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0))))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_duf1 = ((-(pf10.x - pf12.x) * (uf0 * pf00.x + ((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x)) - (pf10.y - pf12.y) * (uf0 * pf00.y + ((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y))) - (pf10.z - pf12.z) * (uf0 * pf00.z + ((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dvf1 = ((-(pf11.x - pf12.x) * (uf0 * pf00.x + ((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x)) - (pf11.y - pf12.y) * (uf0 * pf00.y + ((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y))) - (pf11.z - pf12.z) * (uf0 * pf00.z + ((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf00x = uf0 * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf00y = uf0 * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf00z = uf0 * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf01x = vf0 * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf01y = vf0 * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf01z = vf0 * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf02x = -(uf0 + (vf0 - 1.0)) * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf02y = -(uf0 + (vf0 - 1.0)) * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf02z = -(uf0 + (vf0 - 1.0)) * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf10x = -uf1 * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf10y = -uf1 * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf10z = -uf1 * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf11x = -vf1 * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf11y = -vf1 * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf11z = -vf1 * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf12x = (uf1 + (vf1 - 1.0)) * (uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf12y = (uf1 + (vf1 - 1.0)) * (uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));
    float dt_dpf12z = (uf1 + (vf1 - 1.0)) * (uf0 * pf00.z + (((vf0 * pf01.z - uf1 * pf10.z) - vf1 * pf11.z) - pf02.z * (uf0 + (vf0 - 1.0)))) / sqrt(pow2(uf0 * pf00.x + (((vf0 * pf01.x - uf1 * pf10.x) - vf1 * pf11.x) - pf02.x * (uf0 + (vf0 - 1.0)))) + pow2(uf0 * pf00.y + (((vf0 * pf01.y - uf1 * pf10.y) - vf1 * pf11.y) - pf02.y * (uf0 + (vf0 - 1.0)))));

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
        vec3 p, vec3 p0, vec3 p1, vec3 p2, vec4 c0, vec4 c1, vec4 c2, float u, float v,
        vec4 dOut_dc, float dOut_du, float dOut_dv, // forwarded from segmentLengthAdjoint
        inout vec3 dOut_dp0, inout vec3 dOut_dp1, inout vec3 dOut_dp2,
        out vec4 dOut_dc0, out vec4 dOut_dc1, out vec4 dOut_dc2) {
    float du_dp0x = (-(p1.y - p2.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p1.z - p2.z) * (-(p0.x - p2.x) * (p1.z - p2.z) + (p0.z - p2.z) * (p1.x - p2.x))) * sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp0y = ((p1.x - p2.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p1.z - p2.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp0z = (-(p1.x - p2.x) * (-(p0.x - p2.x) * (p1.z - p2.z) + (p0.z - p2.z) * (p1.x - p2.x)) + (p1.y - p2.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp1x = (-((p.y - p2.y) * ((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x)) + (p.z - p2.z) * ((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p0.y - p2.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p0.z - p2.z) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp1y = (((p.x - p2.x) * ((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x)) - (p.z - p2.z) * ((p.y - p1.y) * (p1.z - p2.z) - (p.z - p1.z) * (p1.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) - ((p0.x - p2.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p0.z - p2.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp1z = (((p.x - p2.x) * ((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)) + (p.y - p2.y) * ((p.y - p1.y) * (p1.z - p2.z) - (p.z - p1.z) * (p1.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) - ((p0.x - p2.x) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)) + (p0.y - p2.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp2x = (((p.y - p1.y) * ((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x)) + (p.z - p1.z) * ((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) - ((p0.y - p1.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p0.z - p1.z) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp2y = ((-(p.x - p1.x) * ((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x)) + (p.z - p1.z) * ((p.y - p1.y) * (p1.z - p2.z) - (p.z - p1.z) * (p1.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p0.x - p1.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p0.z - p1.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float du_dp2z = (-((p.x - p1.x) * ((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)) + (p.y - p1.y) * ((p.y - p1.y) * (p1.z - p2.z) - (p.z - p1.z) * (p1.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p0.x - p1.x) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)) + (p0.y - p1.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x))))) / sqrt(pow2(((p.x - p1.x) * (p1.y - p2.y) - (p.y - p1.y) * (p1.x - p2.x))) + pow2(((p.x - p1.x) * (p1.z - p2.z) - (p.z - p1.z) * (p1.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp0x = ((-(p.y - p2.y) * ((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x)) - (p.z - p2.z) * ((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) - ((p1.y - p2.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p1.z - p2.z) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp0y = (((p.x - p2.x) * ((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x)) - (p.z - p2.z) * ((p.y - p0.y) * (p0.z - p2.z) - (p.z - p0.z) * (p0.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p1.x - p2.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p1.z - p2.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp0z = (((p.x - p2.x) * ((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)) + (p.y - p2.y) * ((p.y - p0.y) * (p0.z - p2.z) - (p.z - p0.z) * (p0.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p1.x - p2.x) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)) + (p1.y - p2.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp1x = ((p0.y - p2.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p0.z - p2.z) * (-(p0.x - p2.x) * (p1.z - p2.z) + (p0.z - p2.z) * (p1.x - p2.x))) * sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp1y = (-(p0.x - p2.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p0.z - p2.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp1z = ((p0.x - p2.x) * (-(p0.x - p2.x) * (p1.z - p2.z) + (p0.z - p2.z) * (p1.x - p2.x)) - (p0.y - p2.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) / pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp2x = (((p.y - p0.y) * ((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x)) + (p.z - p0.z) * ((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) - ((p0.y - p1.y) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) + (p0.z - p1.z) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp2y = ((-(p.x - p0.x) * ((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x)) + (p.z - p0.z) * ((p.y - p0.y) * (p0.z - p2.z) - (p.z - p0.z) * (p0.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p0.x - p1.x) * ((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x)) - (p0.z - p1.z) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);
    float dv_dp2z = (-((p.x - p0.x) * ((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)) + (p.y - p0.y) * ((p.y - p0.y) * (p0.z - p2.z) - (p.z - p0.z) * (p0.y - p2.y))) * (pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)))) + ((p0.x - p1.x) * ((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x)) + (p0.y - p1.y) * ((p0.y - p2.y) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.y - p2.y))) * (pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x))))) / sqrt(pow2(((p.x - p0.x) * (p0.y - p2.y) - (p.y - p0.y) * (p0.x - p2.x))) + pow2(((p.x - p0.x) * (p0.z - p2.z) - (p.z - p0.z) * (p0.x - p2.x)))) * pow(pow2(((p0.x - p2.x) * (p1.y - p2.y) - (p0.y - p2.y) * (p1.x - p2.x))) + pow2(((p0.x - p2.x) * (p1.z - p2.z) - (p0.z - p2.z) * (p1.x - p2.x))), 3.0 / 2.0);

    // c = u * c0 + v * c1 + (1 - u - v) * c2;
    // p = u * p0 + v * p1 + (1 - u - v) * p2;
    dOut_du += dot(dOut_dc, c0 - c2 /* dc_du */);
    dOut_dv += dot(dOut_dc, c1 - c2 /* dc_dv */);
    dOut_dc0 = dOut_dc * u /* dc_c0 */;
    dOut_dc1 = dOut_dc * v /* dc_c1 */;
    dOut_dc2 = dOut_dc * (1.0 - u - v) /* dc_c2 */;
    dOut_dp0 = dOut_du * vec3(du_dp0x, du_dp0y, du_dp0z) + dOut_dv * vec3(dv_dp0x, dv_dp0y, dv_dp0z);
    dOut_dp1 = dOut_du * vec3(du_dp1x, du_dp1y, du_dp1z) + dOut_dv * vec3(dv_dp1x, dv_dp1y, dv_dp1z);
    dOut_dp2 = dOut_du * vec3(du_dp2x, du_dp2y, du_dp2z) + dOut_dv * vec3(dv_dp2x, dv_dp2y, dv_dp2z);
}

vec4 frontToBackPQ(uint fragsCount) {
    uint i;

    // Bring it to heap structure
    for (i = fragsCount/4; i > 0; --i) {
        // First is not one right place - will be done in for
        minHeapSink4(i, fragsCount); // Sink all inner nodes
    }

    vec4 fragment0Color, fragment1Color;
    float fragment0Depth, fragment1Depth;
    bool fragment0Boundary, fragment1Boundary;
    bool fragment0FrontFace, fragment1FrontFace;
    uint if00, if01, if02, if10, if11, if12;
    vec3 pf0, pf00, pf01, pf02, pf1, pf10, pf11, pf12;
    vec4 cf00, cf01, cf02, cf10, cf11, cf12;
    float uf0, vf0, uf1, vf1;
    getNextFragment(
            0, fragsCount, fragment0Color, fragment0Depth, fragment0Boundary, fragment0FrontFace,
            if00, if01, if02, pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0);

    ivec2 workIdx = ivec2(gl_FragCoord.xy);
    vec4 colorRayOut = imageLoad(colorImageOpt, workIdx);
    vec4 dOut_dColorRayOut = imageLoad(adjointColors, workIdx);

    // Start with transparent Ray
    vec4 colorAcc;
    float A;
    float t, tSeg;
    for (i = 1; i < fragsCount; i++) {
        // Load the new fragment.
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
        getNextFragment(
                i, fragsCount, fragment0Color, fragment0Depth, fragment0Boundary, fragment0FrontFace,
                if00, if01, if02, pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0);

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

            // Compute adjoint for the pre-accumulation colors and opacity.
            vec3 dOut_dc0;
            vec3 dOut_dc1;
            float dOut_da;
            accumulateLinearConstAdjoint(
                    tSeg, INV_N_SUB,  c0, c1, alpha * attenuationCoefficient, A, dOut_dColorAcc,
                    dOut_dt, dOut_dc0, dOut_dc1, dOut_da);
            dOut_da *= attenuationCoefficient;

            // Backpropagate gradients wrt. segmented color/opacity to intersection point color/opacity.
            // dc0_dcf0 = (1.0 - fbegin), dc1_dcf0 = (1.0 - fend), dc0_dcf1 = fbegin, dc1_dcf1 = fend.
            dOut_dcf0 += vec4((1.0 - fbegin) * dOut_dc0 + (1.0 - fend) * dOut_dc1, (1.0 - fmid) * dOut_da);
            dOut_dcf1 += vec4(fbegin * dOut_dc0 + fend * dOut_dc1, fmid * dOut_da);
        }

        vec3 dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12;
        vec4 dOut_dcf00, dOut_dcf01, dOut_dcf02, dOut_dcf10, dOut_dcf11, dOut_dcf12;
        float dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1;
        segmentLengthAdjoint(
                pf00, pf01, pf02, pf00, pf01, pf02, uf0, vf0, uf1, vf1, dOut_dt,
                dOut_duf0, dOut_dvf0, dOut_duf1, dOut_dvf1,
                dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dpf10, dOut_dpf11, dOut_dpf12);
        baryAdjoint(
                pf0, pf00, pf01, pf02, cf00, cf01, cf02, uf0, vf0,
                dOut_dcf0, dOut_duf0, dOut_dvf0,
                dOut_dpf00, dOut_dpf01, dOut_dpf02, dOut_dcf00, dOut_dcf01, dOut_dcf02);
        baryAdjoint(
                pf1, pf10, pf11, pf12, cf10, cf11, cf12, uf1, vf1,
                dOut_dcf1, dOut_duf1, dOut_dvf1,
                dOut_dpf10, dOut_dpf11, dOut_dpf12, dOut_dcf10, dOut_dcf11, dOut_dcf12);

        atomicAddGradCol(if00, dOut_dcf00);
        atomicAddGradCol(if01, dOut_dcf01);
        atomicAddGradCol(if02, dOut_dcf02);

        atomicAddGradCol(if10, dOut_dcf10);
        atomicAddGradCol(if11, dOut_dcf11);
        atomicAddGradCol(if12, dOut_dcf12);

        atomicAddGradPos(if00, dOut_dpf00);
        atomicAddGradPos(if01, dOut_dpf01);
        atomicAddGradPos(if02, dOut_dpf02);

        atomicAddGradPos(if10, dOut_dpf10);
        atomicAddGradPos(if11, dOut_dpf11);
        atomicAddGradPos(if12, dOut_dpf12);
    }

    return vec4(0.0);
}
