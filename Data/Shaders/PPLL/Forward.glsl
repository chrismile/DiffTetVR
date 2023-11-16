float erfi(float z) {
    float z2 = z * z;
    float z3 = z * z2;
    float z5 = z3 * z2;
    float z7 = z5 * z2;
    float z9 = z7 * z2;
    return 2.0 * INV_PI_SQRT * (z + z3 / 3.0 + z5 / 10.0 + z7 / 42.0 + z9 / 216.0);
}

float erf(float z) {
    // Maclaurin series
    //float z2 = z * z;
    //float z3 = z * z2;
    //float z5 = z3 * z2;
    //float z7 = z5 * z2;
    //float z9 = z7 * z2;
    //return 2.0 * INV_PI_SQRT * (z - z3 / 3.0 + z5 / 10.0 - z7 / 42.0 + z9 / 216.0);
    // Buermann series
    float A = exp(-z * z);
    float B = sqrt(1.0 - A);
    return 2.0 * INV_PI_SQRT * sign(z) * B * (0.5 * PI_SQRT + 31.0 / 200.0 * A - 341.0 / 8000.0 * A * A);
}

/*vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    vec3 p0 = a * c0;
    vec3 p1 = a * (c1 - c0);
    float A = exp(-a * t);
    float B = -1.0 / a;
    return vec4(B * ((A - 1.0) * p0 + ((t - B) * A + B) * p1), 1.0 - A);
}*/

vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    float A = exp(-a * t);
    return vec4((1.0 - A) * c0 + ((t + 1.0 / a) * A - 1.0 / a) * (c0 - c1), 1.0 - A);
}

vec4 accumulateLinear(float t, vec3 c0, vec3 c1, float a0, float a1) {
    float aDiff = a0 - a1;
    if (abs(aDiff) < 1e-4) {
        return accumulateLinearConst(t, c0, c1, a0);
    }

    float a = 0.5 * aDiff;
    float b = -a0 / aDiff;
    float c = -a0 * a0 / (aDiff * aDiff);
    float sqrta = sqrt(abs(a));
    float tp = t + b;

    float G;
    float H;
    if (a > 0.0) {
        G = erfi(sqrta * tp);
        H = erfi(sqrta * b);
    } else if (a < 0.0) {
        G = erf(sqrta * tp);
        H = erf(sqrta * b);
    }

    vec3 p0 = a0 * c0;
    vec3 p1 = -2.0 * a0 * c0 + a1 * c0 + a0 * c1;
    vec3 p2 = a0 * c0 - a1 * c0 - a0 * c1 + a1 * c1;

    vec3 A = (p0 - b * p1 + b * b * p2) * PI_SQRT / (2.0 * sqrta) * (G - H);
    vec3 B = (p1 - 2 * b * p2) / (2.0 * a) * (exp(a * tp * tp) - exp(a * b * b));
    vec3 C = p2 / (2.0 * a) * (tp * exp(a * tp * tp) - (PI_SQRT * G) / (2.0 * sqrta) - b * exp(a * b * b) + (PI_SQRT * H) / (2.0 * sqrta));
    return vec4(exp(c) * (A + B + C), 1.0 - exp(-a0 * t + 0.5 * (a0 - a1) * t * t));
}

void getNextFragment(in uint i, in uint fragsCount, out vec4 color, out float depthLinear, out bool boundary, out bool frontFace) {
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
    vec3 fragmentPositionWorld = fragPosWorld.xyz / fragPosWorld.w;

    uint i0 = triangleIndices[faceIndex];
    uint i1 = triangleIndices[faceIndex + 1];
    uint i2 = triangleIndices[faceIndex + 2];
    vec3 p0 = vertexPositions[i0];
    vec3 p1 = vertexPositions[i1];
    vec3 p2 = vertexPositions[i2];
    vec4 c0 = vertexColors[i0];
    vec4 c1 = vertexColors[i1];
    vec4 c2 = vertexColors[i2];

    // Barycentric interpolation.
    vec3 d20 = p2 - p0;
    vec3 d21 = p2 - p1;
    float totalArea = max(length(cross(d20, d21)), 1e-5);
    float u = length(cross(d21, fragmentPositionWorld - p1)) / totalArea;
    float v = length(cross(fragmentPositionWorld - p0, d20)) / totalArea;
    const vec3 barycentricCoordinates = vec3(u, v, 1.0 - u - v);
    color = c0 * barycentricCoordinates.x + c1 * barycentricCoordinates.y + c2 * barycentricCoordinates.z;
}

vec4 frontToBackPQ(uint fragsCount) {
    uint i;

    // Bring it to heap structure
    for (i = fragsCount/4; i > 0; --i) {
        // First is not one right place - will be done in for
        minHeapSink4(i, fragsCount); // Sink all inner nodes
    }

    vec4 fragment1Color, fragment2Color;
    float fragment1Depth, fragment2Depth;
    bool fragment1Boundary, fragment2Boundary;
    bool fragment1FrontFace, fragment2FrontFace;
    float t, tSeg;
    getNextFragment(0, fragsCount, fragment2Color, fragment2Depth, fragment2Boundary, fragment2FrontFace);

    // Start with transparent Ray
    vec4 rayColor = vec4(0.0);
    vec4 currentColor;
    for (i = 1; i < fragsCount; i++) {
        // Load the new fragment.
        fragment1Color = fragment2Color;
        fragment1Depth = fragment2Depth;
        fragment1Boundary = fragment2Boundary;
        fragment1FrontFace = fragment2FrontFace;
        getNextFragment(i, fragsCount, fragment2Color, fragment2Depth, fragment2Boundary, fragment2FrontFace);

        // Skip if the closest fragment is a boundary face.
        if ((fragment1Boundary && !fragment1FrontFace) && (fragment2Boundary && fragment2FrontFace)) {
            continue;
        }

        // Compute the accumulated color of the fragments.
        t = fragment2Depth - fragment1Depth;

#ifdef USE_SUBDIVS
        tSeg = t / float(NUM_SUBDIVS);
        const float INV_N_SUB = 1.0 / float(NUM_SUBDIVS);
        for (int s = 0; s < NUM_SUBDIVS; s++) {
            float fbegin = (float(s)) * INV_N_SUB;
            float fmid = (float(s) + 0.5) * INV_N_SUB;
            float fend = (float(s) + 1.0) * INV_N_SUB;
            vec3 c0 = mix(fragment1Color.rgb, fragment2Color.rgb, fbegin);
            vec3 c1 = mix(fragment1Color.rgb, fragment2Color.rgb, fend);
            float alpha = mix(fragment1Color.a, fragment2Color.a, fmid);
            currentColor = accumulateLinearConst(tSeg, c0, c1, alpha * attenuationCoefficient);
            rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
            rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
        }
#else
        currentColor = accumulateLinear(
                t, fragment1Color.rgb, fragment2Color.rgb,
                fragment1Color.a * attenuationCoefficient, fragment2Color.a * attenuationCoefficient);
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;
#endif
    }

    //rayColor.rgb = rayColor.rgb / rayColor.a; // Correct rgb with alpha
    return rayColor;
}
