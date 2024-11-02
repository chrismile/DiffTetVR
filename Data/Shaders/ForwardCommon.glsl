/*vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    vec3 p0 = a * c0;
    vec3 p1 = a * (c1 - c0);
    float A = exp(-a * t);
    float B = -1.0 / a;
    return vec4(B * ((A - 1.0) * p0 + ((t - B) * A + B) * p1), 1.0 - A);
}*/

vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    if (a < 1e-6) {
        // lim a->0 (t + 1.0 / a) * A - 1.0 / a) = 0
        return vec4(0.0);
    }
    float A = exp(-a * t);
    return vec4((1.0 - A) * c0 + ((t + 1.0 / a) * A - 1.0 / a) * (c0 - c1), 1.0 - A);
}

vec3 barycentricInterpolation(vec3 p0, vec3 p1, vec3 p2, vec3 fragmentPositionWorld) {
    // Barycentric interpolation (face 0).
    vec3 d20 = p2 - p0;
    vec3 d21 = p2 - p1;
    float totalArea = max(length(cross(d20, d21)), 1e-5);
    float u = length(cross(d21, fragmentPositionWorld - p1)) / totalArea;
    float v = length(cross(fragmentPositionWorld - p0, d20)) / totalArea;
    const vec3 barycentricCoordinates = vec3(u, v, 1.0 - u - v);
    return barycentricCoordinates;
}
