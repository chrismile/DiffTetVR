layout(binding = 15) uniform ClipPlaneDataBuffer {
    vec3 clipPlaneNormal;
    float clipPlaneDistance;
};

bool checkIsPointOutsideClipPlane(vec3 p) {
    float d = dot(clipPlaneNormal, p) - clipPlaneDistance;
    return d > 0.0;
}
