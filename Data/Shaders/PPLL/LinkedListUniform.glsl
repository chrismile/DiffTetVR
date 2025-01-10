layout(binding = 0) uniform UniformDataBuffer {
    // Inverse of (projectionMatrix * viewMatrix).
    mat4 inverseViewProjectionMatrix;
    mat4 viewProjectionMatrix;

    // Number of fragments we can store in total.
    uint linkedListSize;
    // Size of the viewport in x direction (in pixels).
    int viewportW;
    // Camera near/far plane distance.
    float zNear, zFar;

    // Camera front vector.
    vec3 cameraFront;
    // Volume attenuation.
    float attenuationCoefficient;

    vec3 cameraPosition;
    float cameraPositionPadding;

    // Viewport size in x/y direction.
    uvec2 viewportSize;

    // Size of the viewport in x direction (in pixels) without padding.
    int viewportLinearW;

    // Early ray termination alpha threshold.
    float earlyRayTerminationAlpha;
};
