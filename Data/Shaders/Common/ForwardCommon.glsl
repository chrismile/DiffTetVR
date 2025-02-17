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

/*vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    vec3 p0 = a * c0;
    vec3 p1 = a * (c1 - c0);
    float A = exp(-a * t);
    float B = -1.0 / a;
    return vec4(B * ((A - 1.0) * p0 + ((t - B) * A + B) * p1), 1.0 - A);
}*/

vec4 accumulateConst(float t, vec3 c, float a) {
    float A = 1.0 - exp(-a * t);
    return vec4(A * c, A);
}

vec4 accumulateLinearConst(float t, vec3 c0, vec3 c1, float a) {
    if (a < 1e-6) {
        // lim a->0 (t + 1.0 / a) * A - 1.0 / a) = 0
        return vec4(0.0);
    }
    float A = exp(-a * t);
    return vec4((1.0 - A) * c0 + ((t + 1.0 / a) * A - 1.0 / a) * (c0 - c1), 1.0 - A);
}

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
    vec3 B = (p1 - 2.0 * b * p2) / (2.0 * a) * (exp(a * tp * tp) - exp(a * b * b));
    vec3 C = p2 / (2.0 * a) * (tp * exp(a * tp * tp) - (PI_SQRT * G) / (2.0 * sqrta) - b * exp(a * b * b) + (PI_SQRT * H) / (2.0 * sqrta));
    return vec4(exp(c) * (A + B + C), 1.0 - exp(-a0 * t + a * t * t));
}
