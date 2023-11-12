/**
 * BSD 2-Clause License
 *
 * Copyright (c) 2020, Maximilian Bandle, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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

//#ifdef DEPTH_TYPE_UINT
//#define DEPTH_TYPE uint
//#else
//#define DEPTH_TYPE float
//#endif
#define DEPTH_TYPE uint

// Swap two Frags in color and depth Array => Avoid bacause expensive
void swapFragments(uint i, uint j) {
    uint cTemp = colorList[i];
    colorList[i] = colorList[j];
    colorList[j] = cTemp;
    DEPTH_TYPE dTemp = depthList[i];
    depthList[i] = depthList[j];
    depthList[j] = dTemp;
}

vec4 blendFTB(uint fragsCount) {
    vec4 color = vec4(0.0);
    for (uint i = 0; i < fragsCount; i++) {
        // Front-to-Back (FTB) blending
        // Blend the accumulated color with the color of the fragment node
        vec4 colorSrc = unpackUnorm4x8(colorList[i]);
        #ifdef USE_COVERAGE_MASK
        colorSrc.a *= unpackFloat8(depthList[i]);
        #endif
        color.rgb = color.rgb + (1.0 - color.a) * colorSrc.a * colorSrc.rgb;
        color.a = color.a + (1.0 - color.a) * colorSrc.a;
    }
    return vec4(color.rgb / color.a, color.a);
}


vec4 bubbleSort(uint fragsCount) {
    bool changed; // Has anything changed yet
    do {
        changed = false; // Nothing changed yet
        for (uint i = 0; i < fragsCount - 1; ++i) {
            // Go through all
            if(depthList[i] > depthList[i+1]) {
                // Order not correct => Swap
                swapFragments(i, i+1);
                changed = true; // Something has changed
            }
        }
    } while (changed); // Nothing changed => sorted

    return blendFTB(fragsCount);
}


vec4 insertionSort(uint fragsCount) {
    // Temporary fragment storage
    uint fragColor;
    DEPTH_TYPE fragDepth;

    uint i, j;
    for (i = 1; i < fragsCount; ++i) {
        // Get the fragment
        fragColor = colorList[i];
        fragDepth = depthList[i];

        j = i; // Store its position
        while (j >= 1 && depthList[j-1] > fragDepth) {
            // Shift the fragments through the list until place is found
            colorList[j] = colorList[j-1];
            depthList[j] = depthList[j-1];
            --j;
        }

        // Insert it at the right place
        colorList[j] = fragColor;
        depthList[j] = fragDepth;
    }

    return blendFTB(fragsCount);
}


vec4 shellSort(uint fragsCount) {
    // Temporary fragment storage
    uint fragColor;
    DEPTH_TYPE fragDepth;

    // Optimal gap sequence for 128 elements from [Ciu01, table 1]
    uint i, j, gap;
    uvec4 gaps = uvec4(24, 9, 4, 1);
    for(uint g = 0; g < 4; g++) {
        // For every gap
        gap = gaps[g]; // Current Cap
        for(i = gap; i < fragsCount; ++i) {
            // Get the fragment
            fragColor = colorList[i];
            fragDepth = depthList[i];
            j = i;

            // Shift earlier until correct
            while (j >= gap && depthList[j-gap] > fragDepth) {
                // Shift the fragments through the list until place is found
                colorList[j] = colorList[j-gap];
                depthList[j] = depthList[j-gap];
                j-=gap;
            }

            // Insert it at the right place
            colorList[j] = fragColor;
            depthList[j] = fragDepth;
        }
    }

    return blendFTB(fragsCount);
}


void maxHeapSink(uint x, uint fragsCount) {
    uint c; // Child
    while((c = 2 * x + 1) < fragsCount) {
        // While children exist
        if(c + 1 < fragsCount && depthList[c] < depthList[c+1]) {
            // Find the biggest of both
            ++c;
        }

        if(depthList[x] >= depthList[c]) {
            // Does it have to sink
            return;
        } else {
            swapFragments(x, c);
            x = c; // Swap and sink again
        }
    }
}

vec4 heapSort(uint fragsCount) {
    uint i;
    for (i = (fragsCount + 1)/2 ; i > 0 ; --i) {
        // Bring it to heap structure
        maxHeapSink(i-1, fragsCount); // Sink all inner nodes
    }
    // Heap => Sorted List
    for (i=1;i<fragsCount;++i) {
        swapFragments(0, fragsCount-i); // Swap max to List End
        maxHeapSink(0, fragsCount-i); // Sink the max to obtain correct heap
    }

    return blendFTB(fragsCount);
}


void minHeapSink4(uint x, uint fragsCount) {
    uint c, t; // Child, Tmp
    while ((t = 4 * x + 1) < fragsCount) {
        if (t + 1 < fragsCount && depthList[t] > depthList[t+1]) {
            // 1st vs 2nd
            c = t + 1;
        } else {
            c = t;
        }

        if (t + 2 < fragsCount && depthList[c] > depthList[t+2]) {
            // Smallest vs 3rd
            c = t + 2;
        }

        if (t + 3 < fragsCount && depthList[c] > depthList[t+3]) {
            // Smallest vs 3rd
            c = t + 3;
        }

        if (depthList[x] <= depthList[c]) {
            return;
        } else {
            swapFragments(x, c);
            x = c;
        }
    }
}


void getNextFragment(in uint i, in uint fragsCount, out vec4 color, out float depth, out bool boundary, out bool frontFace) {
    minHeapSink4(0, fragsCount - i);
    color = unpackUnorm4x8(colorList[0]);
    uint depthValuePacked = depthList[0];
    depth = convertDepthBufferValueToLinearDepth(unpackDepth(depthValuePacked));
    boundary = (depthValuePacked & 1u) == 1u ? true : false;
    frontFace = ((depthValuePacked >> 1u) & 1u) == 1u ? true : false;
    colorList[0] = colorList[fragsCount - i - 1];
    depthList[0] = depthList[fragsCount - i - 1];
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

#define USE_SUBDIVS
const uint NUM_SUBDIVS = 4;

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
    bool isInMesh = false;
    for (i = 1; i < fragsCount && rayColor.a < 0.99; i++) {
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
        t = fragment2Depth - fragment1Depth;
        tSeg = t / float(NUM_SUBDIVS);
        for (uint s = 0; s < NUM_SUBDIVS; s++) {
            float fbegin = (float(i)) * tSeg / t;
            float fmid = (float(i) + 0.5) * tSeg / t;
            float fend = (float(i) + 1.0) * tSeg / t;
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
        //currentColor = accumulateLinearConst(
        //        t, fragment1Color.rgb, fragment2Color.rgb, fragment1Color.a * attenuationCoefficient);
        /*float volumeOpacityFactor = clamp(1.0 - exp(-fragment1Color.a * attenuationCoefficient * t), 0.0, 1.0);
        currentColor = vec4(fragment1Color.rgb, volumeOpacityFactor);*/

        // FTB Blending.
        //rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.a * currentColor.rgb;
        //rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * currentColor.rgb;
        //rayColor.a = rayColor.a + (1.0 - rayColor.a) * currentColor.a;

        /*if (fragment1Color.a - fragment2Color.a > 0.0) {
            rayColor = vec4(1.0);
        }*/

        // Max Steps = #frags Stop when color is saturated enough
        /*minHeapSink4(0, fragsCount - i++); // Sink it right + increment i
        vec4 colorSrc = unpackUnorm4x8(colorList[0]); // Heap first is min
#ifdef USE_COVERAGE_MASK
        colorSrc.a *= unpackFloat8(depthList[0]);
#endif

        // FTB Blending
        rayColor.rgb = rayColor.rgb + (1.0 - rayColor.a) * colorSrc.a * colorSrc.rgb;
        rayColor.a = rayColor.a + (1.0 - rayColor.a) * colorSrc.a;

        // Move Fragments up for next run
        colorList[0] = colorList[fragsCount-i];
        depthList[0] = depthList[fragsCount-i];*/
    }

    /*if (fragment1FrontFace) {
        rayColor = vec4(1.0);
    }*/

    rayColor.rgb = rayColor.rgb / rayColor.a; // Correct rgb with alpha
    return rayColor;
}


vec4 bitonicSort(uint fragsCount) {
    DEPTH_TYPE fragDepth_i, fragDepth_l;

    // Cf. https://en.wikipedia.org/wiki/Bitonic_sorter
    uint i, j, k, l;
    for (k = 2; k <= fragsCount; k *= 2) {
        for (j = k / 2; j > 0; j /= 2) {
            for (i = 0; i < fragsCount; i++) {
                l = i ^ j;
                if (l > i && l < fragsCount) {
                    fragDepth_i = depthList[i];
                    fragDepth_l = depthList[l];
                    if (((i & k) == 0 && fragDepth_i > fragDepth_l)
                            || ((i & k) != 0 && fragDepth_i < fragDepth_l)) {
                        swapFragments(i, l);
                    }
                }
            }
        }
    }

    return blendFTB(fragsCount);
}
