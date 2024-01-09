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

#ifndef DIFFTETVR_TETQUALITYFUNCTIONS_HPP
#define DIFFTETVR_TETQUALITYFUNCTIONS_HPP

#include <glm/gtx/norm.hpp>

#include "TetQuality.hpp"

#define TET_QUALITY_METRIC_DEF(name) static float name (\
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)

namespace TetQualityMetrics {

#define EDGE_IDX(i, j) ((i) + (j) - (((i) == 0) ? 1 : 0))

inline float square(float x) { return x * x; }

inline float computeVolume(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    return -glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0) / 6.0f;
}

inline void computeEdgeLengths(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3, float edgeLengths[]) {
    edgeLengths[0] = glm::length(p0 - p1);
    edgeLengths[1] = glm::length(p0 - p2);
    edgeLengths[2] = glm::length(p0 - p3);
    edgeLengths[3] = glm::length(p1 - p2);
    edgeLengths[4] = glm::length(p1 - p3);
    edgeLengths[5] = glm::length(p2 - p3);
}

inline float computeEdgeLengthRms(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    float l[6];
    computeEdgeLengths(p0, p1, p2, p3, l);
    float l_rms = 0.0f;
    for (auto edgeLength : l) {
        l_rms += edgeLength * edgeLength;
    }
    l_rms = std::sqrt(l_rms / 6.0f);
    return l_rms;
}

inline float computeEdgeLengthMax(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    float l[6];
    computeEdgeLengths(p0, p1, p2, p3, l);
    float l_max = 0.0f;
    for (auto edgeLength : l) {
        l_max = std::max(l_max, edgeLength);
    }
    return l_max;
}

inline void computeFaceAreas(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3, float faceAreas[]) {
    faceAreas[0] = 0.5f * glm::length(glm::cross(p3 - p2, p3 - p1));
    faceAreas[1] = 0.5f * glm::length(glm::cross(p3 - p2, p3 - p0));
    faceAreas[2] = 0.5f * glm::length(glm::cross(p3 - p1, p3 - p0));
    faceAreas[3] = 0.5f * glm::length(glm::cross(p0 - p2, p0 - p1));
}

inline float computeFaceAreaRms(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float A_rms = 0.0f;
    for (auto faceArea : A) {
        A_rms += faceArea * faceArea;
    }
    A_rms = std::sqrt(A_rms / 4.0f);
    return A_rms;
}

inline float computeFaceAreaSum(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float A_sum = 0.0f;
    for (auto faceArea : A) {
        A_sum += faceArea;
    }
    return A_sum;
}

inline float computeFaceAreaMax(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float A_max = 0.0f;
    for (auto faceArea : A) {
        A_max = std::max(A_max, faceArea);
    }
    return A_max;
}

/*
 * For more details see sec. A.3 from: https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
 */
/*inline float computeMinContainmentRadius(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    // Compute the circumcenter.
    glm::vec3 O_circ =
            (
                    glm::length2(p0 - p3) * glm::cross(p1 - p3, p2 - p3)
                    + glm::length2(p1 - p3) * glm::cross(p2 - p3, p0 - p3)
                    + glm::length2(p2 - p3) * glm::cross(p0 - p3, p1 - p3)
            ) / (12.0f * computeVolume(p0, p1, p2, p3)) + p3;

    // TODO
    return -glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0);
}*/

TET_QUALITY_METRIC_DEF(volumeSign) {
    float volumeSign = -glm::sign(glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0));
    return volumeSign;
}

/*
 * Metrics from: https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
 * See also: https://people.eecs.berkeley.edu/~jrs/papers/elem.pdf
 */

/*TET_QUALITY_METRIC_DEF(qualityInterpolatedPiecewiseLinear) {
    const float V = computeVolume(p0, p1, p2, p3);
    float r_mc = computeMinContainmentRadius(p0, p1, p2, p3);
    return 9.0f * std::sqrt(3.0f) / 8.0f * V / (r_mc * r_mc * r_mc);
}*/

TET_QUALITY_METRIC_DEF(qualityGradientInterpolatedPiecewiseLinear) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l[6];
    computeEdgeLengths(p0, p1, p2, p3, l);
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float sumFaceAreas = 0.0f;
    float sumDenom0 = 0.0f;
    float sumDenom1 = 0.0f;
    for (int i = 0; i < 4; i++) {
        sumFaceAreas += A[i];
        float sumDenom1New = 0.0f;
        for (int j = i + 1; j < 4; j++) {
            sumDenom0 += A[i] * A[j] * square(l[EDGE_IDX(i, j)]);
            sumDenom1New += A[j] * l[EDGE_IDX(i, j)];
        }
        sumDenom1 = std::max(sumDenom1, sumDenom1New);
    }
    return
            3.0f * std::pow(486.0f * std::sqrt(3.0f) + 594.0f * std::sqrt(2.0f), 0.25f) * V * std::pow(sumFaceAreas, 0.75f)
            / (2.0f * std::pow(sumDenom0 + 6.0f * std::abs(V) * sumDenom1, 0.75f));
}

TET_QUALITY_METRIC_DEF(qualityGradientInterpolatedPiecewiseLinearAlt) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l[6];
    computeEdgeLengths(p0, p1, p2, p3, l);
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float sumFaceAreas = 0.0f;
    float sumDenom = 0.0f;
    for (int i = 0; i < 4; i++) {
        sumFaceAreas += A[i];
        for (int j = i + 1; j < 4; j++) {
            sumDenom += A[i] * A[j] * square(l[EDGE_IDX(i, j)]);
        }
    }
    return
            std::pow(3.0f, 17.0f / 8.0f) / std::pow(2.0f, 3.0f / 4.0f) * V
            * std::pow(sumFaceAreas / sumDenom, 0.75f);
}

// qualityStiffnessPoisson unimplemented so far

TET_QUALITY_METRIC_DEF(qualityStiffnessPoissonAlt) {
    const float V = computeVolume(p0, p1, p2, p3);
    float A_rms = computeFaceAreaRms(p0, p1, p2, p3);
    return std::pow(3.0f, 7.0f / 4.0f) / (2.0f * std::sqrt(2.0f)) * V / std::pow(A_rms, 1.5f);
}

TET_QUALITY_METRIC_DEF(qualityKnupp) {
    const float V = computeVolume(p0, p1, p2, p3);
    float A_rms = computeFaceAreaRms(p0, p1, p2, p3);
    float l_rms = computeEdgeLengthRms(p0, p1, p2, p3);
    return 1.5f * std::sqrt(6.0f) * V / (l_rms * A_rms);
}

TET_QUALITY_METRIC_DEF(qualityParthasarathy) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l_rms = computeEdgeLengthRms(p0, p1, p2, p3);
    return 6.0f * std::sqrt(2.0f) * V / (l_rms * l_rms * l_rms);
}

TET_QUALITY_METRIC_DEF(qualityNonsmooth) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l_max = computeEdgeLengthMax(p0, p1, p2, p3);
    return 6.0f * std::sqrt(2.0f) * V / (l_max * l_max * l_max);
}

TET_QUALITY_METRIC_DEF(qualityNonsmoothAlt) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l_max = computeEdgeLengthMax(p0, p1, p2, p3);
    float A_max = computeFaceAreaMax(p0, p1, p2, p3);
    return 1.5f * std::sqrt(6.0f) * V / (l_max * A_max);
}

TET_QUALITY_METRIC_DEF(qualityBaker) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l_max = computeEdgeLengthMax(p0, p1, p2, p3);
    float A_sum = computeFaceAreaSum(p0, p1, p2, p3);
    return 6.0f * std::sqrt(6.0f) * V / (l_max * A_sum);
}

TET_QUALITY_METRIC_DEF(qualityFreitag) {
    const float V = computeVolume(p0, p1, p2, p3);
    float l[6];
    computeEdgeLengths(p0, p1, p2, p3, l);
    float A[4];
    computeFaceAreas(p0, p1, p2, p3, A);
    float minVal = std::numeric_limits<float>::max();
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            float l_val = l[EDGE_IDX(i, j)];
            for (int k = 0; k < 4; k++) {
                if (k == i || k == j) {
                    continue;
                }
                for (int h = 0; h < 4; h++) {
                    if (h == i || h == j || h == k) {
                        continue;
                    }
                    float newVal = l_val / (A[k] * A[h]);
                    minVal = std::min(minVal, newVal);
                }
            }
        }
    }
    return 9.0f * std::sqrt(2.0f) / 8.0f * V * minVal;
}

}

static TetQualityMetricFunc* getTetQualityMetricFunc(TetQualityMetric metric) {
    TetQualityMetricFunc* functor = nullptr;
    switch(metric) {
        case TetQualityMetric::VOLUME_SIGN: functor = &TetQualityMetrics::volumeSign; break;
        case TetQualityMetric::GRAD_INTERPOLATED: functor = &TetQualityMetrics::qualityGradientInterpolatedPiecewiseLinear; break;
        case TetQualityMetric::GRAD_INTERPOLATED_ALT: functor = &TetQualityMetrics::qualityGradientInterpolatedPiecewiseLinearAlt; break;
        case TetQualityMetric::STIFFNESS_POISSON_ALT: functor = &TetQualityMetrics::qualityStiffnessPoissonAlt; break;
        case TetQualityMetric::KNUPP: functor = &TetQualityMetrics::qualityKnupp; break;
        case TetQualityMetric::PARTHASARATHY: functor = &TetQualityMetrics::qualityParthasarathy; break;
        case TetQualityMetric::NONSMOOTH: functor = &TetQualityMetrics::qualityNonsmooth; break;
        case TetQualityMetric::NONSMOOTH_ALT: functor = &TetQualityMetrics::qualityNonsmoothAlt; break;
        case TetQualityMetric::BAKER: functor = &TetQualityMetrics::qualityBaker; break;
        case TetQualityMetric::FREITAG: functor = &TetQualityMetrics::qualityFreitag; break;
        default: sgl::Logfile::get()->throwError("Error in getTetQualityMetricFunc: Invalid metric.");
    }
    return functor;
}

#endif //DIFFTETVR_TETQUALITYFUNCTIONS_HPP
