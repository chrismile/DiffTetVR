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

#include "TetQuality.hpp"

#define TET_QUALITY_METRIC_DEF(name) static float name (\
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3)

namespace TetQualityMetrics {

inline float computeVolume(
        const glm::vec3 & p0, const glm::vec3 & p1, const glm::vec3 & p2, const glm::vec3 & p3) {
    return -glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0);
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

TET_QUALITY_METRIC_DEF(volumeSign) {
    float volumeSign = glm::sign(glm::dot(glm::cross(p1 - p0, p2 - p0), p3 - p0));
    return volumeSign;
}

/*
 * Metrics from: https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf
 * See also: https://people.eecs.berkeley.edu/~jrs/papers/elem.pdf
 */

TET_QUALITY_METRIC_DEF(parthasarathy) {
    float edgeLengths[6];
    computeEdgeLengths(p0, p1, p2, p3, edgeLengths);
    const float V = computeVolume(p0, p1, p2, p3);
    float l_rms = 0.0f;
    for (auto edgeLength : edgeLengths) {
        l_rms += edgeLength * edgeLength;
    }
    l_rms = std::sqrt(l_rms / 6.0f);
    return 6.0f * std::sqrt(2.0f) * V / (l_rms * l_rms * l_rms);
}

}

static TetQualityMetricFunc* getTetQualityMetricFunc(TetQualityMetric metric) {
    TetQualityMetricFunc* functor = nullptr;
    switch(metric) {
        case TetQualityMetric::VOLUME_SIGN:   functor = &TetQualityMetrics::volumeSign;    break;
        case TetQualityMetric::PARTHASARATHY: functor = &TetQualityMetrics::parthasarathy; break;
        default: sgl::Logfile::get()->throwError("Error in getTetQualityMetricFunc: Invalid metric.");
    }
    return functor;
}

#endif //DIFFTETVR_TETQUALITYFUNCTIONS_HPP
