/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

#ifndef DIFFTETVR_OPTIMIZERDEFINES_HPP
#define DIFFTETVR_OPTIMIZERDEFINES_HPP

#include <cstdint>

enum class OptimizerType {
    SGD, ADAM
};
const char* const OPTIMIZER_TYPE_NAMES[] = {
        "SGD", "Adam"
};

enum class LossType {
    L1, L2
};
const char* const LOSS_TYPE_NAMES[] = {
        "L1", "L2"
};

struct OptimizerSettings {
    // SGD & Adam.
    float learningRate = 0.4f;
    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

struct TetRegularizerSettings {
    // Regularizer loss weight (0 means turned off).
    float lambda = 0.0f;
    // Softplus parameter.
    float beta = 10.0f;
};

struct OptimizationSettings {
    OptimizerType optimizerType = OptimizerType::ADAM;
    LossType lossType = LossType::L2;
    bool optimizePositions = true;
    bool optimizeColors = true;
    OptimizerSettings optimizerSettingsPositions{};
    OptimizerSettings optimizerSettingsColors{};
    TetRegularizerSettings tetRegularizerSettings{};
    int maxNumEpochs = 200;
    bool fixBoundary = false;
    // DVR.
    uint32_t imageWidth = 512;
    uint32_t imageHeight = 512;
    float attenuationCoefficient = 100.0f;
    bool sampleRandomView = true;
    // Selected file name.
    std::string dataSetFileNameGT, dataSetFileNameOpt;
    // Export position gradient field.
    bool exportPositionGradients = false;
    std::string exportFileNameGradientField;
    bool isBinaryVtk = true;
};

#endif //DIFFTETVR_OPTIMIZERDEFINES_HPP
