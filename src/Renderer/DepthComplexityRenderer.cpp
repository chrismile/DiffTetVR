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

#include <Utils/AppSettings.hpp>
#include <Utils/Timer.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "TetMeshVolumeRenderer.hpp"

void TetMeshVolumeRenderer::createDepthComplexityBuffers() {
    depthComplexityCounterBuffer = {};
    stagingBuffers.clear();
    if (!showDepthComplexity) {
        return;
    }

    int width = windowWidth;
    int height = windowHeight;

    size_t depthComplexityCounterBufferSizeBytes = sizeof(uint32_t) * width * height;
    depthComplexityCounterBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), depthComplexityCounterBufferSizeBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    depthComplexityCounterBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            depthComplexityCounterBuffer);

    auto* swapchain = sgl::AppSettings::get()->getSwapchain();
    stagingBuffers.reserve(swapchain->getNumImages());
    for (size_t i = 0; i < swapchain->getNumImages(); i++) {
        stagingBuffers.push_back(std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), depthComplexityCounterBufferSizeBytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU));
    }
}

bool TetMeshVolumeRenderer::needsReRender() {
    bool tmp = reRender;
    reRender = false;

    if (showDepthComplexity && !statisticsUpToDate) {
        // Update & print statistics if enough time has passed
        counterPrintFrags += sgl::Timer->getElapsedSeconds();
        if (tetMesh && (counterPrintFrags > 1.0f || firstFrame)) {
            computeStatistics(true);
            counterPrintFrags = 0.0f;
            firstFrame = false;
            statisticsUpToDate = true;
        }
    }

    return tmp;
}

void TetMeshVolumeRenderer::onHasMoved() {
    statisticsUpToDate = false;
    counterPrintFrags = 0.0f;
}

void TetMeshVolumeRenderer::computeStatistics(bool isReRender) {
    int width = windowWidth;
    int height = windowHeight;
    bufferSize = width * height;

    auto* swapchain = sgl::AppSettings::get()->getSwapchain();
    auto& stagingBuffer = stagingBuffers.at(swapchain->getImageIndex());
    depthComplexityCounterBuffer->copyDataTo(stagingBuffer, renderer->getVkCommandBuffer());
    renderer->syncWithCpu();

    auto *data = (uint32_t*)stagingBuffer->mapMemory();

    // Local reduction variables necessary for older OpenMP implementations
    uint64_t minComplexity = 0;
#ifdef USE_TBB
    using T = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;
    std::tie(totalNumFragments, usedLocations, maxComplexity, minComplexity) = tbb::parallel_reduce(
            tbb::blocked_range<uint64_t>(0, bufferSize), T{},
            [&data](tbb::blocked_range<uint64_t> const& r, T init) {
                uint64_t& totalNumFragments = std::get<0>(init);
                uint64_t& usedLocations = std::get<1>(init);
                uint64_t& maxComplexity = std::get<2>(init);
                uint64_t& minComplexity = std::get<3>(init);
                for (auto i = r.begin(); i != r.end(); i++) {
#else
    uint64_t totalNumFragments = 0;
    uint64_t usedLocations = 0;
    uint64_t maxComplexity = 0;
#if _OPENMP >= 201107
#pragma omp parallel for reduction(+:totalNumFragments,usedLocations) reduction(max:maxComplexity) \
    reduction(min:minComplexity) schedule(static) default(none) shared(data)
#endif
    for (uint64_t i = 0; i < bufferSize; i++) {
#endif
        totalNumFragments += data[i];
        if (data[i] > 0) {
            usedLocations++;
        }
        maxComplexity = std::max(maxComplexity, uint64_t(data[i]));
        minComplexity = std::min(minComplexity, uint64_t(data[i]));
    }
#ifdef USE_TBB
    return init;
            }, [&](T lhs, T rhs) -> T {
                return {
                        std::get<0>(lhs) + std::get<0>(lhs),
                        std::get<1>(lhs) + std::get<1>(lhs),
                        std::max(std::get<2>(lhs), std::get<2>(lhs)),
                        std::min(std::get<3>(lhs), std::get<3>(lhs))
                };
            });
#endif
    // Avoid dividing by zero in code below
    if (totalNumFragments == 0) {
        usedLocations = 1;
    }
#ifndef USE_TBB
    this->totalNumFragments = totalNumFragments;
    this->usedLocations = usedLocations;
    this->maxComplexity = maxComplexity;
#endif

    stagingBuffer->unmapMemory();

    firstFrame = false;

    /*bool performanceMeasureMode = (*sceneData->performanceMeasurer) != nullptr;
    if ((performanceMeasureMode || (*sceneData->recordingMode)) || firstFrame || true) {
        if (!isReRender) {
            firstFrame = false;
        }
        numFragmentsMaxColor = uint32_t(std::max(maxComplexity, uint64_t(4ull)) / intensity);
    }

    if (performanceMeasureMode) {
        (*sceneData->performanceMeasurer)->pushDepthComplexityFrame(
                minComplexity, maxComplexity,
                (float)totalNumFragments / usedLocations,
                (float)totalNumFragments / bufferSize, totalNumFragments);
    }*/

    /*if (totalNumFragments == 0) usedLocations = 1; // Avoid dividing by zero in code below
    std::cout << "Depth complexity: avg used: " << ((float)totalNumFragments / usedLocations)
              << ", avg all: " << ((float)totalNumFragments / bufferSize) << ", max: " << maxComplexity
              << ", #fragments: " << totalNumFragments << std::endl;*/
}
