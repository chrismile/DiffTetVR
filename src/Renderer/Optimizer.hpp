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

#ifndef DIFFTETVR_OPTIMIZER_HPP
#define DIFFTETVR_OPTIMIZER_HPP

#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "OptimizerDefines.hpp"

namespace sgl {
class Camera;
typedef std::shared_ptr<Camera> CameraPtr;
}

class TetMesh;
typedef std::shared_ptr<TetMesh> TetMeshPtr;
class TetMeshVolumeRenderer;
class LossPass;
class TetRegularizerPass;
class OptimizerPass;
class VtkWriter;

class ImGuiVulkanImage {
public:
    ImGuiVulkanImage() = default;
    ImGuiVulkanImage(sgl::vk::Renderer* renderer, sgl::vk::TexturePtr texture);
    ~ImGuiVulkanImage();
    ImTextureID getImGuiTextureId();
    ImVec2 getTextureSizeImVec2();

private:
    sgl::vk::Renderer* renderer{};
    sgl::vk::TexturePtr texture{};
    VkDescriptorSet descriptorSetImGui{};
};

class TetMeshOptimizer {
public:
    TetMeshOptimizer(
            sgl::vk::Renderer* renderer, std::function<void(const TetMeshPtr&)> setTetMeshCallback,
            bool hasDataSets, std::function<std::string()> renderGuiDataSetSelectionMenuCallback,
            sgl::TransferFunctionWindow* transferFunctionWindow);

    void openDialog();
    void renderGuiDialog();
    inline bool getNeedsReRender() { bool tmp = needsReRender; needsReRender = false; return tmp; }

    float getProgress();
    void startRequest();
    bool getHasResult();
    void updateRequest();

private:
    void sampleCameraPoses();
    uint32_t cachedViewportWidth = 0;
    uint32_t cachedViewportHeight = 0;

    sgl::vk::Renderer* renderer = nullptr;
    bool needsReRender = false;
    bool isOptimizationSettingsDialogOpen = false;
    bool isOptimizationProgressDialogOpen = false;

    // Settings.
    OptimizationSettings settings{};
    TetMeshPtr tetMeshGT, tetMeshOpt;
    std::function<void(const TetMeshPtr&)> setTetMeshCallback;
    bool hasDataSets = false;
    std::function<std::string()> renderGuiDataSetSelectionMenuCallback;
    bool hasRequest = false;
    sgl::TransferFunctionWindow* transferFunctionWindow;
    uint32_t numCellsGT = 0, numCellsOpt = 0;

    void recreateGradientBuffers();
    void coarseToFineSubdivide(const glm::vec3* vertexPositionGradients, const glm::vec4* vertexColorGradients);
    const int COARSE_TO_FINE_EPOCH_COLOR = 0;
    const int COARSE_TO_FINE_EPOCH_COLOR_POS = 1;
    const int COARSE_TO_FINE_EPOCH_GATHER = 2;
    bool usePreviewCached = false;
    uint32_t viewportWidth = 0;
    uint32_t viewportHeight = 0;
    int currentEpoch = 0;
    int coarseToFineEpoch = 0;
    uint32_t numCellsInit = 0;
    sgl::CameraPtr camera;

    // Image data & adjoint image data.
    sgl::vk::TexturePtr colorImageGT, colorImageOpt;
    sgl::vk::TexturePtr colorAdjointTexture;
    sgl::vk::ImageViewPtr adjointPassBackbuffer;
    // Gradient data for tetMeshOpt.
    sgl::vk::BufferPtr vertexPositionGradientBuffer;
    sgl::vk::BufferPtr vertexColorGradientBuffer;
    // Per-pixel linked list.
    size_t fragmentBufferSize = 0;
    sgl::vk::BufferPtr fragmentBuffer;
    sgl::vk::BufferPtr startOffsetBuffer;
    sgl::vk::BufferPtr fragmentCounterBuffer;

    int previewDelay = 0;
    bool showPreview = true;
    std::shared_ptr<ImGuiVulkanImage> colorImageGTImGui, colorImageOptImGui;

    // For exporting position gradients to a file.
    std::shared_ptr<VtkWriter> vtkWriter;
    sgl::vk::BufferPtr vertexPositionStagingBuffer;
    sgl::vk::BufferPtr vertexColorStagingBuffer;
    sgl::vk::BufferPtr vertexPositionGradientStagingBuffer;
    sgl::vk::BufferPtr vertexColorGradientStagingBuffer;

    // For Adam.
    sgl::vk::BufferPtr firstMomentEstimateBuffer;
    sgl::vk::BufferPtr secondMomentEstimateBuffer;

    // Compute passes.
    std::shared_ptr<TetMeshVolumeRenderer> tetMeshVolumeRendererGT;
    std::shared_ptr<TetMeshVolumeRenderer> tetMeshVolumeRendererOpt;
    std::shared_ptr<LossPass> lossPass;
    std::shared_ptr<TetRegularizerPass> tetRegularizerPass;
    std::shared_ptr<OptimizerPass> optimizerPassPositions;
    std::shared_ptr<OptimizerPass> optimizerPassColors;
};

#endif //DIFFTETVR_OPTIMIZER_HPP
