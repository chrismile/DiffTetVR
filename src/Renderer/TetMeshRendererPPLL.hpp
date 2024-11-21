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

#ifndef DIFFTETVR_TETMESHRENDERERPPLL_HPP
#define DIFFTETVR_TETMESHRENDERERPPLL_HPP

#include "PPLL.hpp"

#include "TetMeshVolumeRenderer.hpp"

class GatherRasterPass;
class ResolveRasterPass;
class ClearRasterPass;
class AdjointRasterPass;

enum class FragmentBufferMode {
    BUFFER, BUFFER_ARRAY, BUFFER_REFERENCE_ARRAY
};
const char* const FRAGMENT_BUFFER_MODE_NAMES[3] = {
        "Buffer", "Buffer Array", "Buffer Reference Array"
};

const int MESH_MODE_DEPTH_COMPLEXITIES_PPLL[4][2] = {
        {20, 100}, // avg and max depth complexity medium
        {80, 256}, // avg and max depth complexity medium
        //{120, 380} // avg and max depth complexity very large
        {100, 520}, // avg and max depth complexity very large
        {400, 900} // avg and max depth complexity very large
};

/**
 * The order-independent transparency (OIT) technique per-pixel linked lists is used.
 * For more details see: Yang, J. C., Hensley, J., Grün, H. and Thibieroz, N., "Real-Time Concurrent
 * Linked List Construction on the GPU", Computer Graphics Forum, 29, 2010.
 */
class TetMeshRendererPPLL : public TetMeshVolumeRenderer {
public:
    explicit TetMeshRendererPPLL(
            sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow);
    ~TetMeshRendererPPLL() override;
    [[nodiscard]] RendererType getRendererType() const override { return RendererType::PPLL; }

    // Public interface.
    void recreateSwapchain(uint32_t width, uint32_t height) override;
    void setTetMeshData(const TetMeshPtr& _tetMesh) override;

    // Public interface, only for backward pass.
    void setAdjointPassData(
            sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
            sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer) override;
    virtual void recreateSwapchainExternal(
            uint32_t width, uint32_t height, size_t _fragmentBufferSize, const sgl::vk::BufferPtr& _fragmentBuffer,
            const sgl::vk::BufferPtr& _startOffsetBuffer, const sgl::vk::BufferPtr& _fragmentCounterBuffer);

    void getVulkanShaderPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines) override;
    void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) override;
    void setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp) override;
    void render() override;
    void renderAdjoint() override;

    /// Renders the GUI. The "reRender" flag might be set depending on the user's actions.
    void renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) override;

private:
    void onClearColorChanged() override;
    void updateLargeMeshMode();
    void reallocateFragmentBuffer() override;
    void clear();
    void gather();
    void resolve();
    void setShadersDirty(VolumeRendererPassType passType) override;

    void renderGuiMemory(sgl::PropertyEditor& propertyEditor) override;
    std::string getMaxDepthComplexityString() override { return sgl::toString(expectedMaxDepthComplexity); }

    // Render passes.
    std::shared_ptr<GatherRasterPass> gatherRasterPass;
    std::shared_ptr<ResolveRasterPass> resolveRasterPass;
    std::shared_ptr<ClearRasterPass> clearRasterPass;
    std::shared_ptr<AdjointRasterPass> adjointRasterPass; // only for optimization

    // Sorting algorithm for PPLL.
    SortingAlgorithmMode sortingAlgorithmMode = SORTING_ALGORITHM_MODE_PRIORITY_QUEUE;

    // Per-pixel linked list data (some entries are in parent class).
    FragmentBufferMode fragmentBufferMode = FragmentBufferMode::BUFFER;
    size_t maxStorageBufferSize = 0;
    size_t maxDeviceMemoryBudget = 0;
    size_t numFragmentBuffers = 1;
    size_t cachedNumFragmentBuffers = 1;

    // Uniform data buffer shared by all shaders.
    struct UniformData {
        // Inverse of (projectionMatrix * viewMatrix).
        glm::mat4 inverseViewProjectionMatrix;

        // Number of fragments we can store in total.
        uint32_t linkedListSize;
        // Size of the viewport in x direction (in pixels).
        int viewportW;
        // Camera near/far plane distance.
        float zNear, zFar;

        // Camera front vector.
        glm::vec3 cameraFront;
        // Volume attenuation.
        float attenuationCoefficient;

        glm::vec3 cameraPosition;
        float cameraPositionPadding{};

        // Viewport size in x/y direction.
        glm::uvec2 viewportSize;

        // Size of the viewport in x direction (in pixels) without padding.
        int viewportLinearW;
        int paddingUniform{};
    };
    UniformData uniformData = {};
    sgl::vk::BufferPtr uniformDataBuffer;

    // Per-pixel linked list settings.
    enum LargeMeshMode {
        MESH_SIZE_SMALL, MESH_SIZE_MEDIUM, MESH_SIZE_MEDIUM_LARGE, MESH_SIZE_LARGE
    };
    LargeMeshMode largeMeshMode = MESH_SIZE_SMALL;
    int expectedAvgDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[0][0];
    int expectedMaxDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[0][1];
};

#endif //DIFFTETVR_TETMESHRENDERERPPLL_HPP
