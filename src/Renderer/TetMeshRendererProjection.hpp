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

#ifndef DIFFTETVR_TETMESHRENDERERPROJECTION_HPP
#define DIFFTETVR_TETMESHRENDERERPROJECTION_HPP

#include "TetMeshVolumeRenderer.hpp"
#include "RasterCommon.hpp"

class GenerateTrianglesProjPass;
class InitializeIndirectCommandBufferProjPass;
class ComputeTrianglesDepthProjPass;
class ProjectedRasterPass;
class AdjointProjectedRasterPass;
class CompactTriangleTetListPass;
class InitializeIndirectCommandBufferAdjointPass;
class AdjointGenerateTrianglesPass;

/**
 * The tetrahedral elements are projected to triangles in a preprocess pass.
 * For more details see:
 * - Shirley, P., Tuchmann, A., "A Polygonal Approximation to Direct Scalar Volume Rendering",
 *   Proceedings of the 1990 Workshop on Volume Visualization.
 * - Kraus, M., Qiao, W., Ebert, D. S., "Projecting Tetrahedra without Rendering Artifacts",
 *   IEEE Visualization 2004.
 */
class TetMeshRendererProjection : public TetMeshVolumeRenderer {
public:
    explicit TetMeshRendererProjection(
            sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow);
    ~TetMeshRendererProjection() override;
    [[nodiscard]] RendererType getRendererType() const override { return RendererType::PROJECTION; }

    // Public interface.
    void recreateSwapchain(uint32_t width, uint32_t height) override;
    void setTetMeshData(const TetMeshPtr& _tetMesh) override;

    // Public interface, only for backward pass.
    void setAdjointPassData(
            sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer) override;
    void recreateSwapchainExternal(
            uint32_t width, uint32_t height, size_t _fragmentBufferSize, const sgl::vk::BufferPtr& _fragmentBuffer,
            const sgl::vk::BufferPtr& _startOffsetBuffer, const sgl::vk::BufferPtr& _fragmentCounterBuffer) override;

    [[nodiscard]] const sgl::vk::BufferPtr& getUniformDataBuffer() { return uniformDataBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleCounterBuffer() { return triangleCounterBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexPositionBuffer() { return triangleVertexPositionBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexColorBuffer() { return triangleVertexColorBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexDepthBuffer() { return triangleVertexDepthBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleTetIndexBuffer() { return triangleTetIndexBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getDrawIndirectBuffer() { return drawIndirectBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getDispatchIndirectBuffer() { return dispatchIndirectBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleKeyValueBuffer() { return triangleKeyValueBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getSortedTriangleKeyValueBuffer() { return sortedTriangleKeyValueBuffer; }

    // Adjoint pass buffers.
    [[nodiscard]] const sgl::vk::BufferPtr& getTetCounterBuffer() { return tetCounterBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTetTriangleOffsetBuffer() { return tetTriangleOffsetBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getDispatchIndirectAdjointBuffer() { return dispatchIndirectAdjointBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexPositionGradientBuffer() { return triangleVertexPositionGradientBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexColorGradientBuffer() { return triangleVertexColorGradientBuffer; }
    [[nodiscard]] const sgl::vk::BufferPtr& getTriangleVertexDepthGradientBuffer() { return triangleVertexDepthGradientBuffer; }

    void getVulkanShaderPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines) override;
    void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) override;
    void setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp) override;
    void render() override;
    void renderAdjoint() override;

#ifndef DISABLE_IMGUI
    /// Renders the GUI. The "reRender" flag might be set depending on the user's actions.
    void renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) override;
#endif

private:
    void onClearColorChanged() override;
    void setShadersDirty(VolumeRendererPassType passType) override;
    void createTriangleCounterBuffer();
    void recreateSortingBuffers();
    void computeLinearDepthCorrection();

    // For sorting.
#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    radix_sort_vk_target* radixSortVkTarget = nullptr;
    radix_sort_vk* radixSortVk = nullptr;
#endif

    // Render passes.
    std::shared_ptr<GenerateTrianglesProjPass> generateTrianglesPass;
    std::shared_ptr<InitializeIndirectCommandBufferProjPass> initializeIndirectCommandBufferPass;
    std::shared_ptr<ComputeTrianglesDepthProjPass> computeTrianglesDepthPass;
    std::shared_ptr<ProjectedRasterPass> projectedRasterPass;
    // only for optimization
    std::shared_ptr<AdjointProjectedRasterPass> adjointProjectedRasterPass;
    std::shared_ptr<CompactTriangleTetListPass> compactTriangleTetListPass;
    std::shared_ptr<InitializeIndirectCommandBufferAdjointPass> initializeIndirectCommandBufferAdjointPass;
    std::shared_ptr<AdjointGenerateTrianglesPass> adjointGenerateTrianglesPass;

    // Uniform data buffer shared by all shaders.
    struct UniformData {
        glm::mat4 viewProjMat;
        glm::mat4 invProjMat;
        glm::vec3 cameraPosition;
        float attenuationCoefficient;
        uint32_t numTets;
        uint32_t useLinearDepthCorrection;
        float linearDepthCorrection;
        float pad0;
    };
    UniformData uniformData = {};
    sgl::vk::BufferPtr uniformDataBuffer;

    bool useGradientsCached = false;
    sgl::vk::BufferPtr triangleCounterBuffer; // 1x uint
    sgl::vk::BufferPtr triangleVertexPositionBuffer; // ?x vec4
    sgl::vk::BufferPtr triangleVertexColorBuffer; // ?x vec4
    sgl::vk::BufferPtr triangleVertexDepthBuffer; // ?x float
    sgl::vk::BufferPtr triangleTetIndexBuffer; // ?x uint; only if adjoint support is used
    sgl::vk::BufferPtr drawIndirectBuffer; // 1x VkDrawIndirectCommand (4x uint32_t)
    sgl::vk::BufferPtr dispatchIndirectBuffer; // 1x VkDispatchIndirectCommand (3x uint32_t)
    sgl::vk::BufferPtr triangleKeyValueBuffer; // ?x uint64_t

    // Adjoint pass buffers.
    sgl::vk::BufferPtr tetCounterBuffer; // 1x uint
    sgl::vk::BufferPtr tetTriangleOffsetBuffer; // ?x uint
    sgl::vk::BufferPtr dispatchIndirectAdjointBuffer; // 1x VkDispatchIndirectCommand (3x uint32_t)
    sgl::vk::BufferPtr triangleVertexPositionGradientBuffer; // ?x vec3
    sgl::vk::BufferPtr triangleVertexColorGradientBuffer; // ?x vec4
    sgl::vk::BufferPtr triangleVertexDepthGradientBuffer; // ?x float

    SortingAlgorithm sortingAlgorithm = SortingAlgorithm::FUCHSIA_RADIX_SORT;

    // For sorting with radix sort.
    sgl::vk::BufferPtr sortingBufferEven;
    sgl::vk::BufferPtr sortingBufferOdd;
    sgl::vk::BufferPtr sortedTriangleKeyValueBuffer; // One of the two buffers above.
    sgl::vk::BufferPtr sortingInternalBuffer;
    sgl::vk::BufferPtr sortingIndirectBuffer;

    // For sorting on the CPU.
    sgl::vk::BufferPtr triangleCounterBufferCpu;
    sgl::vk::BufferPtr triangleKeyValueBufferCpu;
};

#endif //DIFFTETVR_TETMESHRENDERERPROJECTION_HPP
