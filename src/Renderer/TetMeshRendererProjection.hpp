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

class SortPass;
class GenerateTrianglesPass;
class ComputeTrianglesDepthPass;
class ProjectedRasterPass;
class AdjointProjectedRasterPass;

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

    // Public interface.
    void recreateSwapchain(uint32_t width, uint32_t height) override;
    void setTetMeshData(const TetMeshPtr& _tetMesh) override;

    // Public interface, only for backward pass.
    void setAdjointPassData(
            sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
            sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer) override;
    virtual void recreateSwapchainExternal(
            uint32_t width, uint32_t height, size_t _fragmentBufferSize, sgl::vk::BufferPtr _fragmentBuffer,
            sgl::vk::BufferPtr _startOffsetBuffer, sgl::vk::BufferPtr _fragmentCounterBuffer);

    void getVulkanShaderPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines) override;
    void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) override;
    void setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp) override;
    void render() override;
    void renderAdjoint() override;

    /// Renders the GUI. The "reRender" flag might be set depending on the user's actions.
    void renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) override;

private:
    void onClearColorChanged() override;
    void setShadersDirty(VolumeRendererPassType passType) override;

    // Render passes.
    std::shared_ptr<SortPass> sortPass;
    std::shared_ptr<GenerateTrianglesPass> generateTrianglesPass;
    std::shared_ptr<ProjectedRasterPass> projectedRasterPass;
    // TODO, use fragment shader interlock for adjoint pass.
    std::shared_ptr<AdjointProjectedRasterPass> adjointProjectedRasterPass; // only for optimization

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
};

#endif //DIFFTETVR_TETMESHRENDERERPROJECTION_HPP
