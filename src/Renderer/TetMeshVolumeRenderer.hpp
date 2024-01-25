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

#ifndef DIFFTETVR_TETMESHVOLUMERENDERER_HPP
#define DIFFTETVR_TETMESHVOLUMERENDERER_HPP

#include "PPLL.hpp"

#include <memory>

#include <Graphics/Color.hpp>
#include <Graphics/Scene/Camera.hpp>

#include "Tet/TetQuality.hpp"

namespace sgl { namespace vk {
class Renderer;
class ImageView;
class RenderData;
typedef std::shared_ptr<RenderData> RenderDataPtr;
}}

namespace sgl {
class PropertyEditor;
class TransferFunctionWindow;
}

class TetMesh;
typedef std::shared_ptr<TetMesh> TetMeshPtr;

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

const int MESH_MODE_DEPTH_COMPLEXITIES_PPLL[3][2] = {
        {20, 100}, // avg and max depth complexity medium
        //{80, 256}, // avg and max depth complexity medium
        //{120, 380} // avg and max depth complexity very large
        {100, 520}, // avg and max depth complexity very large
        {400, 900} // avg and max depth complexity very large
};

enum class AlphaMode {
    PREMUL, STRAIGHT
};

/**
 * The order-independent transparency (OIT) technique per-pixel linked lists is used.
 * For more details see: Yang, J. C., Hensley, J., Grün, H. and Thibieroz, N., "Real-Time Concurrent
 * Linked List Construction on the GPU", Computer Graphics Forum, 29, 2010.
 */
class TetMeshVolumeRenderer {
public:
    explicit TetMeshVolumeRenderer(
            sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow);
    ~TetMeshVolumeRenderer();

    // Public interface.
    void setOutputImage(sgl::vk::ImageViewPtr& colorImage);
    void recreateSwapchain(uint32_t width, uint32_t height);
    void setUseLinearRGB(bool _useLinearRGB);
    void setTetMeshData(const TetMeshPtr& _tetMesh);
    void setAttenuationCoefficient(float _attenuationCoefficient) { attenuationCoefficient = _attenuationCoefficient; reRender = true; }
    void setClearColor(const sgl::Color& _clearColor);
    void setNewTilingMode(int newTileWidth, int newTileHeight, bool useMortonCode = false);

    // Public interface, only for backward pass.
    void setAdjointPassData(
            sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
            sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer);
    void setUseExternalFragmentBuffer(bool _useExternal) { useExternalFragmentBuffer = _useExternal; }
    void recreateSwapchainExternal(
            uint32_t width, uint32_t height, size_t _fragmentBufferSize, sgl::vk::BufferPtr _fragmentBuffer,
            sgl::vk::BufferPtr _startOffsetBuffer, sgl::vk::BufferPtr _fragmentCounterBuffer);

    void getVulkanShaderPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines);
    void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData);
    void setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp);
    void render();
    void renderAdjoint();

    [[nodiscard]] inline sgl::vk::Renderer* getRenderer() const { return renderer; }
    [[nodiscard]] inline const TetMeshPtr& getTetMesh() const { return tetMesh; }
    [[nodiscard]] inline const sgl::vk::ImageViewPtr& getOutputImageView() const { return outputImageView; }
    [[nodiscard]] inline size_t getFragmentBufferSize() const { return fragmentBufferSize; }
    [[nodiscard]] inline const sgl::vk::BufferPtr& getFragmentBuffer() const { return fragmentBuffer; }
    [[nodiscard]] inline const sgl::vk::BufferPtr& getStartOffsetBuffer() const { return startOffsetBuffer; }
    [[nodiscard]] inline const sgl::vk::BufferPtr& getFragmentCounterBuffer() const { return fragmentCounterBuffer; }
    [[nodiscard]] inline const sgl::vk::BufferPtr& getDepthComplexityCounterBuffer() const { return depthComplexityCounterBuffer; }
    [[nodiscard]] inline AlphaMode getAlphaMode() const { return alphaMode; }
    [[nodiscard]] inline bool getShowDepthComplexity() const { return showDepthComplexity; }
    [[nodiscard]] inline bool getShowTetQuality() const { return showTetQuality; }
    [[nodiscard]] inline bool getUseShading() const { return useShading; }
    [[nodiscard]] inline TetQualityMetric getTetQualityMetric() const { return tetQualityMetric; }
    [[nodiscard]] inline sgl::TransferFunctionWindow* getTransferFunctionWindow() { return transferFunctionWindow; }

    /// Returns if the data needs to be re-rendered, but the visualization mapping is valid.
    bool needsReRender();
    /// Called when the camera has moved.
    void onHasMoved();
    /// Called when the transfer function (for tet quality analysis) has been rebuilt.
    void onTransferFunctionMapRebuilt();
    /// Renders the GUI. The "reRender" flag might be set depending on the user's actions.
    void renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor);

    /// Returns screen width and screen height padded for tile size
    void getScreenSizeWithTiling(int& screenWidth, int& screenHeight);

private:
    void onClearColorChanged();
    void updateLargeMeshMode();
    void reallocateFragmentBuffer();
    void clear();
    void gather();
    void resolve();

    sgl::vk::Renderer* renderer;
    sgl::CameraPtr* camera;
    TetMeshPtr tetMesh;
    sgl::vk::ImageViewPtr outputImageView;
    sgl::Color clearColor;
    bool reRender = false;
    AlphaMode alphaMode = AlphaMode::PREMUL;

    // Only for tests.
    glm::mat4 projMat;

    // Render passes.
    std::shared_ptr<GatherRasterPass> gatherRasterPass;
    std::shared_ptr<ResolveRasterPass> resolveRasterPass;
    std::shared_ptr<ClearRasterPass> clearRasterPass;
    std::shared_ptr<AdjointRasterPass> adjointRasterPass; // only for optimization

    // Sorting algorithm for PPLL.
    SortingAlgorithmMode sortingAlgorithmMode = SORTING_ALGORITHM_MODE_PRIORITY_QUEUE;

    // Per-pixel linked list data.
    FragmentBufferMode fragmentBufferMode = FragmentBufferMode::BUFFER;
    bool useExternalFragmentBuffer = false;
    size_t maxStorageBufferSize = 0;
    size_t maxDeviceMemoryBudget = 0;
    size_t fragmentBufferSize = 0;
    size_t numFragmentBuffers = 1;
    size_t cachedNumFragmentBuffers = 1;
    sgl::vk::BufferPtr fragmentBuffer; //< if fragmentBufferMode == FragmentBufferMode::BUFFER
    std::vector<sgl::vk::BufferPtr> fragmentBuffers; //< if fragmentBufferMode != FragmentBufferMode::BUFFER
    sgl::vk::BufferPtr fragmentBufferReferenceBuffer; //< if fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY
    sgl::vk::BufferPtr startOffsetBuffer;
    sgl::vk::BufferPtr fragmentCounterBuffer;

    // For showing tet mesh quality metrics.
    bool showTetQuality = false;
    bool useShading = false;
    TetQualityMetric tetQualityMetric = DEFAULT_QUALITY_METRIC;
    sgl::TransferFunctionWindow* transferFunctionWindow;

    // For adjoint pass.
    sgl::vk::ImageViewPtr colorAdjointImage;
    sgl::vk::ImageViewPtr adjointPassBackbuffer;
    sgl::vk::BufferPtr vertexPositionGradientBuffer;
    sgl::vk::BufferPtr vertexColorGradientBuffer;

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

    // Window data.
    int windowWidth = 0, windowHeight = 0;
    int paddedWindowWidth = 0, paddedWindowHeight = 0;

    // Per-pixel linked list settings.
    enum LargeMeshMode {
        MESH_SIZE_SMALL, MESH_SIZE_MEDIUM, MESH_SIZE_LARGE
    };
    LargeMeshMode largeMeshMode = MESH_SIZE_SMALL;
    int expectedAvgDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[0][0];
    int expectedMaxDepthComplexity = MESH_MODE_DEPTH_COMPLEXITIES_PPLL[0][1];
    float attenuationCoefficient = 100.0f;

    // Depth complexity information mode.
    bool showDepthComplexity = false;
    void computeStatistics(bool isReRender);
    void createDepthComplexityBuffers();
    sgl::vk::BufferPtr depthComplexityCounterBuffer;
    std::vector<sgl::vk::BufferPtr> stagingBuffers;
    bool firstFrame = true;
    bool statisticsUpToDate = false;
    float counterPrintFrags = 0.0f;
    uint64_t totalNumFragments = 0;
    uint64_t usedLocations = 1;
    uint64_t maxComplexity = 0;
    uint64_t bufferSize = 1;

    // Tiling mode.
    int tilingModeIndex = 2;
    int tileWidth = 2;
    int tileHeight = 8;
    bool tilingUseMortonCode = false;
    /**
     * Uses ImGui to render a tiling mode selection window.
     * @return True if a new tiling mode was selected (shaders need to be reloaded in this case).
     */
    bool selectTilingModeUI(sgl::PropertyEditor& propertyEditor);
};

#endif //DIFFTETVR_TETMESHVOLUMERENDERER_HPP
