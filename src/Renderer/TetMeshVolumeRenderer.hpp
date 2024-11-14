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

#include <map>
#include <memory>

#include <Graphics/Color.hpp>
#include <Graphics/Scene/Camera.hpp>

#include "Tet/TetQuality.hpp"

namespace sgl { namespace vk {
class Renderer;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
class RenderData;
typedef std::shared_ptr<RenderData> RenderDataPtr;
class Framebuffer;
typedef std::shared_ptr<Framebuffer> FramebufferPtr;
}}

namespace sgl {
class PropertyEditor;
class TransferFunctionWindow;
}

class TetMesh;
typedef std::shared_ptr<TetMesh> TetMeshPtr;

enum class AlphaMode {
    PREMUL, STRAIGHT
};

/*
 * Mainly for specifying which shaders to reload.
 */
enum class VolumeRendererPassType {
    GATHER = 1, // Render the geometry into a buffer
    RESOLVE = 2, // Take the buffer and produce the final image
    OTHER = 4, // Other passes
    ALL = int(GATHER) | int(RESOLVE) | int(OTHER)
};

enum class RendererType {
    PPLL, PROJECTION, INTERSECTION
};

class TetMeshVolumeRenderer {
public:
    explicit TetMeshVolumeRenderer(
            sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow);
    virtual ~TetMeshVolumeRenderer();
    [[nodiscard]] virtual RendererType getRendererType() const = 0;

    // Public interface.
    void setOutputImage(sgl::vk::ImageViewPtr& colorImage);
    virtual void recreateSwapchain(uint32_t width, uint32_t height);
    virtual void setUseLinearRGB(bool _useLinearRGB);
    void setCoarseToFineTargetNumTets(uint32_t _coarseToFineMaxNumTets);
    virtual void setTetMeshData(const TetMeshPtr& _tetMesh);
    [[nodiscard]] inline float getAttenuationCoefficient() const { return attenuationCoefficient; }
    void setAttenuationCoefficient(float _attenuationCoefficient) { attenuationCoefficient = _attenuationCoefficient; reRender = true; }
    virtual void setClearColor(const sgl::Color& _clearColor);
    void setNewTilingMode(int newTileWidth, int newTileHeight, bool useMortonCode = false);

    // Public interface, only for backward pass.
    virtual void setAdjointPassData(
            sgl::vk::ImageViewPtr _colorAdjointImage, sgl::vk::ImageViewPtr _adjointPassBackbuffer,
            sgl::vk::BufferPtr _vertexPositionGradientBuffer, sgl::vk::BufferPtr _vertexColorGradientBuffer);
    void setUseExternalFragmentBuffer(bool _useExternal) { useExternalFragmentBuffer = _useExternal; }
    virtual void recreateSwapchainExternal(
            uint32_t width, uint32_t height, size_t _fragmentBufferSize, const sgl::vk::BufferPtr& _fragmentBuffer,
            const sgl::vk::BufferPtr& _startOffsetBuffer, const sgl::vk::BufferPtr& _fragmentCounterBuffer);

    virtual void getVulkanShaderPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines);
    virtual void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData);
    virtual void setFramebufferAttachments(sgl::vk::FramebufferPtr& framebuffer, VkAttachmentLoadOp loadOp);
    virtual void render()=0;
    virtual void renderAdjoint()=0;

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
    virtual void onHasMoved();
    /// Called when the transfer function (for tet quality analysis) has been rebuilt.
    virtual void onTransferFunctionMapRebuilt() {}
    /// Renders the GUI. The "reRender" flag might be set depending on the user's actions.
    virtual void renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor)=0;

    /// Returns screen width and screen height padded for tile size
    void getScreenSizeWithTiling(int& screenWidth, int& screenHeight);

protected:
    virtual void onClearColorChanged()=0;
    virtual void reallocateFragmentBuffer() {}
    virtual void setShadersDirty(VolumeRendererPassType passType)=0;

    void renderGuiShared(sgl::PropertyEditor& propertyEditor);
    virtual void renderGuiMemory(sgl::PropertyEditor& propertyEditor);
    virtual std::string getMaxDepthComplexityString() { return "unlimited"; }

    sgl::vk::Renderer* renderer;
    sgl::CameraPtr* camera;
    TetMeshPtr tetMesh;
    sgl::vk::ImageViewPtr outputImageView;
    sgl::Color clearColor;
    bool reRender = false;
    AlphaMode alphaMode = AlphaMode::PREMUL;

    // Only for tests.
    glm::mat4 projMat;

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

    // Window data.
    int windowWidth = 0, windowHeight = 0;
    int paddedWindowWidth = 0, paddedWindowHeight = 0;

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

    float attenuationCoefficient = 100.0f;
    bool useCoarseToFine = false;
    uint32_t coarseToFineMaxNumTets = 0;

    // Per-pixel linked list data (only written to in subclass TetMeshRendererPPLL).
    bool useExternalFragmentBuffer = false;
    size_t fragmentBufferSize = 0;
    sgl::vk::BufferPtr fragmentBuffer; //< if fragmentBufferMode == FragmentBufferMode::BUFFER
    std::vector<sgl::vk::BufferPtr> fragmentBuffers; //< if fragmentBufferMode != FragmentBufferMode::BUFFER
    sgl::vk::BufferPtr fragmentBufferReferenceBuffer; //< if fragmentBufferMode == FragmentBufferMode::BUFFER_REFERENCE_ARRAY
    sgl::vk::BufferPtr startOffsetBuffer;
    sgl::vk::BufferPtr fragmentCounterBuffer;

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
