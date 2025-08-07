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

#ifndef DIFFTETVR_REGULARGRIDRENDERERDVR_HPP
#define DIFFTETVR_REGULARGRIDRENDERERDVR_HPP

#include <memory>

#include <Graphics/Color.hpp>
#include <Graphics/Scene/Camera.hpp>

#ifdef BUILD_PYTHON_MODULE
#include <torch/types.h>
#ifdef SUPPORT_COMPUTE_INTEROP
#include <Graphics/Vulkan/Utils/InteropCompute.hpp>
#endif
#endif

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
class TransferFunctionWindow;
#ifndef DISABLE_IMGUI
class PropertyEditor;
#endif
}

class RegularGrid;
typedef std::shared_ptr<RegularGrid> RegularGridPtr;
class RegularGridDvrPass;

enum class RegularGridInterpolationMode {
    NEAREST, LINEAR
};
const char* const REGULAR_GRID_INTERPOLATION_MODE_NAMES[] = {
        "Nearest", "Linear"
};


/**
 * For testing against rendering of regular grids using DVR, this class supports rendering scalar data on regular grids.
 */
class RegularGridRendererDVR {
public:
    RegularGridRendererDVR(
            sgl::vk::Renderer* renderer, sgl::CameraPtr* camera, sgl::TransferFunctionWindow* transferFunctionWindow);
    ~RegularGridRendererDVR();

    // Public interface.
    void setOutputImage(sgl::vk::ImageViewPtr& colorImage);
    virtual void recreateSwapchain(uint32_t width, uint32_t height);
    virtual void setUseLinearRGB(bool _useLinearRGB);
    virtual void setRegularGridData(const RegularGridPtr& _regularGrid);
    [[nodiscard]] const RegularGridPtr& getRegularGridData() { return regularGrid; }
    bool loadTransferFunctionFromFile(const std::string& filePath);
    [[nodiscard]] inline float getAttenuationCoefficient() const { return attenuationCoefficient; }
    void setAttenuationCoefficient(float _attenuationCoefficient) { attenuationCoefficient = _attenuationCoefficient; reRender = true; }
    virtual void setClearColor(const sgl::Color& _clearColor);
    [[nodiscard]] inline float getStepSize() const { return stepSize; }
    void setStepSize(float _stepSize) { stepSize = _stepSize; reRender = true; }
    [[nodiscard]] inline sgl::vk::Renderer* getRenderer() const { return renderer; }
    [[nodiscard]] inline const sgl::CameraPtr& getCamera() const { return *camera; }
    [[nodiscard]] inline sgl::TransferFunctionWindow* getTransferFunctionWindow() { return transferFunctionWindow; }
    void setInterpolationMode(RegularGridInterpolationMode _regularGridInterpolationMode);
    [[nodiscard]] inline const sgl::vk::ImageSamplerPtr& getImageSampler() { return imageSampler; }
    [[nodiscard]] inline const sgl::vk::ImageViewPtr& getOutputImageView() { return outputImageView; }

    void render();

    // PyTorch buffer interface.
#ifdef BUILD_PYTHON_MODULE
    void setUseComputeInterop(bool _useComputeInterop);
    void setUsedDeviceType(torch::DeviceType _usedDeviceType);
    void setViewportSize(uint32_t viewportWidth, uint32_t viewportHeight);
    torch::Tensor getImageTensor();
    void copyOutputImageToBuffer();
#endif

private:
    void onClearColorChanged();

    sgl::vk::Renderer* renderer;
    sgl::CameraPtr* camera;
    sgl::TransferFunctionWindow* transferFunctionWindow;
    RegularGridPtr regularGrid;

    sgl::vk::ImageViewPtr outputImageView;
    sgl::Color clearColor;
    bool reRender = false;

#ifdef BUILD_PYTHON_MODULE
    bool useComputeInterop = false;
    torch::DeviceType usedDeviceType = torch::DeviceType::CPU;
#ifdef SUPPORT_COMPUTE_INTEROP
    sgl::vk::BufferPtr colorImageBuffer;
    sgl::vk::BufferVkComputeApiExternalMemoryPtr colorImageBufferCu;
#endif
    // CPU interop.
    sgl::vk::BufferPtr colorImageBufferCpu;
    void* colorImageBufferCpuPtr = nullptr;
#endif

    float attenuationCoefficient = 100.0f;
    float stepSize = 0.1f;
    float voxelSize = 1.0f;

    // Clip plane data.
    struct RenderSettingsData {
        glm::mat4 inverseViewMatrix;
        glm::mat4 inverseProjectionMatrix;
        glm::vec4 backgroundColor;
        glm::vec3 minBoundingBox;
        float attenuationCoefficient = 100.0f;
        glm::vec3 maxBoundingBox;
        float stepSize;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;

    void createImageSampler();
    RegularGridInterpolationMode regularGridInterpolationMode = RegularGridInterpolationMode::LINEAR;
    sgl::vk::ImageSamplerPtr imageSampler{};

    std::shared_ptr<RegularGridDvrPass> regularGridDvrPass;
};

#endif //DIFFTETVR_REGULARGRIDRENDERERDVR_HPP
