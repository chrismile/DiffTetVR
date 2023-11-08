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

#ifndef DIFFTETVR_DEVICEBUFFER_HPP
#define DIFFTETVR_DEVICEBUFFER_HPP

#include <memory>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

namespace sgl { namespace vk {
class Image;
typedef std::shared_ptr<Image> ImagePtr;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
class ImageSampler;
typedef std::shared_ptr<ImageSampler> ImageSamplerPtr;
class Texture;
typedef std::shared_ptr<Texture> TexturePtr;
class ImageCudaExternalMemoryVk;
typedef std::shared_ptr<ImageCudaExternalMemoryVk> ImageCudaExternalMemoryVkPtr;
class TextureCudaExternalMemoryVk;
typedef std::shared_ptr<TextureCudaExternalMemoryVk> TextureCudaExternalMemoryVkPtr;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
}}

class DeviceBuffer {
public:
    DeviceBuffer();
    inline const sgl::vk::BufferPtr& getVulkanBuffer() { return vulkanBuffer; }
#ifdef SUPPORT_CUDA_INTEROP
    CUdeviceptr getCudaBuffer();
    const sgl::vk::BufferCudaExternalMemoryVkPtr& getBufferCudaExternalMemory();
#endif

private:
    sgl::vk::BufferPtr vulkanBuffer;

#ifdef SUPPORT_CUDA_INTEROP
    /// Optional, created when @see getCudaBuffer is called.
    sgl::vk::BufferCudaExternalMemoryVkPtr cudaBuffer;
#endif
};

typedef std::shared_ptr<DeviceBuffer> DeviceBufferPtr;

#endif //DIFFTETVR_DEVICEBUFFER_HPP
