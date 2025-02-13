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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <torch/script.h>
#include <torch/types.h>
#include <torch/extension.h>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif
#ifdef SUPPORT_HIP_INTEROP
#include <Graphics/Vulkan/Utils/InteropHIP.hpp>
#endif
#ifdef SUPPORT_SYCL_INTEROP
#include <Graphics/Vulkan/Utils/InteropLevelZero.hpp>
#endif
#ifdef SUPPORT_CUDA_INTEROP
#include <c10/cuda/CUDAStream.h>
#endif
#ifdef SUPPORT_HIP_INTEROP
#include <c10/hip/HIPStream.h>
#endif
#ifdef SUPPORT_SYCL_INTEROP
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/api/include/torch/xpu.h>
#endif

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Scene/RenderTarget.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#if defined(SUPPORT_CUDA_INTEROP) && CUDA_VERSION < 11020
#error Please install CUDA >= 11.2 for timeline semaphore support.
#endif

#if defined(SUPPORT_COMPUTE_INTEROP)
#include <Graphics/Vulkan/Utils/InteropCompute.hpp>
#endif

#ifdef SUPPORT_CUDA_INTEROP
namespace sgl { namespace vk {
DLL_OBJECT bool initializeCudaDeviceApiFunctionTable();
DLL_OBJECT bool getIsCudaDeviceApiFunctionTableInitialized();
DLL_OBJECT void freeCudaDeviceApiFunctionTable();
}}
#endif

#ifdef SUPPORT_HIP_INTEROP
namespace sgl { namespace vk {
DLL_OBJECT bool initializeHipDeviceApiFunctionTable();
DLL_OBJECT bool getIsHipDeviceApiFunctionTableInitialized();
DLL_OBJECT void freeHipDeviceApiFunctionTable();
}}
#endif

#ifdef SUPPORT_SYCL_INTEROP
namespace sgl { namespace vk {
DLL_OBJECT bool initializeLevelZeroFunctionTable();
DLL_OBJECT bool getIsLevelZeroFunctionTableInitialized();
DLL_OBJECT void freeLevelZeroFunctionTable();
}}
#endif

#include <Tet/TetMesh.hpp>
#include <Tet/RegularGrid.hpp>
#include <Tet/Writers/VtkWriter.hpp>
#include <Renderer/RadixSortHelper.hpp>
#include <Renderer/TetMeshVolumeRenderer.hpp>
#include <Renderer/TetMeshRendererPPLL.hpp>
#include <Renderer/TetMeshRendererProjection.hpp>
#include <Renderer/TetMeshRendererIntersection.hpp>
#include <Renderer/RegularGridRendererDVR.hpp>
#include <Renderer/TetRegularizerPass.hpp>
#include <Renderer/OptimizerDefines.hpp>
#include <Renderer/Optimizer.hpp>
#include "VoxelCarving/VoxelCarving.hpp"
#include "VoxelCarving/DenseVoxelCarvingCPU.hpp"

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
#include <radix_sort/radix_sort_vk.h>
#endif

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define _WIN32_IE 0x0400
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shlobj.h>
#include <shlwapi.h>
#include <windef.h>
#include <windows.h>
#endif

class ApplicationState {
public:
    ApplicationState();
    ~ApplicationState();
    void vulkanBegin();
    void vulkanFinished();
    void selectNextCommandBuffer();

    sgl::vk::Device* device = nullptr;
    sgl::vk::Renderer* renderer = nullptr;
    sgl::CameraPtr camera;
    sgl::TransferFunctionWindow* transferFunctionWindow = nullptr;
    TetMeshOptimizer* optimizer = nullptr;

    uint32_t cachedWidth = 0, cachedHeight = 0;

    torch::DeviceType usedDeviceType = torch::DeviceType::CPU;

    // For invocation of Vulkan command buffer in CUDA/HIP stream.
    size_t commandBufferIdx = 0;
    std::vector<sgl::vk::CommandBufferPtr> commandBuffers;
    std::vector<sgl::vk::FencePtr> fences;
    sgl::vk::CommandBufferPtr commandBuffer;
    sgl::vk::FencePtr fence;
    uint64_t timelineValue = 0;
#ifdef SUPPORT_COMPUTE_INTEROP
    sgl::vk::SemaphoreVkComputeApiInteropPtr renderReadySemaphore;
    sgl::vk::SemaphoreVkComputeApiInteropPtr renderFinishedSemaphore;
#endif
};
static ApplicationState* sState = nullptr;
static torch::DeviceType globalDeviceType = torch::DeviceType::CPU;
static bool globalDeviceTypeSetManually = false;

void vulkanErrorCallbackHeadless() {
    std::cerr << "Application callback" << std::endl;
}

static const char* argv[] = { "." }; //< Just pass something as argv.

std::string getLibraryPath() {
#if defined(_WIN32)
    // See: https://stackoverflow.com/questions/6924195/get-dll-path-at-runtime
    WCHAR modulePath[MAX_PATH];
    HMODULE hmodule{};
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPWSTR>(&getLibraryPath), &hmodule)) {
        GetModuleFileNameW(hmodule, modulePath, MAX_PATH);
        PathRemoveFileSpecW(modulePath); //< Needs linking with shlwapi.lib.
        return sgl::wideStringArrayToStdString(modulePath);
    } else {
        sgl::Logfile::get()->writeError(
                std::string() + "Error when calling GetModuleHandle: " + std::to_string(GetLastError()));
    }
    return "";
#elif defined(__linux__)
    // See: https://stackoverflow.com/questions/33151264/get-dynamic-library-directory-in-c-linux
    Dl_info dlInfo{};
    dladdr((void*)getLibraryPath, &dlInfo);
    std::string moduleDir = sgl::FileUtils::get()->getPathToFile(dlInfo.dli_fname);
    return moduleDir;
#else
    return ".";
#endif
}

ApplicationState::ApplicationState() {
    // Initialize the filesystem utilities.
    sgl::FileUtils::get()->initialize("pysrg", 1, argv);

    std::string dataDirectory = getLibraryPath();
    if (!sgl::endsWith(dataDirectory, "/") && !sgl::endsWith(dataDirectory, "\\")) {
        dataDirectory += "/";
    }
    dataDirectory += "Data";
    sgl::AppSettings::get()->setDataDirectory(dataDirectory);
    sgl::AppSettings::get()->initializeDataDirectory();

    sgl::AppSettings::get()->setSaveSettings(false);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);
    std::vector<const char*> optionalInstanceExtensions;
    sgl::AppSettings::get()->createHeadless();

    std::vector<const char*> optionalDeviceExtensions;
#ifdef SUPPORT_COMPUTE_INTEROP
    optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif
    optionalDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    optionalDeviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    optionalDeviceExtensions.push_back(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME); // for raster adjoint pass

    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    requestedDeviceFeatures.requestedPhysicalDeviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderInt64 = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.runtimeDescriptorArray = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;


    uint8_t* deviceUuid = nullptr;
    if (globalDeviceTypeSetManually) {
        usedDeviceType = globalDeviceType;
    } else {
        usedDeviceType = torch::DeviceType::CPU;
#ifdef SUPPORT_SYCL_INTEROP
        if (torch::xpu::is_available()) {
            usedDeviceType = torch::DeviceType::XPU;
        }
#endif
#if defined(SUPPORT_CUDA_INTEROP) || defined(SUPPORT_HIP_INTEROP)
        if (torch::cuda::is_available()) {
#ifdef SUPPORT_CUDA_INTEROP
            usedDeviceType = torch::DeviceType::CUDA;
#elif defined(SUPPORT_HIP_INTEROP)
            usedDeviceType = torch::DeviceType::HIP;
#endif
        }
#endif
    }

    bool isBuildConfigurationSupported = false;
    if (usedDeviceType == torch::DeviceType::CPU)
        isBuildConfigurationSupported = true;
#ifdef SUPPORT_CUDA_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA)
        isBuildConfigurationSupported = true;
#endif
#ifdef SUPPORT_HIP_INTEROP
    if (usedDeviceType == torch::DeviceType::HIP)
        isBuildConfigurationSupported = true;
#endif
#ifdef SUPPORT_SYCL_INTEROP
    if (usedDeviceType == torch::DeviceType::XPU)
        isBuildConfigurationSupported = true;
#endif
    if (!isBuildConfigurationSupported) {
        sgl::Logfile::get()->throwError("Error in ApplicationState::ApplicationState: Unsupported device type.");
    }

    // Synchronize needs to be called to make sure cuInit has been called.
#if defined(SUPPORT_CUDA_INTEROP) || defined(SUPPORT_HIP_INTEROP)
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        torch::cuda::synchronize();
    }
#endif
#ifdef SUPPORT_SYCL_INTEROP
    if (usedDeviceType == torch::DeviceType::XPU) {
        //torch::xpu::synchronize(); //< Needs device index, which we might not have yet?
    }
#endif
    if (usedDeviceType == torch::DeviceType::CUDA) {
#ifdef SUPPORT_CUDA_INTEROP
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            bool success = sgl::vk::initializeCudaDeviceApiFunctionTable();
            if (!success) {
                usedDeviceType = torch::DeviceType::CPU;
                sgl::Logfile::get()->writeError(
                        "Error in ApplicationState::ApplicationState: "
                        "sgl::vk::initializeCudaDeviceApiFunctionTable() failed. Switching to CPU.", false);
            } else {
                deviceUuid = new uint8_t[VK_UUID_SIZE];
                CUuuid cudaDeviceUuid = {};
                sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetUuid(
                        &cudaDeviceUuid, at::cuda::getCurrentCUDAStream().device_index()), "Error in cuDeviceGetUuid: ");
                memcpy(deviceUuid, cudaDeviceUuid.bytes, VK_UUID_SIZE);
            }
        }
#else
        sgl::Logfile::get()->throwError(
                        "Error in ApplicationState::ApplicationState: DeviceType::CUDA is not supported.", false);
#endif
    }
    if (usedDeviceType == torch::DeviceType::HIP) {
#ifdef SUPPORT_HIP_INTEROP
        if (!sgl::vk::getIsHipDeviceApiFunctionTableInitialized()) {
            bool success = sgl::vk::initializeHipDeviceApiFunctionTable();
            if (!success) {
                usedDeviceType = torch::DeviceType::CPU;
                sgl::Logfile::get()->writeError(
                        "Error in ApplicationState::ApplicationState: "
                        "sgl::vk::initializeHipDeviceApiFunctionTable() failed. Switching to CPU.", false);
            } else {
                deviceUuid = new uint8_t[VK_UUID_SIZE];
                hipUUID hipDeviceUuid = {};
                sgl::vk::checkHipResult(sgl::vk::g_hipDeviceApiFunctionTable.hipDeviceGetUuid(
                        &hipDeviceUuid, at::hip::getCurrentHIPStream().device_index()), "Error in hipDeviceGetUuid: ");
                memcpy(deviceUuid, hipDeviceUuid.bytes, VK_UUID_SIZE);
            }
        }
#else
        sgl::Logfile::get()->throwError(
                "Error in ApplicationState::ApplicationState: DeviceType::HIP is not supported.", false);
#endif
    }
    if (usedDeviceType == torch::DeviceType::XPU) {
#ifdef SUPPORT_SYCL_INTEROP
        if (!sgl::vk::getIsLevelZeroFunctionTableInitialized()) {
            bool success = sgl::vk::initializeLevelZeroFunctionTable();
            if (!success) {
                usedDeviceType = torch::DeviceType::CPU;
                sgl::Logfile::get()->writeError(
                        "Error in ApplicationState::ApplicationState: "
                        "sgl::vk::initializeLevelZeroFunctionTable() failed. Switching to CPU.", false);
            } else {
                auto& syclQueue = at::xpu::getCurrentXPUStream().queue();
                sgl::vk::setLevelZeroGlobalStateFromSyclQueue(syclQueue);
                auto zeDeviceProperties = sgl::vk::retrieveZeDevicePropertiesFromSyclQueue(syclQueue);
                deviceUuid = new uint8_t[VK_UUID_SIZE];
                memcpy(deviceUuid, zeDeviceProperties.uuid.id, VK_UUID_SIZE);
            }
        }
#else
        sgl::Logfile::get()->throwError(
                "Error in ApplicationState::ApplicationState: DeviceType::XPU is not supported.", false);
#endif
    }


    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallbackHeadless);
    device = new sgl::vk::Device;

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    auto physicalDeviceCheckCallback = [&](
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceProperties physicalDeviceProperties,
            std::vector<const char*>& requiredDeviceExtensions,
            std::vector<const char*>& optionalDeviceExtensions,
            sgl::vk::DeviceFeatures& requestedDeviceFeatures) {
        if (physicalDeviceProperties.apiVersion < VK_API_VERSION_1_1) {
            return false;
        }

        VkPhysicalDeviceSubgroupProperties subgroupProperties{};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceIDProperties physicalDeviceIdProperties{};
        physicalDeviceIdProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        physicalDeviceIdProperties.pNext = &subgroupProperties;
        VkPhysicalDeviceProperties2 deviceProperties2 = {};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &physicalDeviceIdProperties;
        sgl::vk::getPhysicalDeviceProperties2(physicalDevice, deviceProperties2);

        if (deviceUuid) {
            bool isSameUuid = true;
            for (int i = 0; i < int(VK_UUID_SIZE); i++) {
                if (physicalDeviceIdProperties.deviceUUID[i] != deviceUuid[i]) {
                    isSameUuid = false;
                    break;
                }
            }
            if (!isSameUuid) {
                return false;
            }
        }

        auto* target = radix_sort_vk_target_auto_detect(&physicalDeviceProperties, &subgroupProperties, 2u);
        if (!target) {
            return false;
        }
        VkPhysicalDeviceFeatures physicalDeviceFeatures{};
        VkPhysicalDeviceVulkan11Features physicalDeviceVulkan11Features{};
        physicalDeviceVulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features{};
        physicalDeviceVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        radix_sort_vk_target_requirements_t targetRequirements{};
        targetRequirements.pdf = &physicalDeviceFeatures;
        targetRequirements.pdf11 = &physicalDeviceVulkan11Features;
        targetRequirements.pdf12 = &physicalDeviceVulkan12Features;
        radix_sort_vk_target_get_requirements(target, &targetRequirements);
        if (targetRequirements.ext_name_count > 0) {
            targetRequirements.ext_names = new const char*[targetRequirements.ext_name_count];
        }
        if (!radix_sort_vk_target_get_requirements(target, &targetRequirements)) {
            free(target);
            return false;
        }

        for (uint32_t i = 0; i < targetRequirements.ext_name_count; i++) {
            optionalDeviceExtensions.push_back(targetRequirements.ext_names[i]);
        }
        if (targetRequirements.pdf) {
            sgl::vk::mergePhysicalDeviceFeatures(
                    requestedDeviceFeatures.optionalPhysicalDeviceFeatures,
                    *targetRequirements.pdf);
        }
        if (targetRequirements.pdf11) {
            sgl::vk::mergePhysicalDeviceFeatures11(
                    requestedDeviceFeatures.optionalVulkan11Features,
                    *targetRequirements.pdf11);
        }
        if (targetRequirements.pdf12) {
            sgl::vk::mergePhysicalDeviceFeatures12(
                    requestedDeviceFeatures.optionalVulkan12Features,
                    *targetRequirements.pdf12);
        }

        if (targetRequirements.ext_name_count > 0) {
            delete[] targetRequirements.ext_names;
        }

        // target.c, radix_sort_vk_target_auto_detect:
        // radix_sort_vk_target_t* target_ptr = MALLOC_MACRO(sizeof(radix_sort_vk_target_t));
        free(target);
        return true;
    };
    device->setPhysicalDeviceCheckCallback(physicalDeviceCheckCallback);
#endif

    device->createDeviceHeadless(
            instance, {
                    VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
            },
            optionalDeviceExtensions, requestedDeviceFeatures);
    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->setUseMatrixBlock(false);
    sgl::AppSettings::get()->initializeSubsystems();
    checkIsFuchsiaRadixSortSupported(device);
    if (deviceUuid) {
        delete[] deviceUuid;
        deviceUuid = nullptr;
    }

    renderer = new sgl::vk::Renderer(sgl::AppSettings::get()->getPrimaryDevice());
    camera = std::make_shared<sgl::Camera>();
    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(100.0f);
    camera->setFOVy(std::atan(1.0f / 2.0f) * 2.0f);
    camera->setOrientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    camera->setPosition(glm::vec3(0.0f, 0.0f, 0.8f));
    transferFunctionWindow = new sgl::TransferFunctionWindow;
    // TODO: Callbacks
    optimizer = new TetMeshOptimizer(
            renderer, [](const TetMeshPtr&, float) {},
            false, []() -> std::string { return ""; },
            transferFunctionWindow);

    uint32_t maxNumCommandBuffers = 1;
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        renderReadySemaphore = std::make_shared<sgl::vk::SemaphoreVkComputeApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
        renderFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkComputeApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
        maxNumCommandBuffers = 30;
    }
#endif
    sgl::vk::CommandPoolType commandPoolType;
    commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    for (uint32_t frameIdx = 0; frameIdx < maxNumCommandBuffers; frameIdx++) {
        commandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
        fences.push_back(std::make_shared<sgl::vk::Fence>(
                device, usedDeviceType == torch::DeviceType::CPU ? 0 : VK_FENCE_CREATE_SIGNALED_BIT));
    }
}

ApplicationState::~ApplicationState() {
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        torch::cuda::synchronize();
    }
#endif
    device->waitIdle();
    commandBuffers = {};
    fences = {};
    commandBuffer = {};
    fence = {};
#ifdef SUPPORT_COMPUTE_INTEROP
    renderReadySemaphore = {};
    renderFinishedSemaphore = {};
#endif

    delete optimizer;
    delete transferFunctionWindow;
    delete renderer;
#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::freeCudaDeviceApiFunctionTable();
    }
#endif
#ifdef SUPPORT_HIP_INTEROP
    if (sgl::vk::getIsHipDeviceApiFunctionTableInitialized()) {
        sgl::vk::freeHipDeviceApiFunctionTable();
    }
#endif
#ifdef SUPPORT_SYCL_INTEROP
    if (sgl::vk::getIsLevelZeroFunctionTableInitialized()) {
        sgl::vk::freeLevelZeroFunctionTable();
    }
#endif
    sgl::AppSettings::get()->release();
}

void ApplicationState::vulkanBegin() {
    timelineValue++;
    selectNextCommandBuffer();
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        fence->wait();
        fence->reset();
    }
#endif
    commandBuffer->setFence(fence);
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        sgl::vk::StreamWrapper stream{};
#ifdef SUPPORT_CUDA_INTEROP
        if (usedDeviceType == torch::DeviceType::CUDA) {
            stream.cuStream = at::cuda::getCurrentCUDAStream();
        }
#endif
#ifdef SUPPORT_HIP_INTEROP
        if (usedDeviceType == torch::DeviceType::HIP) {
            stream.hipStream = at::hip::getCurrentHIPStream();
        }
#endif
#ifdef SUPPORT_SYCL_INTEROP
        if (usedDeviceType == torch::DeviceType::XPU) {
            auto& syclQueue = at::xpu::getCurrentXPUStream().queue();
            sgl::vk::setLevelZeroGlobalStateFromSyclQueue(syclQueue);
            stream.zeCommandList = sgl::vk::syclStreamToZeCommandList(syclQueue);
        }
#endif
        renderReadySemaphore->signalSemaphoreComputeApi(stream, timelineValue);
        renderReadySemaphore->setWaitSemaphoreValue(timelineValue);
        commandBuffer->pushWaitSemaphore(renderReadySemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
#endif
    renderer->pushCommandBuffer(commandBuffer);
    renderer->beginCommandBuffer();
}

void ApplicationState::vulkanFinished() {
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        renderFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
        commandBuffer->pushSignalSemaphore(renderFinishedSemaphore);
    }
#endif
    renderer->endCommandBuffer();
    renderer->submitToQueue();
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP) {
        sgl::vk::StreamWrapper stream{};
#ifdef SUPPORT_CUDA_INTEROP
        if (usedDeviceType == torch::DeviceType::CUDA) {
            stream.cuStream = at::cuda::getCurrentCUDAStream();
        }
#endif
#ifdef SUPPORT_HIP_INTEROP
        if (usedDeviceType == torch::DeviceType::HIP) {
            stream.hipStream = at::hip::getCurrentHIPStream();
        }
#endif
#ifdef SUPPORT_SYCL_INTEROP
        if (usedDeviceType == torch::DeviceType::XPU) {
            auto& syclQueue = at::xpu::getCurrentXPUStream().queue();
            sgl::vk::setLevelZeroGlobalStateFromSyclQueue(syclQueue);
            stream.zeCommandList = sgl::vk::syclStreamToZeCommandList(syclQueue);
        }
#endif
        renderFinishedSemaphore->waitSemaphoreComputeApi(stream, timelineValue);
    }
#endif

    if (usedDeviceType == torch::DeviceType::CPU) {
        fence->wait();
        fence->reset();
    }
}

void ApplicationState::selectNextCommandBuffer() {
    commandBuffer = commandBuffers.at(commandBufferIdx);
    fence = fences.at(commandBufferIdx);
#ifdef SUPPORT_COMPUTE_INTEROP
    if (usedDeviceType == torch::DeviceType::CUDA || usedDeviceType == torch::DeviceType::HIP)
#endif
        commandBufferIdx = (commandBufferIdx + 1) % commandBuffers.size();
}

/// Sets the used device type. May only be called at the beginning of the program.
void setUsedDeviceType(const std::string& deviceTypeName /* torch::DeviceType deviceType */) {
    if (sState) {
        throw std::runtime_error(
                "Error in setUsedDevice: This function must be called before the application state has been created.");
    }

    torch::DeviceType deviceType = torch::DeviceType::CPU;
    std::string deviceTypeNameLower = sgl::toLowerCopy(deviceTypeName);
    if (deviceTypeNameLower == torch::DeviceTypeName(torch::DeviceType::CPU, true)) {
        deviceType = torch::DeviceType::CPU;
    } else if (deviceTypeNameLower == torch::DeviceTypeName(torch::DeviceType::CUDA, true)) {
#ifdef SUPPORT_CUDA_INTEROP
        deviceType = torch::DeviceType::CUDA;
#elif defined(SUPPORT_HIP_INTEROP)
        deviceType = torch::DeviceType::HIP;
#endif
    } else if (deviceTypeNameLower == torch::DeviceTypeName(torch::DeviceType::HIP, true)) {
        deviceType = torch::DeviceType::HIP;
    } else if (deviceTypeNameLower == torch::DeviceTypeName(torch::DeviceType::XPU, true)) {
        deviceType = torch::DeviceType::XPU;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in setUsedDeviceType: Unsupported device type '" + deviceTypeName + "'.");
    }

    globalDeviceTypeSetManually = true;
    globalDeviceType = deviceType;
}

static void ensureStateExists() {
    if (!sState) {
        sState = new ApplicationState;
    }
}

void difftetvrCleanup() {
    if (sState) {
        delete sState;
        sState = nullptr;
    }
}

torch::Tensor forward(torch::Tensor X) {
    return X;
}

class TetRegularizer {
public:
    TetRegularizer(sgl::vk::Renderer* renderer, const TetMeshPtr& tetMesh, float lambda, float softplusBeta)
            : tetMeshOpt(tetMesh) {
        tetRegularizerPass = std::make_shared<TetRegularizerPass>(renderer);
        tetRegularizerPass->setSettings(lambda, softplusBeta);
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    void computeGrad() {
        auto cellIndicesBuffer = tetMeshOpt->getCellIndicesBuffer();
        auto vertexPositionBuffer = tetMeshOpt->getVertexPositionBuffer();
        auto vertexPositionGradientBuffer = tetMeshOpt->getVertexPositionGradientBuffer();
        if (cellIndicesBuffer != cellIndicesBufferPrev.lock() || vertexPositionBuffer != vertexPositionBufferPrev.lock()
                || vertexPositionGradientBuffer != vertexPositionGradientBufferPrev.lock()) {
            cellIndicesBufferPrev = cellIndicesBuffer;
            vertexPositionBufferPrev = vertexPositionBuffer;
            vertexPositionGradientBufferPrev = vertexPositionGradientBuffer;
            tetRegularizerPass->setBuffers(cellIndicesBuffer, vertexPositionBuffer, vertexPositionGradientBuffer);
        }
        tetRegularizerPass->render();
    }

private:
    TetMeshPtr tetMeshOpt;
    std::weak_ptr<sgl::vk::Buffer> cellIndicesBufferPrev;
    std::weak_ptr<sgl::vk::Buffer> vertexPositionBufferPrev;
    std::weak_ptr<sgl::vk::Buffer> vertexPositionGradientBufferPrev;
    std::shared_ptr<TetRegularizerPass> tetRegularizerPass;
};

PYBIND11_MODULE(difftetvr, m) {
    // The code block below can help to avoid pybind11 exception handling when using a debugger like gdb.
    // See: https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
    /*py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::out_of_range &e) {
            std::terminate();
        }
    });*/

    py::class_<glm::vec3>(m, "vec3")
            .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
            .def_readwrite("x", &glm::vec3::x)
            .def_readwrite("y", &glm::vec3::y)
            .def_readwrite("z", &glm::vec3::z);

    py::class_<glm::vec4>(m, "vec4")
            .def(py::init<float, float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"))
            .def_readwrite("x", &glm::vec4::x)
            .def_readwrite("y", &glm::vec4::y)
            .def_readwrite("z", &glm::vec4::z)
            .def_readwrite("w", &glm::vec4::w);

    py::class_<glm::uvec3>(m, "uvec3")
            .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("x"), py::arg("y"), py::arg("z"))
            .def_readwrite("x", &glm::uvec3::x)
            .def_readwrite("y", &glm::uvec3::y)
            .def_readwrite("z", &glm::uvec3::z);

    py::class_<sgl::AABB3>(m, "AABB3")
            .def(py::init<>())
            .def(py::init<glm::vec3, glm::vec3>(), py::arg("min"), py::arg("max"))
            .def_readwrite("min", &sgl::AABB3::min)
            .def_readwrite("max", &sgl::AABB3::max)
            .def("get_dimensions", &sgl::AABB3::getDimensions)
            .def("get_extent", &sgl::AABB3::getExtent)
            .def("get_center", &sgl::AABB3::getCenter)
            .def("get_minimum", &sgl::AABB3::getMinimum)
            .def("get_maximum", &sgl::AABB3::getMaximum);

    py::class_<FTetWildParams>(m, "FTetWildParams", "https://github.com/wildmeshing/fTetWild?tab=readme-ov-file#command-line-switches")
            .def(py::init<>([]() {
                return FTetWildParams{};
            }))
            .def_readwrite("relative_ideal_edge_length", &FTetWildParams::relativeIdealEdgeLength, "-l")
            .def_readwrite("epsilon", &FTetWildParams::epsilon, "-e")
            .def_readwrite("skip_simplify", &FTetWildParams::skipSimplify, "--skip-simlify")
            .def_readwrite("coarsen", &FTetWildParams::coarsen, "--coarsen");

    py::class_<TetGenParams>(m, "TetGenParams", "https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual005.html")
            .def(py::init<>([]() {
                return TetGenParams{};
            }))
            .def_readwrite("use_steiner_points", &TetGenParams::useSteinerPoints, "-q; to remove badly-shaped tetrahedra")
            .def_readwrite("use_radius_edge_ratio_bound", &TetGenParams::useRadiusEdgeRatioBound, "")
            .def_readwrite("radius_edge_ratio_bound", &TetGenParams::radiusEdgeRatioBound, "radius-edge ratio bound")
            .def_readwrite("use_maximum_volume_constraint", &TetGenParams::useMaximumVolumeConstraint, "-a")
            .def_readwrite("maximum_tetrahedron_volume", &TetGenParams::maximumTetrahedronVolume, "")
            .def_readwrite("coarsen", &TetGenParams::coarsen, "-R")
            .def_readwrite("maximum_dihedral_angle", &TetGenParams::maximumDihedralAngle, "-o/")
            .def_readwrite("mesh_optimization_level", &TetGenParams::meshOptimizationLevel, "Between 0 and 10.")
            .def_readwrite("use_edge_and_face_flips", &TetGenParams::useEdgeAndFaceFlips, "")
            .def_readwrite("use_vertex_smoothing", &TetGenParams::useVertexSmoothing, "")
            .def_readwrite("use_vertex_insertion_and_deletion", &TetGenParams::useVertexInsertionAndDeletion, "");

    py::enum_<TestCase>(m, "TestCase", py::arithmetic(), "")
            .value("SINGLE_TETRAHEDRON", TestCase::SINGLE_TETRAHEDRON, "")
            .value("CUBE_CENTRAL_GRADIENT", TestCase::CUBE_CENTRAL_GRADIENT, "");
    py::enum_<SplitGradientType>(m, "SplitGradientType")
            .value("POSITION", SplitGradientType::POSITION)
            .value("COLOR", SplitGradientType::COLOR)
            .value("ABS_POSITION", SplitGradientType::ABS_POSITION)
            .value("ABS_COLOR", SplitGradientType::ABS_COLOR);
    py::enum_<TetMeshingApp>(m, "TetMeshingApp")
            .value("FTETWILD", TetMeshingApp::FTETWILD)
            .value("TETGEN", TetMeshingApp::TETGEN);
    py::class_<TetMesh, TetMeshPtr>(m, "TetMesh")
            .def(py::init([]() {
                ensureStateExists();
                auto tetMesh = std::make_shared<TetMesh>(sState->device, sState->transferFunctionWindow);
#ifdef SUPPORT_COMPUTE_INTEROP
                tetMesh->setUseComputeInterop(sState->usedDeviceType != torch::DeviceType::CPU);
#endif
                tetMesh->setUsedDeviceType(sState->usedDeviceType);
                return tetMesh;
            }))
            .def("set_use_gradients", &TetMesh::setUseGradients, py::arg("use_gradients") = true)
            .def("load_test_data", &TetMesh::loadTestData, py::arg("test_case"))
            .def("load_from_file", &TetMesh::loadFromFile, py::arg("file_path"))
            .def("save_to_file", &TetMesh::saveToFile, py::arg("file_path"))
            .def("set_triangle_mesh_data", &TetMesh::setTriangleMeshData, py::arg("triangle_indices"), py::arg("vertex_positions"), py::arg("vertex_colors"))
            .def("get_bounding_box", &TetMesh::getBoundingBox)
            .def("set_vertices_changed", &TetMesh::setVerticesChangedOnDevice, py::arg("vertices_changed") = true)
            .def("on_zero_grad", &TetMesh::onZeroGrad,
                 "Necessary for CPU device to propagate zeros to Vulkan buffers. No-op for other devices.")
            .def("set_force_use_ovm_representation", &TetMesh::setForceUseOvmRepresentation, "Coarse to fine strategy.")
            .def("set_hex_mesh_const", &TetMesh::setHexMeshConst,
                 py::arg("aabb"), py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("const_color"),
                 "Initialize with tetrahedralized hex mesh with constant color.")
            .def("set_tetrahedralized_grid_ftetwild", &TetMesh::setTetrahedralizedGridFTetWild,
                 py::arg("aabb"), py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("const_color"),
                 py::arg("params"),
                 "Initialize with constant color tet mesh tetrahedralized from a grid using fTetWild.")
            .def("set_tetrahedralized_grid_tetgen", &TetMesh::setTetrahedralizedGridTetGen,
                 py::arg("aabb"), py::arg("xs"), py::arg("ys"), py::arg("zs"), py::arg("const_color"),
                 py::arg("params"),
                 "Initialize with constant color tet mesh tetrahedralized from a grid using TetGen.")
            .def("get_num_cells", &TetMesh::getNumCells)
            .def("get_num_vertices", &TetMesh::getNumVertices)
            .def("get_vertex_positions", &TetMesh::getVertexPositionTensor)
            .def("get_vertex_colors", &TetMesh::getVertexColorTensor)
            .def("get_vertex_boundary_bit_tensor", &TetMesh::getVertexBoundaryBitTensor)
            .def("check_is_any_tet_degenerate", &TetMesh::checkIsAnyTetDegenerate,
                 "Returns whether any tetrahedral element is degenerate (i.e., has a volume <= 0).")
            .def("unlink_tets", &TetMesh::unlinkTets,
                 "Removes the links between all tets, i.e., a potentially used shared index representation is reversed.")
            .def("split_by_largest_gradient_magnitudes", [](
                    const TetMeshPtr& self, const std::shared_ptr<TetMeshVolumeRenderer>& volumeRenderer,
                    SplitGradientType splitGradientType, float splitsRatio) {
                sState->vulkanBegin();
#ifdef SUPPORT_COMPUTE_INTEROP
                if (sState->usedDeviceType == torch::DeviceType::CUDA || sState->usedDeviceType == torch::DeviceType::HIP) {
                    torch::cuda::synchronize();
                }
#endif
                self->splitByLargestGradientMagnitudes(sState->renderer, splitGradientType, splitsRatio);
                volumeRenderer->setTetMeshData(self);
                sState->vulkanFinished();
            }, py::arg("volume_renderer"), py::arg("split_gradient_type"), py::arg("splits_ratio"));

    py::enum_<RendererType>(m, "RendererType")
            .value("PPLL", RendererType::PPLL)
            .value("PROJECTION", RendererType::PROJECTION)
            .value("INTERSECTION", RendererType::INTERSECTION);
    py::class_<TetMeshVolumeRenderer, std::shared_ptr<TetMeshVolumeRenderer>>(m, "Renderer")
            .def(py::init([](RendererType rendererType) {
                ensureStateExists();
                TetMeshVolumeRenderer* volumeRenderer = nullptr;
                if (rendererType == RendererType::PPLL) {
                    volumeRenderer = new TetMeshRendererPPLL(
                            sState->renderer, &sState->camera, sState->transferFunctionWindow);
                } else if (rendererType == RendererType::PROJECTION) {
                    volumeRenderer = new TetMeshRendererProjection(
                            sState->renderer, &sState->camera, sState->transferFunctionWindow);
                } else if (rendererType == RendererType::INTERSECTION) {
                    volumeRenderer = new TetMeshRendererIntersection(
                            sState->renderer, &sState->camera, sState->transferFunctionWindow);
                } else {
                    return std::shared_ptr<TetMeshVolumeRenderer>();
                }
                if (volumeRenderer) {
#ifdef SUPPORT_COMPUTE_INTEROP
                    volumeRenderer->setUseComputeInterop(sState->usedDeviceType != torch::DeviceType::CPU);
#endif
                    volumeRenderer->setUsedDeviceType(sState->usedDeviceType);
                }
                return std::shared_ptr<TetMeshVolumeRenderer>(volumeRenderer);
            }), py::arg("renderer_type") = RendererType::PPLL)
            .def("get_renderer_type", &TetMeshVolumeRenderer::getRendererType)
            .def("set_tet_mesh", &TetMeshVolumeRenderer::setTetMeshData, py::arg("tet_mesh"))
            .def("get_tet_mesh", &TetMeshVolumeRenderer::getTetMeshData)
            .def("get_attenuation", &TetMeshVolumeRenderer::getAttenuationCoefficient)
            .def("set_attenuation", &TetMeshVolumeRenderer::setAttenuationCoefficient, py::arg("attenuation_coefficient"))
            .def("set_coarse_to_fine_target_num_tets", &TetMeshVolumeRenderer::setCoarseToFineTargetNumTets, py::arg("target_num_tets"))
            .def("set_use_early_ray_termination", &TetMeshVolumeRenderer::setUseEarlyRayTermination, py::arg("use_early_ray_termination"))
            .def("set_early_ray_out_thresh", &TetMeshVolumeRenderer::setEarlyRayOutThresh, py::arg("threshold"))
            .def("set_early_ray_out_alpha", &TetMeshVolumeRenderer::setEarlyRayOutAlpha, py::arg("alpha"))
            .def("set_clear_color", [](const std::shared_ptr<TetMeshVolumeRenderer>& self, const glm::vec4& color) {
                self->setClearColor(sgl::colorFromVec4(color));
            }, py::arg("color"))
            .def("set_viewport_size", [](
                    const std::shared_ptr<TetMeshVolumeRenderer>& self, uint32_t imageWidth, uint32_t imageHeight,
                    bool recreateSwapchain) {
                auto camera = self->getCamera();
                auto renderTarget = std::make_shared<sgl::RenderTarget>(int(imageWidth), int(imageHeight));
                camera->setRenderTarget(renderTarget, false);
                camera->onResolutionChanged({});
                sState->vulkanBegin();
                self->setViewportSize(imageWidth, imageHeight);
                if (recreateSwapchain) {
                    self->recreateSwapchain(imageWidth, imageHeight);
                }
                sState->vulkanFinished();
            }, py::arg("image_width"), py::arg("image_height"), py::arg("recreate_swapchain") = true)
            .def("reuse_intermediate_buffers_from", [](
                    const std::shared_ptr<TetMeshVolumeRenderer>& self,
                    const std::shared_ptr<TetMeshVolumeRenderer>& other) {
                const auto& imageSettings = self->getOutputImageView()->getImage()->getImageSettings();
                auto fragmentBufferSize = other->getFragmentBufferSize();
                auto fragmentBuffer = other->getFragmentBuffer();
                auto startOffsetBuffer = other->getStartOffsetBuffer();
                auto fragmentCounterBuffer = other->getFragmentCounterBuffer();
                self->recreateSwapchainExternal(
                        imageSettings.width, imageSettings.height, fragmentBufferSize,
                        fragmentBuffer, startOffsetBuffer, fragmentCounterBuffer);
            }, py::arg("renderer_other"))
            .def("set_camera_fovy", [](const std::shared_ptr<TetMeshVolumeRenderer>& self, float fovy) {
                auto camera = self->getCamera();
                camera->setFOVy(fovy);
            }, py::arg("fovy"))
            .def("set_view_matrix", [](
                    const std::shared_ptr<TetMeshVolumeRenderer>& self, const std::vector<float>& viewMatrixData) {
                glm::mat4 viewMatrix;
                for (int i = 0; i < 16; i++) {
                    viewMatrix[i / 4][i % 4] = viewMatrixData[i];
                }
                self->getCamera()->overwriteViewMatrix(viewMatrix);
            }, py::arg("view_matrix_array"))
            .def("render", [](const std::shared_ptr<TetMeshVolumeRenderer>& self) {
                sState->vulkanBegin();
                if (!self->getTetMesh() || self->getTetMesh()->getIsEmpty()) {
                    sgl::Logfile::get()->throwError("Missing valid tet mesh in Renderer.render.");
                }
                self->render();
                self->copyOutputImageToBuffer();
                sState->vulkanFinished();
                return self->getImageTensor();
            })
            .def("render_adjoint", [](const std::shared_ptr<TetMeshVolumeRenderer>& self, torch::Tensor imageAdjointTensor, bool useAbsGrad) {
                if (imageAdjointTensor.device().type() != sState->usedDeviceType) {
                    sgl::Logfile::get()->throwError(
                            "Error in render_adjoint: The tensors must be on the device used at initialization time.",
                            false);
                }
                if (imageAdjointTensor.sizes().size() != 3) {
                    sgl::Logfile::get()->throwError(
                            "Error in render_adjoint: imageAdjointTensor.sizes().size() != 3.", false);
                }
                if (imageAdjointTensor.size(2) != 4) {
                    sgl::Logfile::get()->throwError(
                            "Error in render_adjoint: The number of image channels is not equal to 4.",
                            false);
                }
                if (imageAdjointTensor.dtype() != torch::kFloat32) {
                    sgl::Logfile::get()->throwError(
                            "Error in render_adjoint: The only data type currently supported is 32-bit float.",
                            false);
                }
                if (!imageAdjointTensor.is_contiguous()) {
                    imageAdjointTensor = imageAdjointTensor.contiguous();
                }

                if (useAbsGrad) {
                    self->setUseAbsGrad(true);
                }
                self->copyAdjointBufferToImagePreCheck(imageAdjointTensor.data_ptr());

                sState->vulkanBegin();
                self->copyAdjointBufferToImage();
                self->renderAdjoint();
                if (sState->usedDeviceType == torch::DeviceType::CPU) {
                    self->getTetMesh()->copyGradientsToCpu(sState->renderer);
                }
                sState->vulkanFinished();
                self->setUseAbsGrad(false);
            }, py::arg("image_adjoint"), py::arg("use_abs_grad"))
            .def("set_export_linked_list_data", [](const std::shared_ptr<TetMeshVolumeRenderer>& self) {
                if (self->getRendererType() != RendererType::PPLL) {
                    sgl::Logfile::get()->throwError(
                            "Error in set_export_linked_list_data: The renderer must be of type RendererType.PPLL.",
                            false);
                }
                return self->setExportLinkedListData(true);
            })
            .def("get_fragment_buffer", [](const std::shared_ptr<TetMeshVolumeRenderer>& self) {
                if (self->getRendererType() != RendererType::PPLL) {
                    sgl::Logfile::get()->throwError(
                            "Error in get_fragment_buffer: The renderer must be of type RendererType.PPLL.", false);
                }
                return self->getFragmentBufferTensor();
            })
            .def("get_start_offset_buffer", [](const std::shared_ptr<TetMeshVolumeRenderer>& self) {
                if (self->getRendererType() != RendererType::PPLL) {
                    sgl::Logfile::get()->throwError(
                            "Error in get_start_offset_buffer: The renderer must be of type RendererType.PPLL.", false);
                }
                return self->getStartOffsetBufferTensor();
            });

    py::class_<TetRegularizer, std::shared_ptr<TetRegularizer>>(m, "TetRegularizer")
            .def(py::init([](const TetMeshPtr& tetMesh, float lambda, float softplusBeta) {
                ensureStateExists();
                sState->vulkanBegin();
                auto tetRegularizer = std::make_shared<TetRegularizer>(sState->renderer, tetMesh, lambda, softplusBeta);
                sState->vulkanFinished();
                return tetRegularizer;
            }), py::arg("tet_mesh"), py::arg("reg_lambda"), py::arg("softplus_beta"))
            .def("compute_grad", [](const std::shared_ptr<TetRegularizer>& self) {
                sState->vulkanBegin();
                self->computeGrad();
                sState->vulkanFinished();
            });

    // Can be used for logging purposes (e.g., writing gradients).
    py::class_<VtkWriter, std::shared_ptr<VtkWriter>>(m, "TetMeshVtkWriter")
            .def(py::init([](const std::string& filePath, bool isBinary) {
                auto vtkWriter = std::make_shared<VtkWriter>();
                vtkWriter->initializeWriter(filePath, isBinary);
                return vtkWriter;
            }), py::arg("file_path"), py::arg("is_binary") = true)
            .def("write_next_time_step", [](const std::shared_ptr<VtkWriter>& self, const TetMeshPtr& tetMesh) {
                sState->vulkanBegin();
                self->writeNextTimeStep(sState->renderer, tetMesh);
                sState->vulkanFinished();
            }, py::arg("tet_mesh"));

    py::class_<RegularGrid, RegularGridPtr>(m, "RegularGrid")
            .def(py::init([]() {
                ensureStateExists();
                return std::make_shared<RegularGrid>(sState->device, sState->transferFunctionWindow);
            }))
            .def("load_from_file", &RegularGrid::loadFromFile, py::arg("file_path"))
            .def("get_grid_size_x", &RegularGrid::getGridSizeX)
            .def("get_grid_size_y", &RegularGrid::getGridSizeY)
            .def("get_grid_size_z", &RegularGrid::getGridSizeZ)
            .def("get_bounding_box", &RegularGrid::getBoundingBox);

    py::enum_<RegularGridInterpolationMode>(m, "RegularGridInterpolationMode")
            .value("NEAREST", RegularGridInterpolationMode::NEAREST)
            .value("LINEAR", RegularGridInterpolationMode::LINEAR);
    py::class_<RegularGridRendererDVR, std::shared_ptr<RegularGridRendererDVR>>(m, "RegularGridRenderer")
            .def(py::init([]() {
                ensureStateExists();
                auto regularGridRenderer = std::make_shared<RegularGridRendererDVR>(
                        sState->renderer, &sState->camera, sState->transferFunctionWindow);
                if (regularGridRenderer) {
#ifdef SUPPORT_COMPUTE_INTEROP
                    regularGridRenderer->setUseComputeInterop(sState->usedDeviceType != torch::DeviceType::CPU);
#endif
                    regularGridRenderer->setUsedDeviceType(sState->usedDeviceType);
                }
                return regularGridRenderer;
            }))
            .def("set_regular_grid", &RegularGridRendererDVR::setRegularGridData, py::arg("regular_grid"))
            .def("get_regular_grid", &RegularGridRendererDVR::getRegularGridData)
            .def("get_attenuation", &RegularGridRendererDVR::getAttenuationCoefficient)
            .def("set_attenuation", &RegularGridRendererDVR::setAttenuationCoefficient, py::arg("attenuation_coefficient"))
            .def("load_transfer_function_from_file", &RegularGridRendererDVR::loadTransferFunctionFromFile, py::arg("file_path"))
            .def("set_clear_color", [](const std::shared_ptr<RegularGridRendererDVR>& self, const glm::vec4& color) {
                self->setClearColor(sgl::colorFromVec4(color));
            }, py::arg("color"))
            .def("get_step_size", &RegularGridRendererDVR::getStepSize)
            .def("set_step_size", &RegularGridRendererDVR::setStepSize, py::arg("step_size"))
            .def("set_viewport_size", [](
                    const std::shared_ptr<RegularGridRendererDVR>& self, uint32_t imageWidth, uint32_t imageHeight,
                    bool recreateSwapchain) {
                auto camera = self->getCamera();
                auto renderTarget = std::make_shared<sgl::RenderTarget>(int(imageWidth), int(imageHeight));
                camera->setRenderTarget(renderTarget, false);
                camera->onResolutionChanged({});
                sState->vulkanBegin();
                self->setViewportSize(imageWidth, imageHeight);
                if (recreateSwapchain) {
                    self->recreateSwapchain(imageWidth, imageHeight);
                }
                sState->vulkanFinished();
            }, py::arg("image_width"), py::arg("image_height"), py::arg("recreate_swapchain") = true)
            .def("set_camera_fovy", [](const std::shared_ptr<RegularGridRendererDVR>& self, float fovy) {
                auto camera = self->getCamera();
                camera->setFOVy(fovy);
            }, py::arg("fovy"))
            .def("set_view_matrix", [](
                    const std::shared_ptr<RegularGridRendererDVR>& self, const std::vector<float>& viewMatrixData) {
                glm::mat4 viewMatrix;
                for (int i = 0; i < 16; i++) {
                    viewMatrix[i / 4][i % 4] = viewMatrixData[i];
                }
                self->getCamera()->overwriteViewMatrix(viewMatrix);
            }, py::arg("view_matrix_array"))
            .def("render", [](const std::shared_ptr<RegularGridRendererDVR>& self) {
                sState->vulkanBegin();
                if (!self->getRegularGridData() || self->getRegularGridData()->getIsEmpty()) {
                    sgl::Logfile::get()->throwError("Missing valid tet mesh in Renderer.render.");
                }
                self->render();
                self->copyOutputImageToBuffer();
                sState->vulkanFinished();
                return self->getImageTensor();
            });

    py::enum_<OptimizerType>(m, "OptimizerType")
            .value("SGD", OptimizerType::SGD)
            .value("ADAM", OptimizerType::ADAM);
    py::enum_<LossType>(m, "LossType")
            .value("L1", LossType::L1)
            .value("L2", LossType::L2);
    py::class_<OptimizerSettings>(m, "OptimizerSettings")
            .def(py::init<>())
            .def_readwrite("learning_rate", &OptimizerSettings::learningRate)
            .def_readwrite("lr_decay_rate", &OptimizerSettings::lrDecayRate)
            .def_readwrite("beta1", &OptimizerSettings::beta1)
            .def_readwrite("beta2", &OptimizerSettings::beta2)
            .def_readwrite("epsilon", &OptimizerSettings::epsilon);
    py::class_<TetRegularizerSettings>(m, "TetRegularizerSettings")
            .def(py::init<>())
            .def_readwrite("lambda", &TetRegularizerSettings::lambda)
            .def_readwrite("beta", &TetRegularizerSettings::beta);
    py::enum_<InitGridType>(m, "InitGridType")
            .value("DECOMPOSED_HEX_MESH", InitGridType::DECOMPOSED_HEX_MESH)
            .value("MESHING_FTETWILD", InitGridType::MESHING_FTETWILD)
            .value("MESHING_TETGEN", InitGridType::MESHING_TETGEN);
    py::class_<OptimizationSettings>(m, "OptimizationSettings")
            .def(py::init<>())
            .def_readwrite("optimizer_type", &OptimizationSettings::optimizerType)
            .def_readwrite("loss_type", &OptimizationSettings::lossType)
            .def_readwrite("optimize_positions", &OptimizationSettings::optimizePositions)
            .def_readwrite("optimize_colors", &OptimizationSettings::optimizeColors)
            .def_readwrite("optimizer_settings_positions", &OptimizationSettings::optimizerSettingsPositions)
            .def_readwrite("optimizer_settings_colors", &OptimizationSettings::optimizerSettingsColors)
            .def_readwrite("tet_regularizer_settings", &OptimizationSettings::tetRegularizerSettings)
            .def_readwrite("max_num_epochs", &OptimizationSettings::maxNumEpochs)
            .def_readwrite("fix_boundary", &OptimizationSettings::fixBoundary)
            .def_readwrite("image_width", &OptimizationSettings::imageWidth)
            .def_readwrite("image_height", &OptimizationSettings::imageHeight)
            .def_readwrite("attenuation_coefficient", &OptimizationSettings::attenuationCoefficient)
            .def_readwrite("sample_random_view", &OptimizationSettings::sampleRandomView)
            .def_readwrite("use_early_ray_termination", &OptimizationSettings::useEarlyRayTermination)
            .def_readwrite("early_ray_out_thresh", &OptimizationSettings::earlyRayOutThresh)
            .def_readwrite("data_set_file_name_gt", &OptimizationSettings::dataSetFileNameGT, "Selected file name.")
            .def_readwrite("data_set_file_name_opt", &OptimizationSettings::dataSetFileNameOpt, "Selected file name.")
            .def_readwrite("use_coarse_to_fine", &OptimizationSettings::useCoarseToFine)
            .def_readwrite("use_constant_init_grid", &OptimizationSettings::useConstantInitGrid)
            .def_readwrite("init_grid_type", &OptimizationSettings::initGridType)
            .def_readwrite("init_grid_resolution", &OptimizationSettings::initGridResolution)
            .def_readwrite("ftetwild_params", &OptimizationSettings::fTetWildParams)
            .def_readwrite("tetgen_params", &OptimizationSettings::tetGenParams)
            .def_readwrite("max_num_tets", &OptimizationSettings::maxNumTets)
            .def_readwrite("num_splits_ratio", &OptimizationSettings::numSplitsRatio)
            .def_readwrite("split_gradient_type", &OptimizationSettings::splitGradientType)
            .def_readwrite("export_position_gradients", &OptimizationSettings::exportPositionGradients)
            .def_readwrite("export_file_name_gradient_field", &OptimizationSettings::exportFileNameGradientField)
            .def_readwrite("is_binary_vtk", &OptimizationSettings::isBinaryVtk);

    py::class_<CameraSettings>(m, "CameraSettings")
            .def(py::init<>())
            .def("set_intrinsics", &CameraSettings::setIntrinsics, py::arg("img_width"), py::arg("img_height"), py::arg("fovy"), py::arg("near"), py::arg("far"))
            .def("set_view_matrix", [](CameraSettings& self, const std::vector<float>& viewMatrixData) {
                glm::mat4 viewMatrix;
                for (int i = 0; i < 16; i++) {
                    viewMatrix[i / 4][i % 4] = viewMatrixData[i];
                }
                self.setViewMatrix(viewMatrix);
            }, py::arg("view_matrix_array"));
    py::enum_<VoxelCarvingType>(m, "VoxelCarvingType")
            .value("DENSE_CPU", VoxelCarvingType::DENSE_CPU);
    py::class_<VoxelCarving, std::shared_ptr<VoxelCarving>>(m, "VoxelCarving")
            .def(py::init([](const sgl::AABB3& gridBoundingBox, const glm::uvec3& gridResolution, VoxelCarvingType voxelCarvingType) {
                ensureStateExists();
                return std::make_shared<DenseVoxelCarvingCPU>(gridBoundingBox, gridResolution);
            }), py::arg("grid_bounding_box"), py::arg("grid_resolution"), py::arg("voxel_carving_type") = VoxelCarvingType::DENSE_CPU)
            .def("process_next_frame", &VoxelCarving::processNextFrame, py::arg("input_image"), py::arg("camera_settings"))
            .def("compute_non_empty_bounding_box", &VoxelCarving::computeNonEmptyBoundingBox);

    /*py::enum_<torch::DeviceType>(m, "TorchDeviceType")
            .value("CPU", torch::DeviceType::CPU)
            .value("CUDA", torch::DeviceType::CUDA)
            .value("MKLDNN", torch::DeviceType::MKLDNN)
            .value("OPENGL", torch::DeviceType::OPENGL)
            .value("OPENCL", torch::DeviceType::OPENCL)
            .value("IDEEP", torch::DeviceType::IDEEP)
            .value("HIP", torch::DeviceType::HIP)
            .value("FPGA", torch::DeviceType::FPGA)
            .value("ORT", torch::DeviceType::ORT)
            .value("XLA", torch::DeviceType::XLA)
            .value("VULKAN", torch::DeviceType::Vulkan)
            .value("METAL", torch::DeviceType::Metal)
            .value("XPU", torch::DeviceType::XPU);*/
    m.def("set_device_type", setUsedDeviceType,
        "Sets the used device type. May only be called at the beginning of the program.",
        py::arg("device_type"));
    m.add_object("_cleanup", py::capsule(difftetvrCleanup));
}
