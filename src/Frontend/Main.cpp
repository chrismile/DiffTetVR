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

#include <Utils/StringUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/AppLogic.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Shader/ShaderManager.hpp>
#include <ImGui/imgui.h>

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
#include <radix_sort/radix_sort_vk.h>
#endif

#include "MainApp.hpp"

int main(int argc, char *argv[]) {
    // Initialize the filesystem utilities.
    sgl::FileUtils::get()->initialize("DiffTetVR", argc, argv);

#ifdef DATA_PATH
    if (!sgl::FileUtils::get()->directoryExists("Data") && !sgl::FileUtils::get()->directoryExists("../Data")) {
        sgl::AppSettings::get()->setDataDirectory(DATA_PATH);
    }
#endif
    sgl::AppSettings::get()->initializeDataDirectory();

    std::string iconPath = sgl::AppSettings::get()->getDataDirectory() + "Fonts/icon_256.png";
    sgl::AppSettings::get()->setApplicationDescription("Differentiable volume renderer for tetrahedral meshes");
    sgl::AppSettings::get()->loadApplicationIconFromFile(iconPath);

    // Load the file containing the app settings
    ImVector<ImWchar> fontRanges;
    std::string settingsFile = sgl::FileUtils::get()->getConfigDirectory() + "settings.txt";
    sgl::AppSettings::get()->loadSettings(settingsFile.c_str());
    sgl::AppSettings::get()->getSettings().addKeyValue("window-multisamples", 0);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-vSync", true);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-resizable", true);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-savePosition", true);
    //sgl::AppSettings::get()->setVulkanDebugPrintfEnabled();

    ImFontGlyphRangesBuilder builder;
    builder.AddChar(L'\u03BB'); // lambda
    builder.AddChar(L'\u03C3'); // sigma
    builder.BuildRanges(&fontRanges);
    sgl::AppSettings::get()->setLoadGUI(fontRanges.Data, true, false);

    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);

    sgl::Window* window = nullptr;
    std::vector<const char*> optionalDeviceExtensions;
    window = sgl::AppSettings::get()->createWindow();

#ifdef SUPPORT_CUDA_INTEROP
    optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif
    optionalDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    optionalDeviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    auto* device = new sgl::vk::Device;

    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    requestedDeviceFeatures.requestedPhysicalDeviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderInt64 = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.runtimeDescriptorArray = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    auto physicalDeviceCheckCallback = [](
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
        VkPhysicalDeviceProperties2 deviceProperties2 = {};
        deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProperties2.pNext = &subgroupProperties;
        sgl::vk::getPhysicalDeviceProperties2(physicalDevice, deviceProperties2);

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
            return false;
        }

        for (uint32_t i = 0; i < targetRequirements.ext_name_count; i++) {
            requiredDeviceExtensions.push_back(targetRequirements.ext_names[i]);
        }
        if (targetRequirements.pdf) {
            sgl::vk::mergePhysicalDeviceFeatures(
                    requestedDeviceFeatures.requestedPhysicalDeviceFeatures,
                    *targetRequirements.pdf);
        }
        if (targetRequirements.pdf11) {
            sgl::vk::mergePhysicalDeviceFeatures11(
                    requestedDeviceFeatures.requestedVulkan11Features,
                    *targetRequirements.pdf11);
        }
        if (targetRequirements.pdf12) {
            sgl::vk::mergePhysicalDeviceFeatures12(
                    requestedDeviceFeatures.requestedVulkan12Features,
                    *targetRequirements.pdf12);
        }

        if (targetRequirements.ext_name_count > 0) {
            delete[] targetRequirements.ext_names;
        }
        return true;
    };
    device->setPhysicalDeviceCheckCallback(physicalDeviceCheckCallback);
#endif

    device->createDeviceSwapchain(
            instance, window, {
                    VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
            },
            optionalDeviceExtensions, requestedDeviceFeatures);

    sgl::vk::Swapchain* swapchain = new sgl::vk::Swapchain(device);
    swapchain->create(window);
    sgl::AppSettings::get()->setSwapchain(swapchain);

    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->initializeSubsystems();

    auto app = new MainApp();
    app->run();
    delete app;

    sgl::AppSettings::get()->release();

    return 0;
}
