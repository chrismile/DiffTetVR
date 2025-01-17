/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Christoph Neuhauser
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

#include <Graphics/Vulkan/Utils/Device.hpp>
#include "RadixSortHelper.hpp"

#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
#include <radix_sort/radix_sort_vk.h>
#endif

static bool isFuchsiaRadixSortSupported = false;

void checkIsFuchsiaRadixSortSupported(sgl::vk::Device* device) {
#ifdef USE_FUCHSIA_RADIX_SORT_CMAKE
    isFuchsiaRadixSortSupported = true;

    auto subgroupProperties = device->getPhysicalDeviceSubgroupProperties();
    auto physicalDeviceProperties = device->getPhysicalDeviceProperties();

    auto* target = radix_sort_vk_target_auto_detect(&physicalDeviceProperties, &subgroupProperties, 2u);
    if (!target) {
        isFuchsiaRadixSortSupported = false;
        return;
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
        isFuchsiaRadixSortSupported = false;
        free(target);
        return;
    }

    for (uint32_t i = 0; i < targetRequirements.ext_name_count; i++) {
        if (!device->isDeviceExtensionSupported(targetRequirements.ext_names[i])) {
            isFuchsiaRadixSortSupported = false;
        }
    }
    if (targetRequirements.pdf) {
        if (!device->checkPhysicalDeviceFeaturesSupported(*targetRequirements.pdf)) {
            isFuchsiaRadixSortSupported = false;
        }
    }
    if (targetRequirements.pdf11) {
        if (!device->checkPhysicalDeviceFeatures11Supported(*targetRequirements.pdf11)) {
            isFuchsiaRadixSortSupported = false;
        }
    }
    if (targetRequirements.pdf12) {
        if (!device->checkPhysicalDeviceFeatures12Supported(*targetRequirements.pdf12)) {
            isFuchsiaRadixSortSupported = false;
        }
    }

    if (targetRequirements.ext_name_count > 0) {
        delete[] targetRequirements.ext_names;
    }

    // target.c, radix_sort_vk_target_auto_detect:
    // radix_sort_vk_target_t* target_ptr = MALLOC_MACRO(sizeof(radix_sort_vk_target_t));
    free(target);
#endif
}

bool getIsFuchsiaRadixSortSupported() {
    return isFuchsiaRadixSortSupported;
}
