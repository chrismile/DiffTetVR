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

#ifndef DIFFTETVR_REGULARGRIDLOADER_HPP
#define DIFFTETVR_REGULARGRIDLOADER_HPP

#include <vector>
#include <string>
#include <cstdint>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "FieldType.hpp"
#include "HostCacheEntry.hpp"

class RegularGridLoader {
public:
    virtual ~RegularGridLoader() = default;
    virtual bool setInputFiles(RegularGrid* volumeData, const std::string& filePath) = 0;
    virtual bool getFieldEntry(
            RegularGrid* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) = 0;
    virtual bool getHasFloat32Data() { return true; }
    // Metadata reuse for individual time step or ensemble member files can potentially speed up loading.
    virtual bool getSupportsMetadataReuse() { return false; }
    virtual bool setMetadataFrom(RegularGridLoader* other) { return false; }
};

#endif //DIFFTETVR_REGULARGRIDLOADER_HPP
