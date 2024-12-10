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

#ifndef DIFFTETVR_REGULARGRID_HPP
#define DIFFTETVR_REGULARGRID_HPP

#include <vector>
#include <string>
#include <functional>
#include <memory>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <Math/Geometry/AABB3.hpp>
#include <Utils/SciVis/ScalarDataFormat.hpp>

#include "RegularGridLoaders/FieldType.hpp"

namespace sgl { namespace vk {
class Device;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
}}

class HalfFloat;
class RegularGridLoader;
class HostCacheEntryType;

/**
 * For testing against rendering of regular grids using DVR, this class supports loading and storing scalar data on
 * regular grids.
 */
class RegularGrid {
public:
    explicit RegularGrid(sgl::vk::Device* device);
    bool loadFromFile(const std::string& filePath);
    [[nodiscard]] inline int getGridSizeX() const { return xs; }
    [[nodiscard]] inline int getGridSizeY() const { return ys; }
    [[nodiscard]] inline int getGridSizeZ() const { return zs; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBox() { return boundingBox; }
    const sgl::vk::ImageViewPtr& getFieldImageView() { return gridDataImageView; }
    [[nodiscard]] inline bool getIsEmpty() const { return gridDataImageView.get() == nullptr; }

    // File loaders.
    RegularGridLoader* createRegularGridLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<RegularGridLoader*()>> factoriesLoader;

    // Loader interface.
    void setGridExtent(int _xs, int _ys, int _zs, float _dx, float _dy, float _dz);
    void setNumTimeSteps(int _ts);
    void setTimeSteps(const std::vector<int>& timeSteps);
    void setTimeSteps(const std::vector<float>& timeSteps);
    void setTimeSteps(const std::vector<double>& timeSteps);
    void setTimeSteps(const std::vector<std::string>& timeSteps);
    void setEnsembleMemberCount(int _es);
    void setFieldNames(const std::unordered_map<FieldType, std::vector<std::string>>& fieldNamesMap);
    void addField(float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(uint8_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(uint16_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(HalfFloat* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(
            void* fieldData, ScalarDataFormat dataFormat, FieldType fieldType,
            const std::string& fieldName, int timeStepIdx, int ensembleIdx);

private:
    sgl::vk::Device* device;
    sgl::vk::ImageViewPtr gridDataImageView;

    /// Size in x, y, z, time and ensemble dimensions.
    int xs = 0, ys = 0, zs = 0, ts = 1, es = 1;
    int tsFileCount = 1, esFileCount = 1;
    /// Distance between two neighboring points in x/y/z/time direction.
    float dx = 0.0f, dy = 0.0f, dz = 0.0f, dt = 1.0f;
    /// Box encompassing all grid points.
    sgl::AABB3 boundingBox;

    bool transpose = false;
    glm::ivec3 transposeAxes = glm::ivec3(0, 1, 2);
    std::shared_ptr<HostCacheEntryType> hostCacheEntry;
};

typedef std::shared_ptr<RegularGrid> RegularGridPtr;

#endif //DIFFTETVR_REGULARGRID_HPP
