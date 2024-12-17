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

#include <map>
#include <cstring>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Math/half/half.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#include "RegularGridLoaders/RegularGridLoader.hpp"
#include "RegularGridLoaders/DatRawFileLoader.hpp"
#include "RegularGrid.hpp"

template <typename T>
static std::pair<std::vector<std::string>, std::function<RegularGridLoader*()>> registerRegularGridLoader() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

RegularGridLoader* RegularGrid::createRegularGridLoaderByExtension(const std::string& fileExtension) {
    auto it = factoriesLoader.find(fileExtension);
    if (it == factoriesLoader.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TetMeshData::createRegularGridLoaderByExtension: Unsupported file extension '."
                + fileExtension + "'.", true);
        return nullptr;
    } else {
        return it->second();
    }
}

RegularGrid::RegularGrid(sgl::vk::Device* device, sgl::TransferFunctionWindow* transferFunctionWindow)
        : device(device), transferFunctionWindow(transferFunctionWindow) {
    // Create the list of regular grid loaders.
    std::map<std::vector<std::string>, std::function<RegularGridLoader*()>> factoriesLoaderMap = {
            registerRegularGridLoader<DatRawFileLoader>(),
    };
    for (auto& factory : factoriesLoaderMap) {
        for (const std::string& extension : factory.first) {
            factoriesLoader.insert(std::make_pair(extension, factory.second));
        }
    }
}

bool RegularGrid::loadFromFile(const std::string& filePath) {
    std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
    RegularGridLoader* regularGridLoader = createRegularGridLoaderByExtension(fileExtension);
    if (!regularGridLoader) {
        return false;
    }
    if (!regularGridLoader->setInputFiles(this, filePath)) {
        delete regularGridLoader;
        return false;
    }
    HostCacheEntryType* fieldEntryPtr = nullptr;
    if (!regularGridLoader->getFieldEntry(this, FieldType::SCALAR, "Density", 0, 0, fieldEntryPtr)) {
        delete regularGridLoader;
        return false;
    }
    std::shared_ptr<HostCacheEntryType> fieldEntry;
    if (!fieldEntryPtr) {
        fieldEntry = hostCacheEntry;
    } else {
        fieldEntry = std::shared_ptr<HostCacheEntryType>(fieldEntryPtr);
    }
    if (fieldEntry == nullptr) {
        delete regularGridLoader;
        return false;
    }

    const float* attributesPointer = fieldEntry->getDataFloat();
    std::vector<float> attributes(attributesPointer, attributesPointer + fieldEntry->getNumEntries());
    transferFunctionWindow->computeHistogram(attributes);

    FieldType fieldType = FieldType::SCALAR; // TODO
    ScalarDataFormat dataFormat = fieldEntry->getScalarDataFormatNative();
    size_t sizeInBytes = fieldEntry->getNumEntries();
    if (fieldType == FieldType::VECTOR) {
        sizeInBytes *= 3;
    } else if (fieldType == FieldType::COLOR) {
        sizeInBytes *= 4;
    }
    if (dataFormat == ScalarDataFormat::FLOAT) {
        sizeInBytes *= sizeof(float);
    } else if (dataFormat == ScalarDataFormat::SHORT || dataFormat == ScalarDataFormat::FLOAT16) {
        sizeInBytes *= sizeof(uint16_t);
    }

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(xs);
    imageSettings.height = uint32_t(ys);
    imageSettings.depth = uint32_t(zs);
    imageSettings.imageType = VK_IMAGE_TYPE_3D;
    if (dataFormat == ScalarDataFormat::FLOAT) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    } else if (dataFormat == ScalarDataFormat::BYTE) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    } else if (dataFormat == ScalarDataFormat::SHORT) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
    } else if (dataFormat == ScalarDataFormat::FLOAT16) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageSettings.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    auto image = std::make_shared<sgl::vk::Image>(device, imageSettings);
    image->uploadData(sizeInBytes, fieldEntry->getDataNative());

    gridDataImageView = std::make_shared<sgl::vk::ImageView>(image);
    return true;
}

void RegularGrid::setGridExtent(int _xs, int _ys, int _zs, float _dx, float _dy, float _dz) {
    xs = _xs;
    ys = _ys;
    zs = _zs;
    dx = _dx;
    dy = _dy;
    dz = _dz;

    if (transpose) {
        int dimensions[3] = { xs, ys, zs };
        float spacing[3] = { dx, dy, dz };
        xs = dimensions[transposeAxes[0]];
        ys = dimensions[transposeAxes[1]];
        zs = dimensions[transposeAxes[2]];
        dx = spacing[transposeAxes[0]];
        dy = spacing[transposeAxes[1]];
        dz = spacing[transposeAxes[2]];
    }

    auto box = sgl::AABB3(
            glm::vec3(0.0f),
            glm::vec3(float(xs - 1) * dx, float(ys - 1) * dy, float(zs - 1) * dz));
    glm::vec3 dimensions = box.getDimensions();
    float maxDimension = std::max(dimensions.x, std::max(dimensions.y, dimensions.z));
    glm::vec3 normalizedDimensions = dimensions / maxDimension;
    boundingBox.min = -normalizedDimensions * 0.25f;
    boundingBox.max = normalizedDimensions * 0.25f;
}

void RegularGrid::setNumTimeSteps(int _ts) {
    ts = _ts;
}

void RegularGrid::setTimeSteps(const std::vector<int>& timeSteps) {
    ts = int(timeSteps.size());
}

void RegularGrid::setTimeSteps(const std::vector<float>& timeSteps) {
    ts = int(timeSteps.size());
}

void RegularGrid::setTimeSteps(const std::vector<double>& timeSteps) {
    ts = int(timeSteps.size());
}

void RegularGrid::setTimeSteps(const std::vector<std::string>& timeSteps) {
    ts = int(timeSteps.size());
}

void RegularGrid::setEnsembleMemberCount(int _es) {
    es = _es;
}

void RegularGrid::setFieldNames(const std::unordered_map<FieldType, std::vector<std::string>>& fieldNamesMap) {
    /*if (separateFilesPerAttribute) {
        typeToFieldNamesMap[FieldType::SCALAR] = dataSetInformation.attributeNames;
        typeToFieldNamesMapBase[FieldType::SCALAR] = dataSetInformation.attributeNames;
    } else {
        typeToFieldNamesMap = fieldNamesMap;
        typeToFieldNamesMapBase = fieldNamesMap;
    }*/
}

template<class T>
static void addFieldGlobal(
        T* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx,
        int xs, int ys, int zs, bool transpose, const glm::ivec3& transposeAxes,
        std::shared_ptr<HostCacheEntryType>& hostCacheEntry) {
    if (transpose) {
        if (transposeAxes != glm::ivec3(0, 2, 1)) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::addScalarField: At the moment, only transposing the "
                    "Y and Z axis is supported.");
        }
        if (fieldType == FieldType::SCALAR) {
            auto* scalarFieldCopy = new T[xs * ys * zs];
            if constexpr(std::is_same<T, HalfFloat>()) {
                size_t bufferSize = xs * ys * zs;
                for (size_t i = 0; i < bufferSize; i++) {
                    scalarFieldCopy[i] = fieldData[i];
                }
            } else {
                memcpy(scalarFieldCopy, fieldData, sizeof(T) * xs * ys * zs);
            }
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(xs, ys, zs, fieldData, scalarFieldCopy) default(none)
#endif
            for (int z = 0; z < zs; z++) {
#endif
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        int readPos = ((y)*xs*zs + (z)*xs + (x));
                        int writePos = ((z)*xs*ys + (y)*xs + (x));
                        fieldData[writePos] = scalarFieldCopy[readPos];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] scalarFieldCopy;
        } else {
            auto* vectorFieldCopy = new T[3 * xs * ys * zs];
            if constexpr(std::is_same<T, HalfFloat>()) {
                size_t bufferSize = 3 * xs * ys * zs;
                for (size_t i = 0; i < bufferSize; i++) {
                    vectorFieldCopy[i] = fieldData[i];
                }
            } else {
                memcpy(vectorFieldCopy, fieldData, sizeof(T) * 3 * xs * ys * zs);
            }
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(xs, ys, zs, fieldData, vectorFieldCopy) default(none)
#endif
            for (int z = 0; z < zs; z++) {
#endif
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        int readPos = ((y)*xs*zs*3 + (z)*xs*3 + (x)*3);
                        int writePos = ((z)*xs*ys*3 + (y)*xs*3 + (x)*3);
                        fieldData[writePos] = vectorFieldCopy[readPos];
                        fieldData[writePos + 1] = vectorFieldCopy[readPos + 2];
                        fieldData[writePos + 2] = vectorFieldCopy[readPos + 1];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] vectorFieldCopy;
        }
    }

    size_t numEntries = 0;
    if (fieldType == FieldType::SCALAR) {
        numEntries = xs * ys * zs;
    } else if (fieldType == FieldType::VECTOR) {
        numEntries = 3 * size_t(xs * ys * zs);
    } else {
        sgl::Logfile::get()->throwError("Error in VolumeData::addField: Invalid field type.");
    }
    hostCacheEntry = std::make_shared<HostCacheEntryType>(numEntries, fieldData);
}

void RegularGrid::addField(
        float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, transpose, transposeAxes, hostCacheEntry);
}

void RegularGrid::addField(
        uint8_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, transpose, transposeAxes, hostCacheEntry);
}

void RegularGrid::addField(
        uint16_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, transpose, transposeAxes, hostCacheEntry);
}

void RegularGrid::addField(
        HalfFloat* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, transpose, transposeAxes, hostCacheEntry);
}

void RegularGrid::addField(
        void* fieldData, ScalarDataFormat dataFormat, FieldType fieldType,
        const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    if (dataFormat == ScalarDataFormat::FLOAT) {
        addField(static_cast<float*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
    } else if (dataFormat == ScalarDataFormat::BYTE) {
        addField(static_cast<uint8_t*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
    } else if (dataFormat == ScalarDataFormat::SHORT) {
        addField(static_cast<uint16_t*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
    } else if (dataFormat == ScalarDataFormat::FLOAT16) {
        addField(static_cast<HalfFloat*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
    }
}
