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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/File/Logfile.hpp>
#include "DenseVoxelCarvingCPU.hpp"

#define IDX_GRID(x, y, z) ((x) + ((y) + (z) * gridResolution.y) * gridResolution.x)

DenseVoxelCarvingCPU::DenseVoxelCarvingCPU(const sgl::AABB3& gridBoundingBox, const glm::uvec3& gridResolution)
        : VoxelCarving(gridBoundingBox, gridResolution) {
    voxelGrid = new uint8_t[gridResolution.x * gridResolution.y * gridResolution.z];
    for (uint32_t z = 0; z < gridResolution.z; z++) {
        for (uint32_t y = 0; y < gridResolution.y; y++) {
            for (uint32_t x = 0; x < gridResolution.x; x++) {
                voxelGrid[IDX_GRID(x, y, z)] = 2; // 2: unknown, 1: filled, 0: culled
            }
        }
    }
}

DenseVoxelCarvingCPU::~DenseVoxelCarvingCPU() {
    delete[] voxelGrid;
}

void DenseVoxelCarvingCPU::processNextFrame(const torch::Tensor& inputImage, const CameraSettings& cameraSettings) {
    auto gridSpaceToClipSpaceMatrix = cameraSettings.getViewProjectionMatrix() * voxelGridSpaceToWorldSpaceMatrix;
    auto inputImageAccessor = inputImage.accessor<float, 3>();
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, gridResolution.z), [&](auto const& r) {
            for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel for default(none) shared(gridResolution, voxelGrid, inputImageAccessor, gridSpaceToClipSpaceMatrix)
#endif
    for (uint32_t z = 0; z < gridResolution.z; z++) {
#endif
        for (uint32_t y = 0; y < gridResolution.y; y++) {
            for (uint32_t x = 0; x < gridResolution.x; x++) {
                uint8_t& voxelData = voxelGrid[IDX_GRID(x, y, z)];
                uint8_t valueNew = sampleImage(
                        inputImageAccessor, gridSpaceToClipSpaceMatrix, glm::vec4(x, y, z, 1.0f));
                voxelData = std::min(voxelData, valueNew);
            }
        }
    }
#ifdef USE_TBB
    });
#endif
}

sgl::AABB3 DenseVoxelCarvingCPU::computeNonEmptyBoundingBox() {
    const glm::vec3 trafoScale = (gridBoundingBox.max - gridBoundingBox.min) / glm::vec3(gridResolution);
#ifdef USE_TBB
    #define minX init.min.x
    #define minY init.min.y
    #define minZ init.min.z
    #define maxX init.max.x
    #define maxY init.max.y
    #define maxZ init.max.z
    return tbb::parallel_reduce(
            tbb::blocked_range<uint32_t>(0, gridResolution.z), sgl::AABB3(),
            [&](tbb::blocked_range<uint32_t> const& r, sgl::AABB3 init) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
    float minX, minY, minZ, maxX, maxY, maxZ;
    minX = minY = minZ = std::numeric_limits<float>::max();
    maxX = maxY = maxZ = std::numeric_limits<float>::lowest();
#if _OPENMP >= 201107
    #pragma omp parallel for default(none) shared(gridResolution, voxelGrid, trafoScale) \
    reduction(min: minX) reduction(min: minY) reduction(min: minZ) \
    reduction(max: maxX) reduction(max: maxY) reduction(max: maxZ)
#endif
    for (uint32_t z = 0; z < gridResolution.z; z++) {
#endif
        for (uint32_t y = 0; y < gridResolution.y; y++) {
            for (uint32_t x = 0; x < gridResolution.x; x++) {
                uint8_t voxelData = voxelGrid[IDX_GRID(x, y, z)];
                if (voxelData == 1) {
                    glm::vec3 posWorld = (glm::vec3(x, y, z) + glm::vec3(0.5f)) * trafoScale + gridBoundingBox.min;
                    minX = std::min(minX, posWorld.x);
                    minY = std::min(minY, posWorld.y);
                    minZ = std::min(minZ, posWorld.z);
                    maxX = std::max(maxX, posWorld.x);
                    maxY = std::max(maxY, posWorld.y);
                    maxZ = std::max(maxZ, posWorld.z);
                }
            }
        }
    }
#ifdef USE_TBB
                return init;
            },
            [&](sgl::AABB3 lhs, sgl::AABB3 rhs) -> sgl::AABB3 {
                lhs.combine(rhs);
                return lhs;
            });
#else
    //exportVolume("/home/christoph/datasets/imgs/test.cvol"); // For testing.
    return sgl::AABB3(glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ));
#endif
}

uint8_t DenseVoxelCarvingCPU::sampleImage(
        const at::TensorAccessor<float, 3>& inputImageAccessor, const glm::mat4& gridSpaceToClipSpaceMatrix,
        const glm::vec4& voxelPosition) {
    glm::vec4 clipSpacePoint = gridSpaceToClipSpaceMatrix * voxelPosition;
    glm::vec3 ndcPoint(
            clipSpacePoint.x / clipSpacePoint.w,
            clipSpacePoint.y / clipSpacePoint.w,
            clipSpacePoint.z / clipSpacePoint.w);
    uint8_t imageValue = 2;
    if (ndcPoint.x >= -1.0f && ndcPoint.y >= -1.0f && ndcPoint.z >= 0.0f
            && ndcPoint.x <= 1.0f && ndcPoint.y <= 1.0f && ndcPoint.z <= 1.0f) {
        auto pixelLocation = glm::ivec2(
                int(clipSpacePoint.x / clipSpacePoint.z * inputImageAccessor.size(1)),
                int(clipSpacePoint.y / clipSpacePoint.z * inputImageAccessor.size(0)));
        if (pixelLocation.x >= 0 && pixelLocation.x < inputImageAccessor.size(1)
                && pixelLocation.y >= 0 && pixelLocation.y < inputImageAccessor.size(0)) {
            imageValue = uint8_t(inputImageAccessor[pixelLocation[1]][pixelLocation[0]][3] > BG_THRESH);
        }
    }
    return imageValue;
}

enum class CvolDataType {
    UNSIGNED_CHAR, UNSIGNED_SHORT, FLOAT
};

#pragma pack(push, 4)
struct CvolFileHeader {
    char magicNumber[4];
    size_t sizeX, sizeY, sizeZ;
    double voxelSizeX, voxelSizeY, voxelSizeZ;
    CvolDataType fieldType;
    uint64_t padding;
};
#pragma pack(pop)

// https://github.com/chrismile/Correrender/blob/main/src/Export/CvolWriter.cpp
bool DenseVoxelCarvingCPU::exportVolume(const std::string& filePath) {
    std::ofstream outfile(filePath, std::ios::binary);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in CvolWriter::writeFieldToFile: File \"" + filePath + "\" could not be opened for writing.");
        return false;
    }

    CvolFileHeader header{};
    memcpy(header.magicNumber, "cvol", 4);
    header.sizeX = size_t(gridResolution.x);
    header.sizeY = size_t(gridResolution.y);
    header.sizeZ = size_t(gridResolution.z);
    header.voxelSizeX = double(gridBoundingBox.getDimensions().x / float(gridResolution.x));
    header.voxelSizeY = double(gridBoundingBox.getDimensions().y / float(gridResolution.y));
    header.voxelSizeZ = double(gridBoundingBox.getDimensions().z / float(gridResolution.z));
    header.fieldType = CvolDataType::FLOAT;
    outfile.write(reinterpret_cast<char*>(&header), sizeof(CvolFileHeader));

    auto* fieldData = new float[gridResolution.x * gridResolution.y * gridResolution.z];
    for (uint32_t z = 0; z < gridResolution.z; z++) {
        for (uint32_t y = 0; y < gridResolution.y; y++) {
            for (uint32_t x = 0; x < gridResolution.x; x++) {
                uint32_t idx = IDX_GRID(x, y, z);
                fieldData[idx] = float(voxelGrid[idx]) * 0.5f;
            }
        }
    }
    outfile.write(reinterpret_cast<char*>(fieldData), sizeof(float) * header.sizeX * header.sizeY * header.sizeZ);
    delete[] fieldData;

    outfile.close();
    return true;
}
