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

#include <cstring>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Color.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Tet/TetMesh.hpp"
#include "Tet/Loaders/LoadersUtil.hpp"
#include "VtkWriter.hpp"

void VtkWriter::initializeWriter(const std::string& _filename, bool _isBinaryVtk) {
    filename = _filename;
    if (sgl::FileUtils::get()->getFileExtension(filename) == "vtk") {
        filename = sgl::FileUtils::get()->removeExtension(filename);
    }
    isBinaryVtk = _isBinaryVtk;
}

void VtkWriter::writeNextTimeStep(
        const std::vector<uint32_t>& cellIndices,
        const glm::vec3* vertexPositions,
        const glm::vec4* vertexColors,
        const glm::vec3* vertexPositionGradients,
        const glm::vec4* vertexColorGradients,
        int numPoints) {
    std::string vtkFilename = filename + "." + std::to_string(timeStepNumber) + ".vtk";
    FILE* file;
    if (isBinaryVtk) {
        file = fopen(vtkFilename.c_str(), "wb");
    } else {
        file = fopen(vtkFilename.c_str(), "w");
    }
    if (file == nullptr) {
        sgl::Logfile::get()->throwError("Error: Couldn't open file \"" + vtkFilename + "\" for writing.");
    }

    auto* vertexColorGradientsMagnitude = new float[numPoints];
    auto* vertexAlphaGradients = new float[numPoints];
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& r) {
        for (auto i = r.begin(); i != r.end(); i++) {
#else
    #pragma omp parallel for default(none) \
    shared(numPoints, vertexColorGradients, vertexColorGradientsMagnitude, vertexAlphaGradients)
    for (int i = 0; i < numPoints; i++) {
#endif
        const glm::vec4& v = vertexColorGradients[i];
        vertexColorGradientsMagnitude[i] = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        vertexAlphaGradients[i] = v.a;
    }
#ifdef USE_TBB
    });
#endif

    writeVtkHeader(file);
    writePointCoordinates(file, vertexPositions, numPoints);
    writeCells(file, cellIndices);
    fprintf(file, "POINT_DATA %i\n", numPoints);
    writePointDataVector(file, vertexPositionGradients, numPoints, "PositionGrad");
    writePointDataScalar(file, vertexColorGradientsMagnitude, numPoints, "RGBGradMag");
    writePointDataScalar(file, vertexAlphaGradients, numPoints, "AlphaGrad");
    writePointDataColor(file, vertexColors, numPoints, "VertexColor");

    delete[] vertexColorGradientsMagnitude;
    delete[] vertexAlphaGradients;
    fclose(file);
    timeStepNumber++;
}

void VtkWriter::writeNextTimeStep(sgl::vk::Renderer* renderer, const TetMeshPtr& tetMesh) {
    //renderer->insertMemoryBarrier(
    //        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
    //        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    sgl::vk::BufferPtr vertexPositionStagingBuffer;
    sgl::vk::BufferPtr vertexColorStagingBuffer;
    sgl::vk::BufferPtr vertexPositionGradientStagingBuffer;
    sgl::vk::BufferPtr vertexColorGradientStagingBuffer;
    const auto& cellIndices = tetMesh->getCellIndices();
    auto vertexPositionBuffer = tetMesh->getVertexPositionBuffer();
    auto vertexColorBuffer = tetMesh->getVertexColorBuffer();
    auto vertexPositionGradientBuffer = tetMesh->getVertexPositionGradientBuffer();
    auto vertexColorGradientBuffer = tetMesh->getVertexColorGradientBuffer();
    if (!vertexPositionStagingBuffer
            || vertexPositionStagingBuffer->getSizeInBytes() != vertexPositionBuffer->getSizeInBytes()) {
        vertexPositionStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexPositionBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    if (!vertexColorStagingBuffer
            || vertexColorStagingBuffer->getSizeInBytes() != vertexColorBuffer->getSizeInBytes()) {
        vertexColorStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexColorBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    if (!vertexPositionGradientStagingBuffer
            || vertexPositionGradientStagingBuffer->getSizeInBytes() != vertexPositionGradientBuffer->getSizeInBytes()) {
        vertexPositionGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexPositionGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    if (!vertexColorGradientStagingBuffer
            || vertexColorGradientStagingBuffer->getSizeInBytes() != vertexColorGradientBuffer->getSizeInBytes()) {
        vertexColorGradientStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), vertexColorGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
    vertexPositionBuffer->copyDataTo(
            vertexPositionStagingBuffer, 0, 0, vertexPositionStagingBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    vertexColorBuffer->copyDataTo(
            vertexColorStagingBuffer, 0, 0, vertexColorStagingBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    vertexPositionGradientBuffer->copyDataTo(
            vertexPositionGradientStagingBuffer, 0, 0, vertexPositionGradientBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    vertexColorGradientBuffer->copyDataTo(
            vertexColorGradientStagingBuffer, 0, 0, vertexColorGradientBuffer->getSizeInBytes(),
            renderer->getVkCommandBuffer());
    renderer->syncWithCpu();

    auto* vertexPositions = reinterpret_cast<glm::vec3*>(vertexPositionStagingBuffer->mapMemory());
    auto* vertexColors = reinterpret_cast<glm::vec4*>(vertexColorStagingBuffer->mapMemory());
    auto* vertexPositionGradients = reinterpret_cast<glm::vec3*>(vertexPositionGradientStagingBuffer->mapMemory());
    auto* vertexColorGradients = reinterpret_cast<glm::vec4*>(vertexColorGradientStagingBuffer->mapMemory());
    const auto numVertices = int(vertexPositionBuffer->getSizeInBytes() / sizeof(glm::vec3));
    writeNextTimeStep(
            cellIndices, vertexPositions, vertexColors,
            vertexPositionGradients, vertexColorGradients, numVertices);
    vertexPositionStagingBuffer->unmapMemory();
    vertexColorStagingBuffer->unmapMemory();
    vertexPositionGradientStagingBuffer->unmapMemory();
    vertexColorGradientStagingBuffer->unmapMemory();
}

void VtkWriter::writeVtkHeader(FILE* file) const {
    fprintf(file, "# vtk DataFile Version 2.0\n");
    fprintf(file, "Generated by DiffTetVR\n");
    if (isBinaryVtk) {
        fprintf(file, "BINARY\n");
    } else {
        fprintf(file, "ASCII\n");
    }
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");
}

void VtkWriter::writePointCoordinates(FILE* file, const glm::vec3* vertexPositions, int numPoints) const {
    fprintf(file, "POINTS %i float\n", numPoints);
    if (isBinaryVtk) {
        auto* pointDataVector = new float[numPoints * 3];
        memcpy(pointDataVector, vertexPositions, numPoints * sizeof(glm::vec3));
        swapEndianness(pointDataVector, numPoints * 3);
        fwrite(pointDataVector, sizeof(glm::vec3), numPoints, file);
        delete[] pointDataVector;
    } else {
        for (int i = 0; i < numPoints; i++) {
            const glm::vec3& pt = vertexPositions[i];
            fprintf(file, "%f %f %f\n", pt.x, pt.y, pt.z);
        }
    }
}

void VtkWriter::writeCells(FILE* file, const std::vector<uint32_t>& cellIndices) const {
    const int VTK_TET_TYPE = 10;
    const int numCells = int(cellIndices.size() / 4);

    fprintf(file, "CELLS %i %i\n", numCells, int(cellIndices.size()) + numCells);
    if (isBinaryVtk) {
        std::vector<int32_t> dataField(int(cellIndices.size()) + numCells);
        for (int i = 0; i < numCells; i++) {
            auto offset = size_t(i) * 4;
            auto offsetOut = size_t(i) * 5;
            dataField.at(offsetOut + 0) = 4;
            dataField.at(offsetOut + 1) = int(cellIndices.at(offset));
            dataField.at(offsetOut + 2) = int(cellIndices.at(offset + 1));
            dataField.at(offsetOut + 3) = int(cellIndices.at(offset + 2));
            dataField.at(offsetOut + 4) = int(cellIndices.at(offset + 3));
        }
        swapEndianness(dataField.data(), int(dataField.size()));
        fwrite(dataField.data(), sizeof(uint32_t), dataField.size(), file);
    } else {
        for (int i = 0; i < numCells; i++) {
            auto offset = size_t(i) * 4;
            uint32_t i0 = cellIndices.at(offset);
            uint32_t i1 = cellIndices.at(offset + 1);
            uint32_t i2 = cellIndices.at(offset + 2);
            uint32_t i3 = cellIndices.at(offset + 3);
            fprintf(file, "%i %i %i %i %i\n", 4, i0, i1, i2, i3);
        }
    }

    fprintf(file, "CELL_TYPES %i\n", numCells);
    if (isBinaryVtk) {
        std::vector<int32_t> cellTypeArray(numCells, VTK_TET_TYPE);
        swapEndianness(cellTypeArray.data(), int(cellTypeArray.size()));
        fwrite(cellTypeArray.data(), sizeof(int32_t), numCells, file);
    } else {
        for (int i = 0; i < numCells; i++) {
            fprintf(file, "%i\n", VTK_TET_TYPE);
        }
    }
}

void VtkWriter::writePointDataVector(
        FILE* file, const glm::vec3* vectorData, int numPoints, const std::string& vectorName) const {
    std::string header = "VECTORS " + vectorName + " float\n";
    fwrite(header.c_str(), sizeof(char), header.size(), file);

    if (isBinaryVtk) {
        auto* pointDataVector = new float[numPoints * 3];
        memcpy(pointDataVector, vectorData, numPoints * sizeof(glm::vec3));
        swapEndianness(pointDataVector, numPoints * 3);
        fwrite(pointDataVector, sizeof(glm::vec3), numPoints, file);
        delete[] pointDataVector;
    } else {
        for (int i = 0; i < numPoints; i++) {
            const glm::vec3& v = vectorData[i];
            fprintf(file, "%f %f %f\n", v.x, v.y, v.z);
        }
    }
}

void VtkWriter::writePointDataScalar(
        FILE* file, const float* scalarData, int numPoints, const std::string& scalarName) const {
    std::string header = "SCALARS " + scalarName + " float 1\n";
    fwrite(header.c_str(), sizeof(char), header.size(), file);
    fprintf(file, "LOOKUP_TABLE default\n");

    if (isBinaryVtk) {
        auto* pointDataScalar = new float[numPoints];
        memcpy(pointDataScalar, scalarData, numPoints * sizeof(float));
        swapEndianness(pointDataScalar, numPoints);
        fwrite(pointDataScalar, sizeof(float), numPoints, file);
        delete[] pointDataScalar;
    } else {
        for (int i = 0; i < numPoints; i++) {
            const float& s = scalarData[i];
            fprintf(file, "%f\n", s);
        }
    }
}

void VtkWriter::writePointDataColor(
        FILE* file, const glm::vec4* colorDataVec4, int numPoints, const std::string& scalarName) const {
    std::string header = "COLOR_SCALARS " + scalarName + " 4\n";
    fwrite(header.c_str(), sizeof(char), header.size(), file);

    if (isBinaryVtk) {
        auto* pointDataScalar = new uint32_t[numPoints];
        for (int i = 0; i < numPoints; i++) {
            pointDataScalar[i] = sgl::colorFromVec4(colorDataVec4[i]).getColorRGBA();
        }
        swapEndianness(pointDataScalar, numPoints);
        fwrite(pointDataScalar, sizeof(uint32_t), numPoints, file);
        delete[] pointDataScalar;
    } else {
        for (int i = 0; i < numPoints; i++) {
            const glm::vec4& c = colorDataVec4[i];
            fprintf(file, "%f %f %f %f\n", c.r, c.g, c.b, c.a);
        }
    }
}
