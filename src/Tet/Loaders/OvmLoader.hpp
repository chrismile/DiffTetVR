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

#ifndef DIFFTETVR_OVMLOADER_HPP
#define DIFFTETVR_OVMLOADER_HPP

#include "TetMeshLoader.hpp"

class OvmLoader : public TetMeshLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "ovm", "ovmb", "vtk" }; }
    ~OvmLoader() override = default;
    bool getNeedsOpenVolumeMeshSupport() override { return true; }
    bool loadFromFile(
            const std::string& filePath, std::vector<uint32_t>& cellIndices,
            std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors) override { return false; }
    bool loadFromFileOvm(
            const std::string& filePath, OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh) override;
};

class OvmWriter : public TetMeshWriter {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "ovm", "ovmb" }; }
    ~OvmWriter() override = default;
    bool getNeedsOpenVolumeMeshSupport() override { return true; }
    bool saveToFile(
            const std::string& filePath, const std::vector<uint32_t>& cellIndices,
            const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec4>& vertexColors) override { return false; }
    bool saveToFileOvm(
            const std::string& filePath, const OpenVolumeMesh::GeometricTetrahedralMeshV3f& ovmMesh) override;
};

#endif //DIFFTETVR_OVMLOADER_HPP
