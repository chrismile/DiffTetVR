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

#ifndef DIFFTETVR_MESHLOADER_HPP
#define DIFFTETVR_MESHLOADER_HPP

#include "TetMeshLoader.hpp"

namespace sgl {
class LineReader;
}

/**
 * Loader for MEDIT .mesh files. For more information see:
 * - https://victorsndvg.github.io/FEconv/formats/ff++mesh.xhtml
 * - https://wias-berlin.de/software/tetgen/1.5/doc/manual/manual006.html#ff_mesh
 */
class MeshLoader : public TetMeshLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "mesh" }; }
    ~MeshLoader() override = default;
    bool loadFromFile(
            const std::string& filePath, std::vector<uint32_t>& cellIndices,
            std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors,
            std::vector<glm::vec4>& cellColors) override;
    bool peekSizes(const std::string& filePath, size_t& numCells, size_t& numVertices) override;

private:
    bool parseVertices(
            const std::string& filePath, sgl::LineReader& lineReader,
            size_t numVertices, std::vector<glm::vec3>& vertexPositions);
    bool parseTetrahedra(
            const std::string& filePath, sgl::LineReader& lineReader,
            size_t numTetrahedra, std::vector<uint32_t>& cellIndices);
    bool parseTriangles(
            const std::string& filePath, sgl::LineReader& lineReader, size_t numTriangles);
    bool parseEdges(
            const std::string& filePath, sgl::LineReader& lineReader, size_t numTriangles);
};

#endif //DIFFTETVR_MESHLOADER_HPP
