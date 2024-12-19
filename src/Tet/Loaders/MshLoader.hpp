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

#ifndef DIFFTETVR_MSHLOADER_HPP
#define DIFFTETVR_MSHLOADER_HPP

#include "TetMeshLoader.hpp"

namespace sgl {
class LineReader;
}

/**
 * Loader for Gmsh .msh files. For more information see:
 * - https://victorsndvg.github.io/FEconv/formats/gmshmsh.xhtml
 * - https://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php (ASCII)
 * - https://www.manpagez.com/info/gmsh/gmsh-2.4.0/gmsh_57.php (binary)
 * - https://gmsh.info/#Documentation
 */
class MshLoader : public TetMeshLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "msh" }; }
    ~MshLoader() override = default;
    bool loadFromFile(
            const std::string& filePath, std::vector<uint32_t>& cellIndices,
            std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors) override;
    bool peekSizes(const std::string& filePath, size_t& numCells, size_t& numVertices) override;

private:
    bool expectSectionEnd(const std::string& filePath, sgl::LineReader& lineReader, const std::string& sectionEndTag);
    bool parseNodes(const std::string& filePath, sgl::LineReader& lineReader, std::vector<glm::vec3>& vertexPositions);
    bool parseElements(const std::string& filePath, sgl::LineReader& lineReader, std::vector<uint32_t>& cellIndices);
    bool parseNodeData(const std::string& filePath, sgl::LineReader& lineReader);
    bool parseElementData(const std::string& filePath, sgl::LineReader& lineReader);
    bool isBinary = false;
    size_t numberByteSize = 0;
};

#endif //DIFFTETVR_MSHLOADER_HPP
