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

#include <fstream>

#include <Utils/StringUtils.hpp>
#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/LineReader.hpp>

#include "MshLoader.hpp"

bool MshLoader::expectSectionEnd(
        const std::string& filePath, sgl::LineReader& lineReader, const std::string& sectionEndTag) {
    std::string parsedSectionEndTag = lineReader.readLine();
    if (parsedSectionEndTag != sectionEndTag) {
        sgl::Logfile::get()->writeErrorVar(
                "Error in MshLoader::loadFromFile: Unexpected section end tag (got \"", parsedSectionEndTag,
                "\", expected \"", sectionEndTag, "\") in file \"", filePath, "\".");
        return false;
    }
    return true;
}

#pragma pack(1)
struct MshBinaryNode {
    int32_t nodeNumber;
    double x;
    double y;
    double z;
};

bool MshLoader::parseNodes(
        const std::string& filePath, sgl::LineReader& lineReader, std::vector<glm::vec3>& vertexPositions) {
    auto numElements = lineReader.readScalarLine<size_t>();
    if (isBinary) {
        //size_t dataNumBytes = numElements * (4 + 3 * numberByteSize);
        const auto* nodes = lineReader.getTypedPointerAndAdvance<MshBinaryNode>(numElements);
        for (size_t elementIdx = 0; elementIdx < numElements; elementIdx++) {
            auto& node = nodes[elementIdx];
            if (node.nodeNumber - 1 != int32_t(elementIdx)) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: The loader currently expects contiguous nodes, but file \"",
                        filePath, "\" does not start with numbering entries starting from 1.");
                return false;
            }
            vertexPositions.emplace_back(float(node.x), float(node.y), float(node.z));
        }
    } else {
        MshBinaryNode node{};
        for (size_t elementIdx = 0; elementIdx < numElements; elementIdx++) {
            lineReader.readStructLine(node.nodeNumber, node.x, node.y, node.z);
            vertexPositions.emplace_back(float(node.x), float(node.y), float(node.z));
        }
    }
    if (!expectSectionEnd(filePath, lineReader, "$EndNodes")) {
        return false;
    }
    return true;
}

bool MshLoader::parseElements(
        const std::string& filePath, sgl::LineReader& lineReader, std::vector<uint32_t>& cellIndices) {
    auto numElements = lineReader.readScalarLine<size_t>();
    if (isBinary) {
        // Parse the header.
        auto elementType = lineReader.readBinaryValue<int32_t>();
        auto numElementsFollowing = lineReader.readBinaryValue<int32_t>();
        auto numTags = lineReader.readBinaryValue<int32_t>();
        if (elementType != 4) {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: Expected element type 4 (tetrahedron), but got element type ",
                    elementType, " instead in file \"", filePath, "\".");
            return false;
        }
        if (numTags != 0) {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: Expected zero tags, but got ",
                    numTags, " instead in file \"", filePath, "\".");
            return false;
        }
        if (numElementsFollowing != int32_t(numElements)) {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: The loader currently does not support multiple element headers ",
                    "as used in file \"", filePath, "\".");
            return false;
        }

        for (size_t elementIdx = 0; elementIdx < numElements; elementIdx++) {
            auto elementNumber = lineReader.readBinaryValue<int32_t>();
            if (elementNumber - 1 != int32_t(elementIdx)) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: The loader currently expects contiguous elements, but file \"",
                        filePath, "\" does not start with numbering entries starting from 1.");
                return false;
            }

            // No tags supported so far, so the node number list directly follows.
            /*for (int i = 0; i < 4; i++) {
                auto nodeNumber = lineReader.readBinaryValue<int32_t>();
                cellIndices.push_back(uint32_t(nodeNumber) - 1);
            }*/
            auto nodeNumber1 = lineReader.readBinaryValue<int32_t>();
            auto nodeNumber2 = lineReader.readBinaryValue<int32_t>();
            auto nodeNumber3 = lineReader.readBinaryValue<int32_t>();
            auto nodeNumber4 = lineReader.readBinaryValue<int32_t>();
            // The default ordering in .msh files is in different winding than what DiffTetVR expects.
            cellIndices.push_back(uint32_t(nodeNumber2) - 1);
            cellIndices.push_back(uint32_t(nodeNumber1) - 1);
            cellIndices.push_back(uint32_t(nodeNumber3) - 1);
            cellIndices.push_back(uint32_t(nodeNumber4) - 1);
        }
    } else {
        std::vector<std::string> lineVec;
        for (size_t elementIdx = 0; elementIdx < numElements; elementIdx++) {
            lineVec.clear();
            lineReader.readVectorLine(lineVec);
            if (lineVec.size() != 7) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: Expected 7 entries per line in file \"",
                        filePath, "\", but got ", lineVec.size(), " instead.");
                return false;
            }
            auto elementNumber = sgl::fromString<int32_t>(lineVec.at(0));
            if (elementNumber - 1 != int32_t(elementIdx)) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: The loader currently expects contiguous elements, but file \"",
                        filePath, "\" does not start with numbering entries starting from 1.");
                return false;
            }
            auto elementType = sgl::fromString<int32_t>(lineVec.at(1));
            if (elementType != 4) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: Expected element type 4 (tetrahedron), but got element type ",
                        elementType, " instead in file \"", filePath, "\".");
                return false;
            }
            auto numTags = sgl::fromString<int32_t>(lineVec.at(2));
            if (numTags != 0) {
                sgl::Logfile::get()->writeErrorVar(
                        "Error in MshLoader::loadFromFile: Expected zero tags, but got ",
                        numTags, " instead in file \"", filePath, "\".");
                return false;
            }
            auto nodeNumber1 = sgl::fromString<int32_t>(lineVec.at(3));
            auto nodeNumber2 = sgl::fromString<int32_t>(lineVec.at(4));
            auto nodeNumber3 = sgl::fromString<int32_t>(lineVec.at(5));
            auto nodeNumber4 = sgl::fromString<int32_t>(lineVec.at(6));
            // The default ordering in .msh files is in different winding than what DiffTetVR expects.
            cellIndices.push_back(uint32_t(nodeNumber2) - 1);
            cellIndices.push_back(uint32_t(nodeNumber1) - 1);
            cellIndices.push_back(uint32_t(nodeNumber3) - 1);
            cellIndices.push_back(uint32_t(nodeNumber4) - 1);
        }
    }

    if (!expectSectionEnd(filePath, lineReader, "$EndElements")) {
        return false;
    }

    return true;
}

bool MshLoader::parseNodeData(
        const std::string& filePath, sgl::LineReader& lineReader) {
    // The first string tag is the name of the data.
    auto numStringTags = lineReader.readScalarLine<size_t>();
    for (size_t i = 0; i < numStringTags; i++) {
        lineReader.readLine();
    }
    // The first real tag is a time value.
    auto numRealTags = lineReader.readScalarLine<size_t>();
    for (size_t i = 0; i < numRealTags; i++) {
        lineReader.readScalarLine<double>();
    }
    // The first integer tag is a time step index, the second the number of field components,
    // and the third is the number of entities.
    auto numIntegerTags = lineReader.readScalarLine<size_t>();
    size_t numEntries = 0;
    for (size_t i = 0; i < numIntegerTags; i++) {
        auto val = lineReader.readScalarLine<int32_t>();
        if (i == 2) {
            numEntries = size_t(val);
        }
    }

    if (isBinary) {
        for (size_t entryIdx = 0; entryIdx < numEntries; entryIdx++) {
            auto nodeNumber = lineReader.readBinaryValue<int32_t>();
            auto value = lineReader.readBinaryValue<double>();
            (void)nodeNumber;
            (void)value;
        }
    } else {
        int32_t nodeNumber;
        double value;
        for (size_t entryIdx = 0; entryIdx < numEntries; entryIdx++) {
            lineReader.readStructLine(nodeNumber, value);
        }
    }

    if (!expectSectionEnd(filePath, lineReader, "$EndNodeData")) {
        return false;
    }
    return true;
}

bool MshLoader::parseElementData(
        const std::string& filePath, sgl::LineReader& lineReader) {
    // The first string tag is the name of the data.
    auto numStringTags = lineReader.readScalarLine<size_t>();
    for (size_t i = 0; i < numStringTags; i++) {
        lineReader.readLine();
    }
    // The first real tag is a time value.
    auto numRealTags = lineReader.readScalarLine<size_t>();
    for (size_t i = 0; i < numRealTags; i++) {
        lineReader.readScalarLine<double>();
    }
    // The first integer tag is a time step index, the second the number of field components,
    // and the third is the number of entities.
    auto numIntegerTags = lineReader.readScalarLine<size_t>();
    size_t numEntries = 0;
    for (size_t i = 0; i < numIntegerTags; i++) {
        auto val = lineReader.readScalarLine<int32_t>();
        if (i == 2) {
            numEntries = size_t(val);
        }
    }

    if (isBinary) {
        for (size_t entryIdx = 0; entryIdx < numEntries; entryIdx++) {
            auto elementNumber = lineReader.readBinaryValue<int32_t>();
            auto value = lineReader.readBinaryValue<double>();
            (void)elementNumber;
            (void)value;
        }
    } else {
        int32_t elementNumber;
        double value;
        for (size_t entryIdx = 0; entryIdx < numEntries; entryIdx++) {
            lineReader.readStructLine(elementNumber, value);
        }
    }

    if (!expectSectionEnd(filePath, lineReader, "$EndElementData")) {
        return false;
    }
    return true;
}

bool MshLoader::loadFromFile(
        const std::string& filePath, std::vector<uint32_t>& cellIndices,
        std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec4>& vertexColors,
        std::vector<glm::vec4>& cellColors) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(filePath, buffer, length, true);
    if (!loaded) {
        return false;
    }
    //char* fileBuffer = reinterpret_cast<char*>(buffer);
    sgl::LineReader lineReader(reinterpret_cast<const char*>(buffer), length);

    // Check whether the file is in ASCII or binary format.
    std::string formatNameString = lineReader.readLine();
    if (formatNameString != "$MeshFormat") {
        sgl::Logfile::get()->writeErrorVar(
                "Error in MshLoader::loadFromFile: Invalid format name \"", formatNameString,
                "\" in file \"", filePath, "\".");
        return false;
    }
    // The following line is structured as follows: version-number file-type data-size.
    std::string formatString = lineReader.readLine();
    std::vector<std::string> formatParts;
    sgl::splitStringWhitespace(formatString, formatParts);
    if (formatParts.size() != 3) {
        sgl::Logfile::get()->writeErrorVar(
                "Error in MshLoader::loadFromFile: Malformed format in file \"", filePath, "\".");
        return false;
    }
    isBinary = formatParts.at(1) != "0";
    if (isBinary) {
        auto oneBinary = lineReader.readBinaryValue<int32_t>();
        if (oneBinary != 1) {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: Expected binary one in header in file \"", filePath, "\".");
            return false;
        }
    }
    numberByteSize = sgl::fromString<size_t>(formatParts.at(2));
    if (numberByteSize != sizeof(double)) {
        sgl::Logfile::get()->writeErrorVar(
                "Error in MshLoader::loadFromFile: Expected data size of 8 in \"", filePath, "\".");
        return false;
    }
    if (!expectSectionEnd(filePath, lineReader, "$EndMeshFormat")) {
        return false;
    }

    // Next comes the node data.
    while (lineReader.isLineLeft()) {
        std::string sectionNameString = lineReader.readLine();
        if (sectionNameString == "$Nodes") {
            if (!parseNodes(filePath, lineReader, vertexPositions)) {
                return false;
            }
        } else if (sectionNameString == "$Elements") {
            if (!parseElements(filePath, lineReader, cellIndices)) {
                return false;
            }
        } else if (sectionNameString == "$NodeData") {
            if (!parseNodeData(filePath, lineReader)) {
                return false;
            }
        } else if (sectionNameString == "$ElementData") {
            if (!parseElementData(filePath, lineReader)) {
                return false;
            }
        } else if (sectionNameString == "$PhysicalNames") {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: Encountered not yet supported section \"", sectionNameString,
                    "\" in file \"", filePath, "\".");
            return false;
        } else {
            sgl::Logfile::get()->writeErrorVar(
                    "Error in MshLoader::loadFromFile: Encountered invalid section \"", sectionNameString,
                    "\" in file \"", filePath, "\".");
            return false;
        }
    }
    // TODO: Support loading of color data.
    vertexColors.resize(vertexPositions.size(), glm::vec4(0.5f, 0.5f, 0.5f, 0.1f));

    delete[] buffer;
    return true;
}

bool MshLoader::peekSizes(const std::string& filePath, size_t& numCells, size_t& numVertices) {
    return false;
}
