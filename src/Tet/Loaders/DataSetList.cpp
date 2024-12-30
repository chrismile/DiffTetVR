/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <json/json.h>

#include <Utils/AppSettings.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/Regex/TransformString.hpp>

#include "DataSetList.hpp"

bool jsonValueToBool(const Json::Value& value) {
    if (value.isString()) {
        std::string valueString = value.asString();
        if (valueString == "true") {
            return true;
        } else if (valueString == "false") {
            return false;
        } else {
            sgl::Logfile::get()->throwError("Error in jsonValueToBool: Invalid value \"" + valueString + "\".");
            return false;
        }
    } else {
        return value.asBool();
    }
}

void processDataSetNodeChildren(Json::Value& childList, DataSetInformation* dataSetInformationParent) {
    for (Json::Value& source : childList) {
        auto* dataSetInformation = new DataSetInformation;

        // Get the type information.
        std::string typeName = source.isMember("type") ? source["type"].asString() : "tetmesh";
        if (typeName == "node") {
            dataSetInformation->type = DataSetType::NODE;
        } else if (typeName == "tetmesh") {
            dataSetInformation->type = DataSetType::TET_MESH;
        } else {
            sgl::Logfile::get()->writeError(
                    "Error in processDataSetNodeChildren: Invalid type name \"" + typeName + "\".");
            return;
        }

        dataSetInformation->name = source["name"].asString();

        if (dataSetInformation->type == DataSetType::NODE) {
            dataSetInformationParent->children.emplace_back(dataSetInformation);
            processDataSetNodeChildren(source["children"], dataSetInformation);
            continue;
        }

        Json::Value filenames;
        if (source.isMember("filenames")) {
            filenames = source["filenames"];
        } else if (source.isMember("filename")) {
            filenames = source["filename"];
        }
        const std::string tetMeshDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "DataSets/";
        if (filenames.isArray()) {
            for (const auto& filename : filenames) {
                std::string pathString = filename.asString();
                bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(pathString);
                if (isAbsolutePath) {
                    dataSetInformation->filenames.push_back(pathString);
                } else {
                    dataSetInformation->filenames.push_back(tetMeshDataSetsDirectory + pathString);
                }
            }
        } else {
            std::string pathString = filenames.asString();
            bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(pathString);
            if (isAbsolutePath) {
                dataSetInformation->filenames.push_back(pathString);
            } else {
                dataSetInformation->filenames.push_back(tetMeshDataSetsDirectory + pathString);
            }
        }

        // Optional data: Transform.
        dataSetInformation->hasCustomTransform = source.isMember("transform");
        if (dataSetInformation->hasCustomTransform) {
            glm::mat4 transformMatrix = parseTransformString(source["transform"].asString());
            dataSetInformation->transformMatrix = transformMatrix;
        }

        dataSetInformationParent->children.emplace_back(dataSetInformation);
    }
}

DataSetInformationPtr loadDataSetList(const std::string& filename, bool isFileWatchReload) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filename.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString, !isFileWatchReload);
        return {};
    }
    jsonFileStream.close();

    DataSetInformationPtr dataSetInformationRoot(new DataSetInformation);
    Json::Value& dataSetNode = root["datasets"];
    processDataSetNodeChildren(dataSetNode, dataSetInformationRoot.get());
    return dataSetInformationRoot;
}
