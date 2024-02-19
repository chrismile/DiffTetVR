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

#include <iostream>

#include "CSPSolver.hpp"

const char* RF_ARRAY[] = {
        "R", "F"
};

bool writeGraphviz(const std::vector<Prism>& prisms) {
    std::cout << "digraph CspGraph {" << std::endl;
    for (int prismIdx = 0; prismIdx < int(prisms.size()); prismIdx++) {
        std::cout << prismIdx << ";" << std::endl;
    }
    for (int prismIdx = 0; prismIdx < int(prisms.size()); prismIdx++) {
        const Prism& prism = prisms.at(prismIdx);
        for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
            int neighborIdx = prism.neighbors.at(neighborIdxLocal);
            if (neighborIdx >= 0) {
                const Prism& neighbor = prisms.at(neighborIdx);
                int neighborFaceIdx = prism.neighborFaceIndices.at(neighborIdxLocal) % 3;
                std::string_view attrs = "";
                if (prism.cuts.getCut(neighborIdxLocal) != 1u - neighbor.cuts.getCut(neighborFaceIdx)
                        || prism.cuts.bitfield == 0b000u || prism.cuts.bitfield == 0b111u) {
                    attrs = ";color=red";
                }
                std::cout << prismIdx << " -> " << neighborIdx << "[label=" << RF_ARRAY[prism.cuts.getCut(neighborIdxLocal)] << attrs << "];" << std::endl;
                //std::cout << neighborIdx << " -> " << prismIdx << "[label=" << RF_ARRAY[1u - neighbor.cuts.getCut(neighborFaceIdx)] << attrs << "];" << std::endl;
            }
        }
    }
    std::cout << "}" << std::endl;
    return true;
}

bool checkIsCspFulfilled(const std::vector<Prism>& prisms) {
    for (const Prism& prism : prisms) {
        if (prism.cuts.bitfield == 0b000u || prism.cuts.bitfield == 0b111u) {
            writeGraphviz(prisms);
            return false;
        }
        for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
            int neighborIdx = prism.neighbors.at(neighborIdxLocal);
            if (neighborIdx >= 0) {
                const Prism& neighbor = prisms.at(neighborIdx);
                int neighborFaceIdx = prism.neighborFaceIndices.at(neighborIdxLocal) % 3;
                if (prism.cuts.getCut(neighborIdxLocal) != 1u - neighbor.cuts.getCut(neighborFaceIdx)) {
                    writeGraphviz(prisms);
                    return false;
                }
            }
        }
    }
    return true;
}
