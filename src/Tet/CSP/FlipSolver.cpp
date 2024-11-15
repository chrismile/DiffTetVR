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

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <optional>
#include <stdexcept>

#include "FlipSolver.hpp"

struct PossibleCuts {
    uint32_t bitfield;
    PossibleCuts() : bitfield(0b111111u) {}
    void update(uint32_t idx, uint32_t neighborCut) {
        bitfield |= (0b11u) << (idx * 2);
        bitfield ^= 1u << (idx * 2u + (1u - neighborCut));
    }
    [[nodiscard]] std::optional<Cuts> getConsistentCuts() const {
        if (bitfield == 0b010101u || bitfield == 0b101010u) {
            return {};
        }
        uint32_t p0 = bitfield & 0b11u;
        uint32_t p1 = (bitfield >> 2u) & 0b11u;
        uint32_t p2 = (bitfield >> 4u) & 0b11u;
        if (p0 == 0u || p1 == 0u || p2 == 0u) {
            return {};
        }
        if ((p0 & 0b01) != 0 && (p1 & 0b01) != 0 && (p2 & 0b10) != 0) {
            return Cuts(0, 0, 1);
        } else if ((p0 & 0b01) != 0 && (p1 & 0b10) != 0 && (p2 & 0b01) != 0) {
            return Cuts(0, 1, 0);
        } else if ((p0 & 0b01) != 0 && (p1 & 0b10) != 0 && (p2 & 0b10) != 0) {
            return Cuts(0, 1, 1);
        } else if ((p0 & 0b10) != 0 && (p1 & 0b01) != 0 && (p2 & 0b01) != 0) {
            return Cuts(1, 0, 0);
        } else if ((p0 & 0b10) != 0 && (p1 & 0b01) != 0 && (p2 & 0b10) != 0) {
            return Cuts(1, 0, 1);
        } else {
            return Cuts(1, 1, 0);
        }
    }
    [[nodiscard]] Cuts getAnyConsistentCut() const {
        if (bitfield == 0b010101u) {
            return Cuts(0, 0, 0);
        } else if (bitfield == 0b101010u) {
            return Cuts(1, 1, 1);
        } else {
            return getConsistentCuts().value();
        }
    }
};

struct BfsEntry {
    int currIdx = -1;
    std::unordered_map<int, Cuts> tetToCutsMap;
};

void doRippling(std::vector<Prism>& prisms, int startIdx, const std::vector<bool>& prismVisitedArray) {
    std::queue<std::shared_ptr<BfsEntry>> bfsQueue;
    auto startEntry = std::make_shared<BfsEntry>();
    startEntry->currIdx = startIdx;
    bfsQueue.push(startEntry);

    std::shared_ptr<BfsEntry> validRipplePath{};

    while (!bfsQueue.empty()) {
        auto currBfsEntry = bfsQueue.front();
        bfsQueue.pop();
        int currIdx = currBfsEntry->currIdx;
        Prism* p = &prisms.at(currIdx);

        // Check if cuts pattern consistent with neighbors.
        PossibleCuts possibleCuts{};
        for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
            int neighborIdx = p->neighbors.at(neighborIdxLocal);
            if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                Prism* neighbor = &prisms.at(neighborIdx);
                int neighborFaceIdx = p->neighborFaceIndices.at(neighborIdxLocal) % 3;
                auto* neighborCuts = &neighbor->cuts;
                auto it = currBfsEntry->tetToCutsMap.find(neighborIdx);
                if (it != currBfsEntry->tetToCutsMap.end()) {
                    neighborCuts = &it->second;
                }
                possibleCuts.update(neighborIdxLocal, 1u - neighborCuts->getCut(neighborFaceIdx));
            }
        }
        auto consistentCut = possibleCuts.getConsistentCuts();
        if (consistentCut) {
            currBfsEntry->tetToCutsMap[currIdx] = consistentCut.value();
            //p->cuts = consistentCut.value();
            validRipplePath = currBfsEntry;
            break;
        } else {
            bool existsFlippableNeighbor = false;
            for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
                int neighborIdx = p->neighbors.at(neighborIdxLocal);
                if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                    Prism* neighbor = &prisms.at(neighborIdx);
                    int neighborFaceIdx = p->neighborFaceIndices.at(neighborIdxLocal) % 3;
                    auto newCut = neighbor->cuts;
                    auto it = currBfsEntry->tetToCutsMap.find(neighborIdx);
                    if (it != currBfsEntry->tetToCutsMap.end()) {
                        newCut = it->second;
                    }
                    newCut.setCut(neighborFaceIdx, 1u - newCut.getCut(neighborFaceIdx));
                    if (newCut.isValid()) {
                        existsFlippableNeighbor = true;
                        // Flip cut shared with p and assign opposite cut to p.
                        currBfsEntry->tetToCutsMap[neighborIdx] = newCut;
                        //neighbor->cuts = newCut;
                        possibleCuts.update(neighborIdxLocal, 1u - newCut.getCut(neighborFaceIdx));
                        currBfsEntry->tetToCutsMap[currIdx] = possibleCuts.getConsistentCuts().value();
                        //p->cuts = possibleCuts.getConsistentCuts().value();
                        break;
                    }
                    newCut.setCut(neighborFaceIdx, 1u - newCut.getCut(neighborFaceIdx));
                    possibleCuts.update(neighborIdxLocal, 1u - newCut.getCut(neighborFaceIdx));
                }
            }
            if (!existsFlippableNeighbor) {
                // Use rippling algorithm.
                for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
                    int neighborIdx = p->neighbors.at(neighborIdxLocal);
                    if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                        Prism* neighbor = &prisms.at(neighborIdx);
                        int neighborFaceIdx = p->neighborFaceIndices.at(neighborIdxLocal) % 3;

                        auto cutSelf = p->cuts;
                        auto cutNeig = neighbor->cuts;
                        auto itSelf = currBfsEntry->tetToCutsMap.find(currIdx);
                        auto itNeig = currBfsEntry->tetToCutsMap.find(neighborIdx);
                        if (itSelf != currBfsEntry->tetToCutsMap.end()) {
                            cutSelf = itSelf->second;
                        }
                        if (itNeig != currBfsEntry->tetToCutsMap.end()) {
                            cutNeig = itNeig->second;
                        }
                        cutSelf.setCut(neighborIdxLocal, 1u - cutSelf.getCut(neighborIdxLocal));
                        cutNeig.setCut(neighborFaceIdx, 1u - cutNeig.getCut(neighborFaceIdx));

                        auto neighborRippleEntry = std::make_shared<BfsEntry>(*currBfsEntry);
                        neighborRippleEntry->currIdx = neighborIdx;
                        neighborRippleEntry->tetToCutsMap[currIdx] = cutSelf;
                        neighborRippleEntry->tetToCutsMap[neighborIdx] = cutNeig;
                        bfsQueue.push(neighborRippleEntry);
                    }
                }
            } else {
                validRipplePath = currBfsEntry;
                break;
            }
        }
    }

    if (!validRipplePath) {
        throw std::runtime_error("Error: Rippling algorithm did not find a solution.");
    }

    for (auto& it : validRipplePath->tetToCutsMap) {
        prisms.at(it.first).cuts = it.second;
    }
}

bool FlipSolver::solve(std::vector<Prism>& prisms) {
    // Use algorithm from Figure 14 from paper mentioned in the header file.
    std::queue<int> Q;
    std::vector<bool> prismVisitedArray(prisms.size(), false);
    Q.push(0);
    while (!Q.empty()) {
        int currIdx = Q.front();
        Prism* p = &prisms.at(currIdx);
        Q.pop();
        prismVisitedArray.at(currIdx) = true;

        bool isNeighborTesselated = false;
        for (int neighborIdx : p->neighbors) {
            if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                isNeighborTesselated = true;
                break;
            }
        }
        if (!isNeighborTesselated) {
            // Pick random tesselation pattern.
            p->cuts = Cuts(0, 0, 1);
        } else {
            // Check if cuts pattern consistent with neighbors.
            PossibleCuts possibleCuts{};
            for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
                int neighborIdx = p->neighbors.at(neighborIdxLocal);
                if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                    Prism* neighbor = &prisms.at(neighborIdx);
                    int neighborFaceIdx = p->neighborFaceIndices.at(neighborIdxLocal) % 3;
                    possibleCuts.update(neighborIdxLocal, 1u - neighbor->cuts.getCut(neighborFaceIdx));
                }
            }
            auto consistentCut = possibleCuts.getConsistentCuts();
            if (consistentCut) {
                p->cuts = consistentCut.value();
            } else {
                bool existsFlippableNeighbor = false;
                for (int neighborIdxLocal = 0; neighborIdxLocal < 3; neighborIdxLocal++) {
                    int neighborIdx = p->neighbors.at(neighborIdxLocal);
                    if (neighborIdx >= 0 && prismVisitedArray.at(neighborIdx)) {
                        Prism* neighbor = &prisms.at(neighborIdx);
                        int neighborFaceIdx = p->neighborFaceIndices.at(neighborIdxLocal) % 3;
                        auto newCut = neighbor->cuts;
                        newCut.setCut(neighborFaceIdx, 1u - newCut.getCut(neighborFaceIdx));
                        if (newCut.isValid()) {
                            existsFlippableNeighbor = true;
                            // Flip cut shared with p and assign opposite cut to p.
                            neighbor->cuts = newCut;
                            possibleCuts.update(neighborIdxLocal, 1u - newCut.getCut(neighborFaceIdx));
                            p->cuts = possibleCuts.getConsistentCuts().value();
                            break;
                        }
                        possibleCuts.update(neighborIdxLocal, 1u - neighbor->cuts.getCut(neighborFaceIdx));
                    }
                }
                if (!existsFlippableNeighbor) {
                    // Use rippling algorithm.
                    p->cuts = possibleCuts.getAnyConsistentCut();
                    doRippling(prisms, currIdx, prismVisitedArray);
                    //throw std::runtime_error("Error: Rippling algorithm not yet implemented.");
                }
            }
        }
        for (int neighborIdx : p->neighbors) {
            if (neighborIdx >= 0 && !prismVisitedArray.at(neighborIdx)) {
                Q.push(neighborIdx);
            }
        }
    }

    return true;
}
