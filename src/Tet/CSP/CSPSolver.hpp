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

#ifndef TESTOVM_CSPSOLVER_HPP
#define TESTOVM_CSPSOLVER_HPP

#include <vector>
#include <array>
#include <cstdint>

/**
 * Cuts of the faces of a prism.
 */
struct Cuts {
    uint32_t bitfield;
    Cuts() : bitfield(0) {}
    Cuts(uint32_t cut0, uint32_t cut1, uint32_t cut2) {
        bitfield = cut0 | (cut1 << 1u) | (cut2 << 2u);
    }
    void setCut(uint32_t idx, uint32_t cut) {
        bitfield = (bitfield & ~(1u << idx)) | (cut << idx);
    }
    [[nodiscard]] uint32_t getCut(uint32_t idx) const {
        return (bitfield >> idx) & 1u;
    }
    [[nodiscard]] bool isValid() const {
        return bitfield != 0b000 && bitfield != 0b111;
    }
};

struct Prism {
    std::array<int, 3> neighborFaceIndices{};
    std::array<int, 3> neighbors{};
    Cuts cuts;
};

/**
 * Constraint satisfaction problem (CSP) for the prism tetrahedralization problem described in:
 * - Tetrahedralization of a Hexahedral Mesh. Aman Timalsina and Matthew Knepley, 2023.
 * https://arxiv.org/abs/2208.07128
 * - The Adaptive Thin Shell Tetrahedral Mesh. Kenny Erleben, Henrik Dohlmann and Jon Sporring, 2005.
 */
class CSPSolver {
public:
    virtual ~CSPSolver() = default;
    virtual bool solve(std::vector<Prism>& prisms) = 0;
};

/// Checks whether the prism cut CSP criterion is fulfilled.
bool checkIsCspFulfilled(const std::vector<Prism>& prisms);

/// Writes the CSP graph to std::out for debugging purposes.
bool writeGraphviz(const std::vector<Prism>& prisms);

#endif //TESTOVM_CSPSOLVER_HPP
