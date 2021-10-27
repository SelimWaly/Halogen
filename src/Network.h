#pragma once
#include "BitBoardDefine.h"
#include "EvalCache.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

constexpr size_t INPUT_NEURONS = 12 * 64;
constexpr size_t HIDDEN_NEURONS = 512;

class Network
{
public:
    enum class Toggle
    {
        Add,
        Remove
    };

    void RecalculateIncremental(const std::array<int16_t, INPUT_NEURONS>& inputs);

    // calculates starting from the first hidden layer and skips input -> hidden
    int16_t Eval() const;

    // call DoMove and then update inputs as required
    void DoMove();

    template <Toggle update>
    void UpdateInput(Square square, Pieces piece)
    {
        assert(square < N_SQUARES);
        assert(piece < N_PIECES);

        size_t index = piece * 64 + square;

        if constexpr (update == Toggle::Add)
            for (size_t j = 0; j < HIDDEN_NEURONS; j++)
                Zeta.back()[j] += hiddenWeights[index][j];
        if constexpr (update == Toggle::Remove)
            for (size_t j = 0; j < HIDDEN_NEURONS; j++)
                Zeta.back()[j] -= hiddenWeights[index][j];
    }

    // do undo the last move
    void UndoMove();

    static void Init();

private:
    std::vector<std::array<int16_t, HIDDEN_NEURONS>> Zeta;

    static std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> hiddenWeights;
    static std::array<int16_t, HIDDEN_NEURONS> hiddenBias;
    static std::array<int16_t, HIDDEN_NEURONS> outputWeights;
    static int16_t outputBias;
};
