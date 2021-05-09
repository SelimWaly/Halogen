#pragma once
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <assert.h>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstring>
#include "EvalCache.h"
#include "BitBoardDefine.h"

constexpr size_t INPUT_NEURONS = 12 * 64;
constexpr size_t HIDDEN_NEURONS = 512;

struct deltaArray
{
    struct deltaPoint
    {
        int16_t index;
        int16_t delta;
    };

    int8_t size;
    deltaPoint deltas[4];
};

class Network
{
public:
    void RecalculateIncremental(const std::array<bool, INPUT_NEURONS>& inputs);
    void ApplyDelta(const deltaArray& update);  //incrementally update the connections between input layer and first hidden layer
    void ApplyInverseDelta();                   //for un-make moves
    int16_t QuickEval() const;                  //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

    static void Init();

private:
    std::vector<std::array<float, HIDDEN_NEURONS>> Zeta;

    static std::array<std::array<float, HIDDEN_NEURONS>, INPUT_NEURONS> hiddenWeights;
    static std::array<float, HIDDEN_NEURONS> hiddenBias;
    static std::array<float, HIDDEN_NEURONS> outputWeights;
    static float outputBias;
};

