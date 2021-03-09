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

enum
{
    INPUT_LAYER,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    OUTPUT_LAYER,

    N_LAYERS
};

constexpr size_t ARCHITECTURE[N_LAYERS] = { 768, 256, 32, 1 };

constexpr int16_t MAX_VALUE = 128;
constexpr int16_t PRECISION = ((size_t)std::numeric_limits<int16_t>::max() + 1) / MAX_VALUE;
constexpr int32_t SQUARE_PRECISION = (int32_t)PRECISION * PRECISION;

struct deltaArray
{
    struct deltaPoint
    {
        size_t index;
        int16_t delta;
    };

    size_t size;
    deltaPoint deltas[4];
};

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
class InputLayer
{
public:
    void Init(float*& data);

    void RecalculateIncremental(std::array<T_in, INPUT> inputs, std::vector<std::array<T_out, OUTPUT>>& Zeta) const;
    void ApplyDelta(const deltaArray& update, std::vector<std::array<T_out, OUTPUT>>& Zeta) const;
    void ApplyInverseDelta(std::vector<std::array<T_out, OUTPUT>>& Zeta) const;

    std::array<T_out, OUTPUT> GetActivation(const std::vector<std::array<T_out, OUTPUT>>& Zeta) const;

private:
    std::array<std::array<T_in, OUTPUT>, INPUT> weights;
    std::array<T_out, OUTPUT> bias;

    T_out Activation(T_out val) const { return std::max(T_out(0), val); }
};

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
class HiddenLayer
{
public:
    void Init(float*& data);
    std::array<T_out, OUTPUT> FeedForward(const std::array<T_in, INPUT>& input) const;

private:
    std::array<std::array<T_in, INPUT>, OUTPUT> weights;
    std::array<T_out, OUTPUT> bias;

    void Activation(T_out& val) const { val = std::max(T_out(0), val); }
};

template <typename T_in, typename T_out, size_t INPUT>
class OuputLayer
{
public:
    void Init(float*& data);
    T_out FeedForward(const std::array<T_in, INPUT>& input) const;

private:
    std::array<T_in, INPUT> weights;
    T_out bias;
};

class Network
{
public:
    void RecalculateIncremental(std::array<int16_t, ARCHITECTURE[INPUT_LAYER]> inputs);
    void ApplyDelta(const deltaArray& update);  //incrementally update the connections between input layer and first hidden layer
    void ApplyInverseDelta();                   //for un-make moves
    int16_t Eval() const;                       //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

    static void Init();

private:
    std::vector<std::array<int16_t, ARCHITECTURE[HIDDEN_LAYER_1]>> Zeta;

    static InputLayer <int16_t, int16_t, ARCHITECTURE[INPUT_LAYER], ARCHITECTURE[HIDDEN_LAYER_1]> layer1;
    static HiddenLayer <int16_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_1], ARCHITECTURE[HIDDEN_LAYER_2]> layer2;
    static OuputLayer <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_2]> layer3;
};
