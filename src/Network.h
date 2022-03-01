#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "BitBoardDefine.h"

constexpr size_t INPUT_NEURONS = 12 * 64;

class Position;

struct Accumulator
{
    float value;
};

class Network
{
public:
    void Recalculate(const Position& position);

    // calculates starting from the first hidden layer and skips input -> hidden
    int16_t Eval(Players stm) const;

    // call and then update inputs as required
    void AccumulatorPush();

    void AddInput(Square square, Pieces piece);
    void RemoveInput(Square square, Pieces piece);

    // do undo the last move
    void AccumulatorPop();

    static void Init();

private:
    std::vector<Accumulator> AccumulatorStack;

    static std::array<float, INPUT_NEURONS> l1_weight;
    static float l1_bias;
};
