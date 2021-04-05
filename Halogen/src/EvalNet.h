#pragma once
#include "Position.h"
#include <functional>
#include <valarray>
#include <array>
#include <algorithm>

bool DeadPosition(const Position& position);
int EvaluatePositionNet(const Position& position, EvalCacheTable& evalTable);

namespace UnitTestEvalNet
{
    int TempoAdjustment(const Position& position);
    void NoPawnAdjustment(int& eval, const Position& position);
	int MobilityAdjustment(const Position& position);
}

