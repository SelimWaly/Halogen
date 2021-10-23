#pragma once
#include "Position.h"
#include "EvalCache.h"
#include <functional>
#include <valarray>

bool DeadPosition(const Position& position);
int EvaluatePositionNet(const Position& position, EvalCacheTable& evalTable);
