#pragma once
#include "Position.h"
#include "EvalNet.h"
#include "MoveList.h"

template <typename T> void LegalMoves(Position& position, FixedVector<T>& moves);
template <typename T> void QuiescenceMoves(Position& position, FixedVector<T>& moves);
template <typename T> void QuietMoves(Position& position, FixedVector<T>& moves);

bool IsInCheck(const Position& position, Players colour);
bool IsInCheck(const Position& position);

bool MoveIsLegal(Position& position, const Move& move);

template <PieceTypes pieceType>
uint64_t AttackBB(Square sq, uint64_t occupied = EMPTY);

#include "MoveGeneration.tpp"