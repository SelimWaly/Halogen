#pragma once

#include "Pyrrhic/tbprobe.h"
#include "MoveGenerator.h"
#include <ctime>
#include <algorithm>
#include <thread>
#include <cmath>

/*Tuneable search constants*/

inline double LMR_constant = -1.61;
inline double LMR_coeff = 1.04;

inline int Null_constant = 5;
inline int Null_depth_quotent = 8;
inline int Null_beta_quotent = 190;

inline int Futility_constant = 64;
inline int Futility_coeff = 55;
inline int Futility_depth = 11;

inline int Aspiration_window = 14;

inline int Delta_margin = 236;

inline int SNMP_coeff = 144;
inline int SNMP_depth = 9;

inline int LMP_constant = 9;
inline int LMP_coeff = 9;
inline int LMP_depth = 4;

/*----------------*/

struct SearchResult
{
	SearchResult(short score, Move move = Move()) : m_score(score), m_move(move) {}

	int GetScore() const { return m_score; }
	Move GetMove() const { return m_move; }

private:
	short m_score;
	Move m_move;
};

uint64_t SearchThread(Position position, SearchParameters parameters, const SearchLimits& limits, bool noOutput = false);