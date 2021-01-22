#pragma once
#include "Move.h"
#include "BitBoardDefine.h"
#include <climits>

constexpr unsigned int HALF_MOVE_MODULO = 16;

enum class EntryType : uint8_t {
	EMPTY_ENTRY,
	EXACT,
	LOWERBOUND,
	UPPERBOUND
};

//10 bytes
class TTEntry
{
public:
	TTEntry();
	TTEntry(Move best, uint64_t ZobristKey, int Score, int Depth, int currentTurnCount, int distanceFromRoot, EntryType Cutoff);
	~TTEntry() = default;

	bool IsAncient(unsigned int currentTurnCount, unsigned int distanceFromRoot) const { return halfmove != static_cast<char>((currentTurnCount - distanceFromRoot) % (HALF_MOVE_MODULO)); }

	void SetHalfMove(int currentTurnCount, int distanceFromRoot) { halfmove = CalculateAge(currentTurnCount, distanceFromRoot); }	//halfmove is from current position, distanceFromRoot adjusts this to get what the halfmove was at the root of the search
	void MateScoreAdjustment(int distanceFromRoot);
	void Reset();

	static uint8_t CalculateAge(int currenthalfmove, int distanceFromRoot) { return (currenthalfmove - distanceFromRoot) % (HALF_MOVE_MODULO); }

	int GetScore() const { return score; }
	int GetDepth() const { return depth; }
	EntryType GetCutoff() const { return cutoff; }
	char GetAge() const { return halfmove; }
	Move GetMove() const { return bestMove; }

	bool KeyMatch(uint64_t ZobristKey) const { return static_cast<int32_t>(ZobristKey & 0xffffffff) == (lowerkey + (higherkey << 16)); }

private:
	/*Arranged to minimize padding*/

	//we only store the lowest 32 bits of the 64 bit key
	uint16_t lowerkey;		//2 bytes (bits 0-15)
	uint16_t higherkey;		//2 bytes (bits 16-31)

	Move bestMove;			//2 bytes 
	short int score;		//2 bytes
	char depth;				//1 bytes

	EntryType cutoff : 2;	//2 bits
	uint8_t halfmove : 4;	//4 bits (is stored as the halfmove at the ROOT of this current search, modulo 16)
};

//64 bytes
struct TTBucket
{
	void Reset();

	static const size_t Size = 64 / sizeof(TTEntry);
	std::array<TTEntry, Size> entry;

private:
	std::array<char, 64 - Size * sizeof(TTEntry)> padding;	//add extra bytes to get to 64 if TTentry does not go evenly into 64 bytes
};

