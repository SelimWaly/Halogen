#include "TTEntry.h"

TTEntry::TTEntry()
{
	Reset();
}

TTEntry::TTEntry(Move best, uint64_t ZobristKey, int Score, int Depth, int currentTurnCount, int distanceFromRoot, EntryType Cutoff)
{
	assert(Score < SHRT_MAX && Score > SHRT_MIN);
	assert(Depth < CHAR_MAX && Depth > CHAR_MIN);

	lowerkey = ZobristKey & 0xffff;	
	higherkey = (ZobristKey >> 16) & 0xffff;
	bestMove = best;
	score = static_cast<short>(Score);
	depth = static_cast<char>(Depth);
	cutoff = Cutoff;
	SetHalfMove(currentTurnCount, distanceFromRoot);
}

void TTEntry::MateScoreAdjustment(int distanceFromRoot)
{
	//checkmate node or TB win/loss
	if (score > EVAL_MAX)	
		score -= static_cast<short>(distanceFromRoot);
	if (score < EVAL_MIN)
		score += static_cast<short>(distanceFromRoot);
}

void TTEntry::Reset()
{
	bestMove = Move();
	lowerkey = 0;
	higherkey = 0;
	score = -1;
	depth = -1;
	cutoff = EntryType::EMPTY_ENTRY;
	halfmove = -1;
}

void TTBucket::Reset()
{
	for (auto& val : entry)
		val.Reset();
}
