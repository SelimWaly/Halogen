#include "EvalNet.h"
#include "MoveGeneration.h"

using namespace UnitTestEvalNet;

int EvaluatePositionNet(const Position& position, EvalCacheTable& evalTable)
{
    int eval;

    if (!evalTable.GetEntry(position.GetZobristKey(), eval))
    {
        eval = position.GetEvaluation();

        NoPawnAdjustment(eval, position);
        eval += TempoAdjustment(position);
        eval += MobilityAdjustment(position);

        evalTable.AddEntry(position.GetZobristKey(), eval);
    }

    return std::min(4000, std::max(-4000, eval));
}

bool DeadPosition(const Position& position)
{
    if ((position.GetPieceBB(WHITE_PAWN)) != 0) return false;
    if ((position.GetPieceBB(WHITE_ROOK)) != 0) return false;
    if ((position.GetPieceBB(WHITE_QUEEN)) != 0) return false;

    if ((position.GetPieceBB(BLACK_PAWN)) != 0) return false;
    if ((position.GetPieceBB(BLACK_ROOK)) != 0) return false;
    if ((position.GetPieceBB(BLACK_QUEEN)) != 0) return false;

    /*
    From the Chess Programming Wiki:
        According to the rules of a dead position, Article 5.2 b, when there is no possibility of checkmate for either side with any series of legal moves, the position is an immediate draw if
        - both sides have a bare king													1.
        - one side has a king and a minor piece against a bare king						2.
        - both sides have a king and a bishop, the bishops being the same color			Not covered
    */

    //We know the board must contain just knights, bishops and kings
    int WhiteBishops = GetBitCount(position.GetPieceBB(WHITE_BISHOP));
    int BlackBishops = GetBitCount(position.GetPieceBB(BLACK_BISHOP));
    int WhiteKnights = GetBitCount(position.GetPieceBB(WHITE_KNIGHT));
    int BlackKnights = GetBitCount(position.GetPieceBB(BLACK_KNIGHT));
    int WhiteMinor = WhiteBishops + WhiteKnights;
    int BlackMinor = BlackBishops + BlackKnights;

    if (WhiteMinor == 0 && BlackMinor == 0) return true;	//1
    if (WhiteMinor == 1 && BlackMinor == 0) return true;	//2
    if (WhiteMinor == 0 && BlackMinor == 1) return true;	//2

    return false;
}

namespace UnitTestEvalNet
{

int TempoAdjustment(const Position& position)
{
    constexpr static int TEMPO = 10;
    return position.GetTurn() == WHITE ? TEMPO : -TEMPO;
}

void NoPawnAdjustment(int& eval, const Position& position)
{
    if (eval > 0 && position.GetPieceBB(PAWN, WHITE) == 0)
        eval /= 2;
    if (eval < 0 && position.GetPieceBB(PAWN, BLACK) == 0)
        eval /= 2;
}

int MobilityAdjustment(const Position& position)
{
    static constexpr int MobilityScore[] = { 0, 1, 1, 1, 1, 0 };

    uint64_t wKnight = position.GetPieceBB(WHITE_KNIGHT);
    uint64_t bKnight = position.GetPieceBB(BLACK_KNIGHT);
    uint64_t wBishop = position.GetPieceBB(WHITE_BISHOP);
    uint64_t bBishop = position.GetPieceBB(BLACK_BISHOP);

    uint64_t occupancy = position.GetAllPieces();

    int score = 0;

    while (wKnight)
    {
        Square square = static_cast<Square>(LSBpop(wKnight));
        uint64_t attacks = AttackBB<KNIGHT>(square, occupancy) & ~occupancy;
        score += GetBitCount(attacks) * MobilityScore[KNIGHT];
    }

    while (bKnight)
    {
        Square square = static_cast<Square>(LSBpop(bKnight));
        uint64_t attacks = AttackBB<KNIGHT>(square, occupancy) & ~occupancy;
        score -= GetBitCount(attacks) * MobilityScore[KNIGHT];
    }

    while (wBishop)
    {
        Square square = static_cast<Square>(LSBpop(wBishop));
        uint64_t attacks = AttackBB<BISHOP>(square, occupancy) & ~occupancy;
        score += GetBitCount(attacks) * MobilityScore[BISHOP];
    }

    while (bBishop)
    {
        Square square = static_cast<Square>(LSBpop(bBishop));
        uint64_t attacks = AttackBB<BISHOP>(square, occupancy) & ~occupancy;
        score -= GetBitCount(attacks) * MobilityScore[BISHOP];
    }

    return score;
}

}



