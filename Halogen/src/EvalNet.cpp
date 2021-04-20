#include "EvalNet.h"

using namespace UnitTestEvalNet;

int EvaluatePositionNet(const Position& position, EvalCacheTable& evalTable)
{
    int eval;

    if (!evalTable.GetEntry(position.GetZobristKey(), eval))
    {
        eval = position.GetEvaluation();

        TempoAdjustment(eval, position);
        ComplexityAdjustment(eval, position);

        evalTable.AddEntry(position.GetZobristKey(), eval);
    }

    return std::min<int>(EVAL_MAX, std::max<int>(EVAL_MIN, eval));
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

void TempoAdjustment(int& eval, const Position& position)
{
    constexpr static int TEMPO = 10;
    eval += position.GetTurn() == WHITE ? TEMPO : -TEMPO;
}

void ComplexityAdjustment(int& eval, const Position& position)
{
    static constexpr int PhaseValues[] = { 0, 1, 1, 2, 4, 0 };
    static constexpr int PieceValues[] = { 0, 3, 3, 5, 9, 0 };

    //not actual max due to promotions!
    constexpr int maxPhase = PhaseValues[KNIGHT] * 4 + PhaseValues[BISHOP] * 4 + PhaseValues[ROOK] * 4 + PhaseValues[QUEEN] * 2;

    int phase = 0;
    int nonPawnMaterial = 0;

    for (int i = KNIGHT; i <= QUEEN; i++)
    {
        int white = GetBitCount(position.GetPieceBB(static_cast<PieceTypes>(i), WHITE));
        int black = GetBitCount(position.GetPieceBB(static_cast<PieceTypes>(i), BLACK));
        phase += PhaseValues[i] * (white + black);
        nonPawnMaterial += PieceValues[i] * (white - black);
    }

    phase = (phase * 256 + (maxPhase / 2)) / maxPhase;

    //phase now represents a value from 0 for king-pawn endgames and 256 for the opening position.

    Players stronger = eval > 0 ? WHITE : BLACK;

    int complexity;

    if (abs(nonPawnMaterial) < 4)
        complexity = GetBitCount(position.GetPieceBB(PAWN, stronger)) * 32;
    else
        complexity = 256;

    int scale = complexity + (256 - complexity) * (phase) / 256;
    eval = eval * scale / 256;
}

}



