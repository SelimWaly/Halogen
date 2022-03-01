#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "../EvalNet.h"
#include "../MoveGeneration.h"
#include "../Position.h"
#include "../Search.h"
#include "../SearchData.h"
#include "TrainableNetwork.h"
#include "td-leaf-learn.h"

void SelfPlayGame(TrainableNetwork& network, ThreadSharedData& data);
void PrintNetworkDiagnostics(TrainableNetwork& network);

void learn()
{
    TrainableNetwork network;
    network.InitializeWeightsRandomly();

    SearchLimits limits;
    limits.SetDepthLimit(4);
    ThreadSharedData data(std::move(limits));

    std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    for (int games = 0; games < 10000000; games++)
    {
        SelfPlayGame(network, data);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 10)
        {
            last_print = std::chrono::steady_clock::now();

            std::cout << "Game " << games << std::endl;
            PrintNetworkDiagnostics(network);
            auto duration = std::chrono::duration<float>(now - start).count();
            std::cout << "Games per second: " << games / duration << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }

        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count() >= 15)
        {
            last_save = std::chrono::steady_clock::now();

            network.SaveWeights("768-1_g" + std::to_string(games) + ".nn");
        }
    }
}

void PrintNetworkDiagnostics(TrainableNetwork& network)
{
    for (int i = 0; i < N_PIECES; i++)
    {
        float sum = 0;

        for (int j = 0; j < N_SQUARES; j++)
        {
            sum += network.l1_weight[i * 64 + j][0];

            std::cout << network.l1_weight[i * 64 + j][0] << " ";

            if (j % N_FILES == N_FILES - 1)
            {
                std::cout << std::endl;
            }
        }

        sum /= N_SQUARES;

        std::cout << "piece " << i << ": " << sum << std::endl;
        std::cout << std::endl;
    }

    std::cout << "bias: " << network.l1_bias[0] << std::endl;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(2.5 / 400.0 * -x));
}

struct TD_game_result
{
    float score;
    std::vector<int> sparseInputs = {};
    float difference = 0; // between this and next
};

void SelfPlayGame(TrainableNetwork& network, ThreadSharedData& data)
{
    // std::chrono::steady_clock::time_point fn_begin = std::chrono::steady_clock::now();
    // uint64_t time_spend_in_search_ns = 0;

    static std::mt19937 gen(0);
    static std::binomial_distribution<bool> turn(1);

    Position position;
    auto& searchData = data.GetData(0);

    std::vector<TD_game_result> results;

    int turns = 0;
    while (true)
    {
        turns++;

        // check for a terminal position

        // checkmate or stalemate
        BasicMoveList moves;
        LegalMoves(position, moves);
        if (moves.size() == 0)
        {
            if (IsInCheck(position))
            {
                results.push_back({ position.GetTurn() == WHITE ? 0.f : 1.f });
            }
            else
            {
                results.push_back({ 0.5f });
            }

            break;
        }

        // 50 move rule
        if (position.GetFiftyMoveCount() >= 100)
        {
            results.push_back({ 0.5f });
            break;
        }

        // 3 fold repitition rule
        if (position.CheckForRep(0, 3))
        {
            results.push_back({ 0.5f });
            break;
        }

        // insufficent material rule
        if (DeadPosition(position))
        {
            results.push_back({ 0.5f });
            break;
        }

        // apply a random opening if at the start of the game and it's not over yet
        if (turns < 10)
        {
            std::uniform_int_distribution<> move(0, moves.size() - 1);
            position.ApplyMove(moves[move(gen)]);
            continue;
        }

        // -----------------------------

        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        SearchThread(position, data);
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // time_spend_in_search_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        const auto& pv = searchData.PvTable[0];

        for (size_t i = 0; i < pv.size(); i++)
        {
            position.ApplyMove(pv[i]);
        }

        /*
        You could arguably use either the eval of the end of the pv or the score returned by the search.
        In theory for basic alpha-beta search these would be equal. That is not the case in practice due to
        search instability (transposition table etc) and search score adjustments (checkmate, draw randomness etc).
        */
        results.push_back({ sigmoid(position.GetEvaluation()), network.GetSparseInputs(position) });

        for (size_t i = 0; i < pv.size(); i++)
        {
            position.RevertMove();
        }

        position.ApplyMove(data.GetBestMove());
    }

    for (int i = 0; i < static_cast<int>(results.size()) - 1; i++)
    {
        results[i].difference = results[i + 1].score - results[i].score;
        // std::cout << "difference: " << results[i].difference << " scores: " << results[i].score << ", " << results[i + 1].score << std::endl;
    }

    // main td-leaf update step:

    for (int t = 0; t < static_cast<int>(results.size()) - 1; t++)
    {
        float discounted_td = 0;

        for (int j = t; j < static_cast<int>(results.size()) - 1; j++)
        {
            discounted_td += results[j].difference * pow(0.7f, j - t);
        }

        // for simple case, all activated inputs make for a gradient of 1 so math is easy.
        // note derivative of sigmoid with coefficent k is k*(s)*(1-s)
        discounted_td *= results[t].score * (1 - results[t].score) * 2.5 / 400.0;

        discounted_td *= 160; // learning rate

        for (size_t i = 0; i < results[t].sparseInputs.size(); i++)
        {
            network.l1_weight[results[t].sparseInputs[i]][0] += discounted_td;
        }
        network.l1_bias[0] += discounted_td;
    }

    // std::cout << "Game result: " << results.back().score << " turns: " << turns << std::endl;

    // std::chrono::steady_clock::time_point fn_end = std::chrono::steady_clock::now();
    // auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(fn_end - fn_begin).count();
    // std::cout << "Time spend in search: " << static_cast<double>(time_spend_in_search_ns) / static_cast<double>(total_time) * 100 << "%" << std::endl;
}