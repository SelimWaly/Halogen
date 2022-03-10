#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <thread>

#include "../EvalNet.h"
#include "../MoveGeneration.h"
#include "../Position.h"
#include "../Search.h"
#include "../SearchData.h"
#include "TrainableNetwork.h"
#include "td-leaf-learn.h"

void SelfPlayGame(TrainableNetwork& network, ThreadSharedData& data);
void PrintNetworkDiagnostics(TrainableNetwork& network);

// hyperparameters
constexpr double LAMBDA = 0.7; // credit discount factor
constexpr double GAMMA = 1; // discount rate of future rewards

constexpr int training_depth = 4;
constexpr double sigmoid_coeff = 2.5 / 400.0;
// -----------------

constexpr int max_threads = 11;

std::atomic<int> game_count = 0;

void learn_thread(TrainableNetwork& network)
{
    SearchLimits limits;
    limits.SetDepthLimit(training_depth);
    ThreadSharedData data(std::move(limits));

    while (true)
    {
        SelfPlayGame(network, data);
        game_count++;
    }
}

void info_thread(TrainableNetwork& network)
{
    std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_save = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    while (true)
    {
        // wake up once a second and print diagnostics if needed.
        // by sleeping, we avoid spinning endlessly and contesting shared memory.
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 10)
        {
            last_print = std::chrono::steady_clock::now();

            std::cout << "Game " << game_count << std::endl;
            network.PrintNetworkDiagnostics();
            auto duration = std::chrono::duration<float>(now - start).count();
            std::cout << "Games per second: " << game_count / duration << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }

        if (std::chrono::duration_cast<std::chrono::minutes>(now - last_save).count() >= 15)
        {
            last_save = std::chrono::steady_clock::now();

            network.SaveWeights("768-16-1_g" + std::to_string(game_count) + ".nn");
        }
    }
}

void learn()
{
    TrainableNetwork network;
    network.InitializeWeightsRandomly();

    std::vector<std::thread> threads;

    threads.emplace_back(info_thread, std::ref(network));

    // always have at least one learning and one info thread.
    // at most we want max_threadsm total threads.
    for (int i = 0; i < std::max(1, max_threads - 1); i++)
    {
        threads.emplace_back(learn_thread, std::ref(network));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(sigmoid_coeff * -x));
}

struct TD_game_result
{
    float score;
    Players stm;
    std::array<std::vector<int>, N_PLAYERS> sparseInputs = {};
    double delta = 0; // between this and next
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
                results.push_back({ position.GetTurn() == WHITE ? 0.f : 1.f, position.GetTurn() });
            }
            else
            {
                results.push_back({ 0.5f, position.GetTurn() });
            }

            break;
        }

        // 50 move rule
        if (position.GetFiftyMoveCount() >= 100)
        {
            results.push_back({ 0.5f, position.GetTurn() });
            break;
        }

        // 3 fold repitition rule
        if (position.CheckForRep(0, 3))
        {
            results.push_back({ 0.5f, position.GetTurn() });
            break;
        }

        // insufficent material rule
        if (DeadPosition(position))
        {
            results.push_back({ 0.5f, position.GetTurn() });
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
        results.push_back({ sigmoid(position.GetEvaluation()), position.GetTurn(), network.GetSparseInputs(position) });

        for (size_t i = 0; i < pv.size(); i++)
        {
            position.RevertMove();
        }

        position.ApplyMove(data.GetBestMove());
    }

    for (int i = 0; i < static_cast<int>(results.size()) - 1; i++)
    {
        results[i].delta = GAMMA * results[i + 1].score - results[i].score;
        // std::cout << "difference: " << results[i].difference << " scores: " << results[i].score << ", " << results[i + 1].score << std::endl;
    }

    // main td-leaf update step:

    for (int t = 0; t < static_cast<int>(results.size()) - 1; t++)
    {
        double delta_sum = 0;

        for (int j = t; j < static_cast<int>(results.size()) - 1; j++)
        {
            delta_sum += results[j].delta * pow(LAMBDA * GAMMA, j - t);
        }

        // note derivative of sigmoid with coefficent k is k*(s)*(1-s)
        double loss_gradient = -delta_sum * results[t].score * (1 - results[t].score) * sigmoid_coeff;
        network.Backpropagate(loss_gradient, results[t].sparseInputs, results[t].stm);
    }

    // std::cout << "Game result: " << results.back().score << " turns: " << turns << std::endl;

    // std::chrono::steady_clock::time_point fn_end = std::chrono::steady_clock::now();
    // auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(fn_end - fn_begin).count();
    // std::cout << "Time spend in search: " << static_cast<double>(time_spend_in_search_ns) / static_cast<double>(total_time) * 100 << "%" << std::endl;
}