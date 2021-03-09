#include "Network.h"
#include "epoch5.net"

InputLayer <int16_t, int16_t, ARCHITECTURE[INPUT_LAYER], ARCHITECTURE[HIDDEN_LAYER_1]> Network::layer1;
HiddenLayer <int16_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_1], ARCHITECTURE[HIDDEN_LAYER_2]> Network::layer2;
OuputLayer <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_2]> Network::layer3;

void Network::Init()
{
    auto Data = reinterpret_cast<float*>(label);
    layer1.Init(Data);
    layer2.Init(Data);
    layer3.Init(Data);
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
void InputLayer<T_in, T_out, INPUT, OUTPUT>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_out)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[i][j] = (T_in)round(*data++ * PRECISION);
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
void HiddenLayer<T_in, T_out, INPUT, OUTPUT>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_out)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[j][i] = (T_in)round(*data++ * PRECISION);
}

template <typename T_in, typename T_out, size_t INPUT>
void OuputLayer<T_in, T_out, INPUT>::Init(float*& data)
{
    bias = (T_out)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        weights[i] = (T_in)round(*data++ * PRECISION);
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
void InputLayer<T_in, T_out, INPUT, OUTPUT>::RecalculateIncremental(std::array<T_in, INPUT> inputs, std::vector<std::array<T_out, OUTPUT>>& Zeta) const
{
    Zeta = { bias };

    for (size_t i = 0; i < OUTPUT; i++)
        for (size_t j = 0; j < INPUT; j++)
            Zeta[0][i] += inputs[j] * weights[j][i];
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
void InputLayer<T_in, T_out, INPUT, OUTPUT>::ApplyDelta(const deltaArray& update, std::vector<std::array<T_out, OUTPUT>>& Zeta) const
{
    Zeta.push_back(Zeta.back());

    for (size_t i = 0; i < update.size; i++)
    {
        if (update.deltas[i].delta == 1)
            for (size_t j = 0; j < OUTPUT; j++)
                Zeta.back()[j] += weights[update.deltas[i].index][j];
        else
            for (size_t j = 0; j < OUTPUT; j++)
                Zeta.back()[j] -= weights[update.deltas[i].index][j];
    }
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
void InputLayer<T_in, T_out, INPUT, OUTPUT>::ApplyInverseDelta(std::vector<std::array<T_out, OUTPUT>>& Zeta) const
{
    Zeta.pop_back();
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
std::array<T_out, OUTPUT> InputLayer<T_in, T_out, INPUT, OUTPUT>::GetActivation(const std::vector<std::array<T_out, OUTPUT>>& Zeta) const
{
    std::array<T_out, OUTPUT> ret;

    for (size_t i = 0; i < OUTPUT; i++)
        ret[i] = Activation(Zeta.back()[i]);

    return ret;
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT>
std::array<T_out, OUTPUT> HiddenLayer<T_in, T_out, INPUT, OUTPUT>::FeedForward(const std::array<T_in, INPUT>& input) const
{
    std::array<T_out, OUTPUT> ret = {};

    for (size_t i = 0; i < OUTPUT; i++)
    {
        for (size_t j = 0; j < INPUT; j++)
            ret[i] += input[j] * weights[i][j];

        ret[i] /= PRECISION;
        ret[i] += bias[i];
        Activation(ret[i]);
    }

    return ret;
}


template <typename T_in, typename T_out, size_t INPUT>
T_out OuputLayer<T_in, T_out, INPUT>::FeedForward(const std::array<T_in, INPUT>& input) const
{
    T_out zeta = bias * PRECISION;

    for (size_t j = 0; j < INPUT; j++)
        zeta += input[j] * weights[j];

    return zeta;
}

void Network::RecalculateIncremental(std::array<int16_t, ARCHITECTURE[INPUT_LAYER]> inputs)
{
    layer1.RecalculateIncremental(inputs, Zeta);
}

void Network::ApplyDelta(const deltaArray& update)
{
    layer1.ApplyDelta(update, Zeta);
}

void Network::ApplyInverseDelta()
{
    layer1.ApplyInverseDelta(Zeta);
}

int16_t Network::Eval() const
{
    return layer3.FeedForward(layer2.FeedForward(layer1.GetActivation(Zeta))) / SQUARE_PRECISION;
}