#include <array>
#include <cstdlib>

template <typename OUT, typename IN, size_t SIZE>
OUT dot_product(const std::array<IN, SIZE>& lhs, const std::array<IN, SIZE>& rhs)
{
    OUT result = 0;
    for (size_t i = 0; i < SIZE; i++)
    {
        result += lhs[i] * rhs[i];
    }
    return result;
}