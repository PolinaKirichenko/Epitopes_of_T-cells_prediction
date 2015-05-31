#include <algorithm>
#include <cmath>
#include <vector>
#include "hmm.h"

typedef std::vector<std::vector<double>> Matrix;

template <typename T>
std::ostream& operator << (std::ostream& o, const std::vector<T>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        o << v[i]<< " ";
    o << std::endl;
    return o;
}

std::ostream& operator << (std::ostream& o, const Matrix& M) {
    for (size_t i = 0; i < M.size(); ++i)
        o << M[i];
    return o;
}

void HMM::scale(std::vector<double>& v, double c) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] /= c;
}

void HMM::ForwardProcedure(const std::vector<int>& Y, std::vector<double>& scaling_numbers) {
    int T = Y.size();
    for (size_t j = 0; j < n; ++j)
        alpha[0][j] = initial_distribution[j] * observation[Y[0]][j];
    scaling_numbers[0] = accumulate(alpha[0].begin(), alpha[0].end(), 0.0);
    scale(alpha[0], scaling_numbers[0]);

    for (size_t t = 1; t < T; ++t) {
        for (size_t j = 0; j < n; ++j) {
            alpha[t][j] = 0;
            for (size_t i = 0; i < n; ++i)
                alpha[t][j] += alpha[t - 1][i] * transition[i][j];
            alpha[t][j] *= observation[Y[t]][j];
        }
        scaling_numbers[t] = accumulate(alpha[t].begin(), alpha[t].end(), 0.0);
        scale(alpha[t], scaling_numbers[t]);
    }
    //std::cout << "ALPHA" << std::endl << alpha << std::endl;
}

void HMM::BackwardProcedure(const std::vector<int>& Y, std::vector<double>& scaling_numbers) {
    int T = Y.size();
    for (size_t j = 0; j < n; ++j)
        beta[T - 1][j] = 1;
    scale(beta[T - 1], scaling_numbers[T - 1]);

    for (int t = T - 2; t >= 0; --t) {
        for (size_t i = 0; i < n; ++i) {
            beta[t][i] = 0;
            for (size_t j = 0; j < n; ++j)
                beta[t][i] += beta[t + 1][j] * transition[i][j] * observation[Y[t + 1]][j];
        }
        scale(beta[t], scaling_numbers[t]);
    }
    //std::cout << "BETA" << std::endl << beta << std::endl;
}

void HMM::CalculateGamma(int T, int k) {
    double denom;
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < n; ++i) {
            denom = 0;
            for (size_t j = 0; j < n; ++j)
                denom += alpha[t][j] * beta[t][j];
            gamma[k][t][i] = alpha[t][i] * beta[t][i] / denom;
        }
    //std::cout << "GAMMA" << std::endl << gamma << std::endl;
}

void HMM::CalculateXi(const std::vector<int>& Y, int k) {
    double denom;
    for (size_t t = 0; t < Y.size() - 1; ++t) {
        denom = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                denom += alpha[t][i] * transition[i][j] * beta[t + 1][j] * observation[Y[t + 1]][j];
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                xi[k][t][i][j] = alpha[t][i] * transition[i][j] * beta[t + 1][j] * observation[Y[t + 1]][j] / denom;
    }
    /*std::cout << "XI" << std::endl;
    for (size_t t = 0; t < T - 1; ++t)
        std::cout << xi[t] << std::endl;*/
}

void HMM::UpdateTransition(const std::vector<std::vector<int>>& training_set) {
    int K = training_set.size();
    double num, denom;
    for (size_t i = 0; i < n; ++i) {
        denom = 0;
        for (size_t k = 0; k < K; ++k)
            for (size_t t = 0; t < training_set[k].size() - 1; ++t)
                denom += gamma[k][t][i];
        for (size_t j = 0; j < n; ++j) {
            num = 0;
            for (size_t k = 0; k < K; ++k)
                for (size_t t = 0; t < training_set[k].size() - 1; ++t)
                    num += xi[k][t][i][j];
            transition[i][j] = 0.1 * transition[i][j] + 0.9 * num / denom;
        }
    }
}

void HMM::UpdateObservation(const std::vector<std::vector<int>>& training_set) {
    int K = training_set.size();
    double num, denom;
    for (size_t i = 0; i < n; ++i) {
        denom = 0;
        for (size_t k = 0; k < K; ++k)
            for (size_t t = 0; t < training_set[k].size(); ++t)
                denom += gamma[k][t][i];
        for (size_t j = 0; j < m; ++j) {
            num = 0;
            for (size_t k = 0; k < K; ++k)
                for (size_t t = 0; t < training_set[k].size(); ++t)
                    if (training_set[k][t] == j)
                        num += gamma[k][t][i];
            observation[j][i] = 0.1 * observation[j][i] + 0.9 * num / denom;
        }
    }
}

double HMM::CalculateLogProbability(const std::vector<double>& scaling_numbers) {
    double logprobability = 0;
    for (size_t i = 0; i < scaling_numbers.size(); ++i)
        logprobability += log(scaling_numbers[i]);
    return logprobability;
}

HMM::HMM(const Matrix& A, const Matrix& B, const std::vector<double>& p0) : transition(A), observation(B), initial_distribution(p0) {
    n = initial_distribution.size();
    m = observation.size();
}

void HMM::ShowModelParameters(std::ostream& out) {
    out << "INITIAL DISTRIBUTION" << std::endl << initial_distribution << std::endl;
    out << "TRANSITION" << std::endl << transition << std::endl;
    out << "OBSERVATION PROBABILITY" << std::endl << observation << std::endl;
}

double HMM::SequenceLogProbability(const std::vector<int>& sequence) {
    int T = sequence.size();
    alpha = Matrix(T, std::vector<double>(n));
    std::vector<double> scaling_numbers(T);
    ForwardProcedure(sequence, scaling_numbers);
    return CalculateLogProbability(scaling_numbers);
}

/*
std::vector<int> HMM::Apply(const std::vector<int>& sequence) {
    int T = sequence.size();
    alpha = Matrix(T, std::vector<double>(n));
    beta = Matrix(T, std::vector<double>(n));
    gamma = std::vector<Matrix>(1, Matrix(T, std::vector<double>(n)));

    std::vector<double> scaling_numbers(T);
    ForwardProcedure(sequence, scaling_numbers);
    BackwardProcedure(sequence, scaling_numbers);
    CalculateGamma(sequence.size(), 0);

    std::vector<int> most_likely_states(T);
    for (size_t t = 0; t < T; ++t)
        most_likely_states[t] = max_element(gamma[t].begin(), gamma[t].end()) - gamma[t].begin();
    return most_likely_states;
}*/

std::vector<int> HMM::Apply(const std::vector<int>& sequence) {
    int T = sequence.size();
    std::vector<int> most_likely_states(T);
    std::vector<std::vector<int>> tr(T);
    Matrix delta(T, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        delta[0][i] = log(initial_distribution[i] * observation[sequence[0]][i]);
        tr[0][i] = i;
    }
    for (size_t t = 1; t < T; ++t) {
        for (size_t i = 0; i < n; ++i) {
            delta[t][i] = -INFINITY;
            tr[t][i] = 0;
            for (size_t j = 0; j < n; ++j) {
                if (delta[t][i] < delta[t - 1][j] + log(transition[j][i]) + log(observation[sequence[t]][i])) {
                    delta[t][i] = delta[t - 1][j] + log(transition[j][i]) + log(observation[sequence[t]][i]);
                    tr[t][i] = j;
                }
            }
        }
    }
    most_likely_states[T - 1] = max_element(delta[T - 1].begin(), delta[T - 1].end()) - delta[T - 1].begin();
    for (size_t t = T - 1; t > 0; --t)
        most_likely_states[t - 1] = tr[t][most_likely_states[t]];
    return most_likely_states;
}

void HMM::Train(const std::vector<std::vector<int>>& training_set) {
    int T;
    int K = training_set.size();
    gamma = std::vector<Matrix>(K);
    xi = std::vector<std::vector<Matrix>>(K);

    for (size_t k = 0; k < K; ++k) {
        T = training_set[k].size();
        alpha = Matrix(T, std::vector<double>(n));
        beta = Matrix(T, std::vector<double>(n));
        gamma[k] = Matrix(T, std::vector<double>(n));
        xi[k] = std::vector<Matrix>(T - 1, Matrix(n, std::vector<double>(n)));
        std::vector<double> scaling_numbers(T);

        ForwardProcedure(training_set[k], scaling_numbers);
        BackwardProcedure(training_set[k], scaling_numbers);
        CalculateGamma(T, k);
        CalculateXi(training_set[k], k);
    }

    for (size_t i = 0; i < n; ++i) {
        initial_distribution[i] = 0;
        for (size_t k = 0; k < K; ++k)
            initial_distribution[i] += gamma[k][0][i];
        initial_distribution[i] /= K;
    }
    UpdateTransition(training_set);
    UpdateObservation(training_set);
}
