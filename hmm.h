#ifndef HMM_H_INCLUDED
#define HMM_H_INCLUDED

#include <iostream>
#include <vector>

typedef std::vector<std::vector<double>> Matrix;

class HMM {
private:
    std::vector<double> initial_distribution; //initial probability distribution
    Matrix transition; // transition[i][j] is probability of going into state j after state i
    Matrix observation; // observation[i][j] is probability of symbol j in state i
    Matrix alpha; //alpha[t][j] is probability to observe Y1,...Yt and state Xt = j
    Matrix beta; //beta [t][j] is probability to observe Y(t+1),...Yk if Xt = j
    int n, m; // numbers of state, observation symbols and observed symbols respectively
    std::vector<Matrix> gamma;
    std::vector<std::vector<Matrix>> xi;

    void scale(std::vector<double>&, double);
    void ForwardProcedure(const std::vector<int>&, std::vector<double>&);
    void BackwardProcedure(const std::vector<int>&, std::vector<double>&);
    void CalculateGamma(int, int);
    void CalculateXi(const std::vector<int>&, int);
    void UpdateTransition(const std::vector<std::vector<int>>&);
    void UpdateObservation(const std::vector<std::vector<int>>&);
    double CalculateLogProbability(const std::vector<double>&);
public:
    HMM(const Matrix&, const Matrix&, const std::vector<double>&);
    void ShowModelParameters(std::ostream&);
    double SequenceLogProbability(const std::vector<int>&);
    std::vector<int> Apply(const std::vector<int>&);
    void Train(const std::vector<std::vector<int>>&);
};

#endif // HMM_H_INCLUDED
