#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<double>> Matrix;

template <typename T>
ostream& operator << (ostream& o, const vector<T>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        o << v[i]<< " ";
    o << endl;
    return o;
}

ostream& operator << (ostream& o, const Matrix& M) {
    for (size_t i = 0; i < M.size(); ++i)
        o << M[i];
    return o;
}

void int_randomize(vector<int>& v, int x) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = rand() % x;
}

//TODO: use methods from random module instead of rand()
void bad_randomize(vector<double>& v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = rand() % 100 + 1;
        s += v[i];
    }
    for (size_t i = 0; i < v.size(); ++i)
        v[i]  = 1.0 * v[i] / s;
}

void bad_randomize(Matrix& M) {
    for (size_t i = 0; i < M.size(); ++i)
        bad_randomize(M[i]);
}

class HMM {
private:
    vector<double> initial_distribution; //initial probability distribution
    Matrix transition; // transition[i][j] is probability of going into state j after state i
    Matrix observation; // observation[i][j] is probability of symbol i in state j
    Matrix alpha; //alpha[t][j] is probability to observe Y1,...Yt and state Xt = j
    Matrix beta; //beta [t][j] is probability to observe Y(t+1),...Yk if Xt = j
    vector<int> Y; // the given sequence
    int n, m, T; // numbers of state, observation symbols and observed symbols respectively
    Matrix gamma;
    vector<Matrix> xi;

    void scale(vector<double>& v, double c) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] /= c;
    }

    void ForwardProcedure(vector<double>& scaling_numbers) {
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
        //cout << "ALPHA" << endl << alpha << endl;
    }

    void BackwardProcedure(vector<double>& scaling_numbers) {
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
        //cout << "BETA" << endl << beta << endl;
    }

    void CalculateGamma() {
        double denom;
        for (size_t t = 0; t < T; ++t)
            for (size_t i = 0; i < n; ++i) {
                denom = 0;
                for (size_t j = 0; j < n; ++j)
                    denom += alpha[t][j] * beta[t][j];
                gamma[t][i] = alpha[t][i] * beta[t][i] / denom;
            }
        //cout << "GAMMA" << endl << gamma << endl;
    }

    void CalculateXi() {
        double denom;
        for (size_t t = 0; t < T - 1; ++t) {
            denom = 0;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    denom += alpha[t][i] * transition[i][j] * beta[t + 1][j] * observation[Y[t + 1]][j];
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    xi[t][i][j] = alpha[t][i] * transition[i][j] * beta[t + 1][j] * observation[Y[t + 1]][j] / denom;
        }
        /*cout << "XI" << endl;
        for (size_t t = 0; t < T - 1; ++t)
            cout << xi[t] << endl;*/
    }

    void UpdateTransition() {
        double num, denom;
        for (size_t i = 0; i < n; ++i) {
            denom = 0;
            for (size_t t = 0; t < T - 1; ++t)
                denom += gamma[t][i];
            for (size_t j = 0; j < n; ++j) {
                num = 0;
                for (size_t t = 0; t < T - 1; ++t)
                    num += xi[t][i][j];
                transition[i][j] = 0.1 * transition[i][j] + 0.9 * num / denom;
            }
        }
    }

    void UpdateObservation() {
        double num, denom;
        for (size_t i = 0; i < n; ++i) {
            denom = 0;
            for (size_t t = 0; t < T; ++t)
                denom += gamma[t][i];
            for (size_t j = 0; j < m; ++j) {
                num = 0;
                for (size_t t = 0; t < T; ++t)
                    if (Y[t] == j)
                        num += gamma[t][i];
                observation[j][i] = 0.1 * observation[j][i] + 0.9 * num / denom;
            }
        }
    }
public:
    HMM(const Matrix& A, const Matrix& B, const vector<double>& p0) : transition(A), observation(B), initial_distribution(p0) {
        n = initial_distribution.size();
        m = observation.size();
    }

    void ShowModelParameters(ostream& out) {
        out << "INITIAL DISTRIBUTION" << endl << initial_distribution << endl;
        out << "TRANSITION" << endl << transition << endl;
        out << "OBSERVATION PROBABILITY" << endl << observation << endl;
    }

    double SequenceProbability(const vector<int>& sequence) {
        Y = sequence;
        T = Y.size();
        alpha = Matrix(T, vector<double>(n));
        vector<double> scaling_numbers(T);
        ForwardProcedure(scaling_numbers);
        double probability = 1;
        for (size_t i = 0; i < T; ++i)
            probability *= scaling_numbers[i];
        return probability;
    }

    vector<int> Apply(const vector<int>& sequence) {
        Y = sequence;
        T = Y.size();
        alpha = Matrix(T, vector<double>(n));
        beta = Matrix(T, vector<double>(n));
        gamma = Matrix(T, vector<double>(n));

        vector<double> scaling_numbers(T);
        ForwardProcedure(scaling_numbers);
        BackwardProcedure(scaling_numbers);
        CalculateGamma();

        vector<int> most_likely_states(T);
        for (size_t t = 0; t < T; ++t)
            most_likely_states[t] = max_element(gamma[t].begin(), gamma[t].end()) - gamma[t].begin();
        return most_likely_states;
    }

    void Train(const vector<int>& sequence) {
        Y = sequence;
        T = Y.size();
        alpha = Matrix(T, vector<double>(n));
        beta = Matrix(T, vector<double>(n));
        gamma = Matrix(T, vector<double>(n));
        xi = vector<Matrix>(T - 1, Matrix(n, vector<double>(n)));

        vector<double> scaling_numbers(T);
        ForwardProcedure(scaling_numbers);
        BackwardProcedure(scaling_numbers);
        CalculateGamma();
        CalculateXi();
        initial_distribution = gamma[0];
        UpdateTransition();
        UpdateObservation();
    }
};
