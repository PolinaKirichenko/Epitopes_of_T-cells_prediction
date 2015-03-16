#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

typedef vector<vector<double>> matrix;

template <typename T>
void print(vector<T>& v) {
    for(size_t i = 0; i < v.size(); ++i)
        cout << v[i]<< " ";
    cout << endl;
}

void print(matrix& M) {
    for(size_t i = 0; i < M.size(); ++i)
        print(M[i]);
}

void int_randomize(vector<int>& v, int x) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = rand() % x;
}

void bad_randomize(vector<double>& v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = rand() % 100;
        s += v[i];
    }
    for (size_t i = 0; i < v.size(); ++i)
        v[i]  = 1.0 * v[i] / s;
}

void bad_randomize(matrix& M) {
    for (size_t i = 0; i < M.size(); ++i)
        bad_randomize(M[i]);
}

class HMM {
private:
    vector<double> initial_distribtion; //initial probability distribution
    matrix transition; // transition[i][j] is probability of going into state j after state i
    matrix observation; // observation[i][j] is probability of symbol i in state j
    matrix alpha; //alpha[t][j] is probability to see Y1,...Yt and state Xt = j
    matrix beta; //beta [t][j] is probability to see Y(t+1),...Yk if Xt = j
    matrix gamma;
    vector<matrix> xi;
    vector<int> Y; // the given sequence
    int n, m, T;
public:
    HMM(matrix& A, matrix& B, vector<double>& p0) : transition(A), observation(B), initial_distribtion(p0) {
        n = initial_distribtion.size();
        m = observation.size();
    }

    void forward_procedure() {
        for (size_t j = 0; j < n; ++j)
            alpha[0][j] = initial_distribtion[j] * observation[Y[0]][j];
        for (size_t t = 1; t < T; ++t)
            for (size_t j = 0; j < n; ++j) {
                alpha[t][j] = 0;
                for (size_t i = 0; i < n; ++i)
                    alpha[t][j] += alpha[t - 1][i] * transition[i][j];
                alpha[t][j] *= observation[Y[t]][j];
            }
    }

    void backward_procedure() {
        for (size_t j = 0; j < n; ++j)
            beta[T - 1][j] = 1;
        for (int t = T - 2; t >= 0; --t)
            for (size_t i = 0; i < n; ++i) {
                beta[t][i] = 0;
                for (size_t j = 0; j < n; ++j)
                    beta[t][i] += beta[t + 1][j] * transition[i][j] * observation[Y[t + 1]][j];
            }
    }

    void calculate_gamma() {
        double sum = 0;
        for (size_t t = 0; t < T; ++t) {
            sum = 0;
            for (size_t j = 0; j < n; ++j)
                sum += alpha[t][j] * beta[t][j];
            for (size_t i = 0; i < n; ++i)
                gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
            }
        /*cout << "GAMMA" << endl;
        print(gamma);
        cout << endl;*/
    }

    void calculate_xi() {
        double sum = 0;
        for (size_t t = 0; t < T - 1; ++t) {
            sum = 0;
            for (size_t j = 0; j < n; ++j)
                sum += alpha[t][j] * beta[t][j];
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    xi[t][i][j] = alpha[t][i] * transition[i][j] * beta[t + 1][j] * observation[Y[t + 1]][j] / sum;
        }
        /*cout << "XI" << endl;
        for (size_t t = 0; t < T - 1; ++t) {
            print(xi[t]);
            cout << endl;
        }*/
    }

    void update_transition() {
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum = 0;
            for (size_t t = 0; t < T - 1; ++t)
                sum += gamma[t][i];
            for (size_t j = 0; j < n; ++j) {
                transition[i][j] = 0;
                for (size_t t = 0; t < T - 1; ++t)
                    transition[i][j] += xi[t][i][j];
                transition[i][j] /= sum;
            }
        }
    }

    void update_observation() {
        double sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum = 0;
            for (size_t t = 0; t < T; ++t)
                sum += gamma[t][i];
            for (size_t j = 0; j < m; ++j) {
                observation[j][i] = 0;
                for (size_t t = 0; t < T; ++t)
                    if (Y[t] == j) {
                        observation[j][i] += gamma[t][i];
                    }
                observation[j][i] /= sum;
            }
        }
    }

    void learn(const vector<int>& sequence) {
        T = sequence.size();
        Y = sequence;
        alpha = matrix(T, vector<double>(n));
        beta = matrix(T, vector<double>(n));
        gamma = matrix(T, vector<double>(n));
        xi = vector<matrix>(T - 1, matrix(n, vector<double>(n)));
        forward_procedure();
        backward_procedure();
        calculate_gamma();
        calculate_xi();
        initial_distribtion = gamma[0];
        update_transition();
        update_observation();

        cout << "NEW PI" << endl;
        print(initial_distribtion);
        cout << endl;
        cout << "NEW TRANSITION" << endl;
        print(transition);
        cout << endl;
        cout << "NEW OBSERVATION" << endl;
        print(observation);
        cout << endl;
    }
};

int main()
{
    int n = 3, m = 5, T = 6;
    matrix A (n, vector<double>(n));
    matrix B (m, vector<double>(n));
    vector<int> Y(T);
    vector<double> p0(n);

    srand(time(NULL));
    bad_randomize(p0);
    bad_randomize(A);
    bad_randomize(B);
    int_randomize(Y, m);
    cout << "INITIAL PROBABILITY" << endl;
    print(p0);
    cout << endl;
    cout << "TRANSITION" << endl;
    print(A);
    cout << endl;
    cout << "OBSERVATION PROBABILITY" << endl;
    print(B);
    cout << endl;
    cout << "SEQUENCE" << endl;
    print(Y);
    cout << endl;

    HMM a(A, B, p0);
    a.learn(Y);
}
