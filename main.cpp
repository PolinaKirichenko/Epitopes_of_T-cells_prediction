#include <ctime>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "hmm.h"

using namespace std;

typedef std::vector<std::vector<double>> Matrix;

void int_randomize(vector<int>& v, int x) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = rand() % x;
}

//TODO: use methods from random module instead of rand()
void bad_randomize(std::vector<double>& v) {
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

template <typename T>
std::ostream& operator << (std::ostream& o, const std::vector<T>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        o << v[i]<< " ";
    o << std::endl;
    return o;
}

//TODO: regular expressions
string ReadEpitope(string x) {
    size_t i = x.find('_');
    x.erase(x.begin(), x.begin() + i + 1);
    size_t j = x.find('_');
    x.erase(x.begin() + j, x.end());
    return x;
}

vector<int> AminoToInt (const string& s) {
    map<char, int> a = { {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4},
                         {'G', 5}, {'H', 6}, {'I', 7}, {'K', 8}, {'L', 9},
                         {'M', 10}, {'N', 11}, {'P', 12}, {'Q', 13}, {'R', 14},
                         {'S', 15}, {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19},
                         {'X', 20}, {'J', 21} };
    vector<int> v(s.size());
    int i = 0;
    for (char c : s) {
        v[i] = a[c];
        ++i;
    }
    return v;
}

void ReadFastaFile(map<string, string>& epitope, vector<vector<int>>& data, ifstream& f) {
    string s, ep, protein = "";
    if (f.is_open()) {
        while(!f.eof()) {
            getline(f, s);
            if(s[0] == '>') {
                if (!(protein.empty()) && protein.find(ep) != string::npos) {
                    epitope[protein] = ep;
                    data.push_back(AminoToInt(protein.substr(max(0, int(protein.find(ep)) - 9), 27)));
                    protein = "";
                }
                ep = ReadEpitope(s);
            } else {
                protein += s;
            }
        }
        f.close();
    } else {
        cout << "NO FILE";
    }
}

int main()
{
    map<string, string> epitope;
    vector<vector<int>> data;
    ifstream f("t_cell_epitopes.fasta");
    ofstream out("output.txt");
    ReadFastaFile(epitope, data, f);

    int n = 3, m = 22;
    Matrix A (n, std::vector<double>(n));
    Matrix B (n, std::vector<double>(m));
    std::vector<double> p0(n);
    srand(time(NULL));
    bad_randomize(p0);
    bad_randomize(A);
    bad_randomize(B);

    std::vector<std::vector<int>> training_data(data);

    HMM a(A, B, p0);
    a.ShowModelParameters(out);
    out << "PROBABILITIES " << std::endl;
    for (size_t i = 0; i < training_data.size(); ++i)
        out << a.SequenceLogProbability(training_data[i]) << " ";
    out << std::endl;

    double old_logprob = -INFINITY, new_logprob = 0;
        for (size_t i = 0; i < training_data.size(); ++i)
            new_logprob += a.SequenceLogProbability(training_data[i]);

    int c = 0;
    double logeps = 0.3;
    out << "EDGE IS  " << logeps << std::endl;
    while (new_logprob - old_logprob > logeps) {
    //for (size_t j = 0; j < 1000; ++j) {
        ++c;
        std::cout << c << std::endl;
        a.Train(training_data);
        old_logprob = new_logprob;
        new_logprob = 0;
        for (size_t i = 0; i < training_data.size(); ++i)
            new_logprob += a.SequenceLogProbability(training_data[i]);
        out << c  << " PROBABILITY " << new_logprob - old_logprob << std::endl;
        for (size_t i = 0; i < training_data.size(); ++i)
            out << a.SequenceLogProbability(training_data[i]) << " ";
        out << std::endl << std::endl;
    }

    out << "THE NUMBER OF ITERATIONS " << c << std::endl << "NEW PARAMETERS" << std::endl;
    a.ShowModelParameters(out);
}
