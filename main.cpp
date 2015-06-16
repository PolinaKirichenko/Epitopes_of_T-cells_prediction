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

typedef vector<vector<double>> Matrix;

void int_randomize(vector<int>& v, int x) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = rand() % x;
}

//TODO: use methods from random module instead of rand()
void randomize(vector<double>& v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = rand() % 100 + 1;
        s += v[i];
    }
    for (size_t i = 0; i < v.size(); ++i)
        v[i]  = 1.0 * v[i] / s;
}

void randomize(Matrix& M) {
    for (size_t i = 0; i < M.size(); ++i)
        randomize(M[i]);
}

template <typename T>
ostream& operator << (ostream& o, const vector<T>& v) {
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

/*
Amino acid classification:
0 is hydrophobic
1 is hydrophilic
2 is + charged
3 is - charged
*/
vector<int> AminoProperties (const string& s) {
    map<char, int> a = { {'A', 0}, {'I', 0}, {'L', 0}, {'V', 0}, {'P', 0}, {'G', 0}, {'F', 0}, {'Y', 0}, {'W', 0},
                         {'S', 1}, {'T', 1}, {'C', 1}, {'M', 1}, {'N', 1}, {'Q', 1},
                         {'D', 2}, {'E', 2},
                         {'H', 3}, {'K', 3}, {'R', 3}, };
    vector<int> v(s.size());
    int i = 0;
    for (char c : s) {
        v[i] = a[c];
        ++i;
    }
    return v;
}

vector<int> AminoToInt (const string& s) {
    map<char, int> a = { {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3}, {'F', 4},
                         {'G', 5}, {'H', 6}, {'I', 7}, {'K', 8}, {'L', 9},
                         {'M', 10}, {'N', 11}, {'P', 12}, {'Q', 13}, {'R', 14},
                         {'S', 15}, {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19},
                         {'X', 9}, {'L', 9} };
    vector<int> v(s.size());
    int i = 0;
    for (char c : s) {
        v[i] = a[c];
        ++i;
    }
    return v;
}

void ReadFastaFile(ifstream& f, map<vector<int>, vector<int>>& prot_ep, vector<vector<int>>& epitopes, vector<vector<int>>& proteins,
                   map<vector<int>, vector<int>>& properties_prot_ep, vector<vector<int>>& properties_epitopes,
                   vector<vector<int>>& properties_proteins, int border) {
    string s, ep, pr = "", considerable_sub;
    if (f.is_open()) {
        while(!f.eof()) {
            getline(f, s);
            if(s[0] == '>') {
                if (!(pr.empty()) && pr.find(ep) != string::npos) {
                    considerable_sub = pr.substr(max(0, int(pr.find(ep)) - border), 2 * border + 9);
                    prot_ep[AminoToInt(pr)] = AminoToInt(considerable_sub);
                    properties_prot_ep[AminoProperties(pr)] = AminoProperties(considerable_sub);
                    proteins.push_back(AminoToInt(pr));
                    epitopes.push_back(AminoToInt(considerable_sub));
                    properties_proteins.push_back(AminoProperties(pr));
                    properties_epitopes.push_back(AminoProperties(considerable_sub));
                    pr = "";
                }
                ep = ReadEpitope(s);
            } else {
                pr += s;
            }
        }
        f.close();
    } else {
        cout << "NO FILE";
    }
}

void TrainModel(HMM& m, const vector<vector<int>>& training_data, ostream& out) {
    double old_logprob = -INFINITY, new_logprob = 0;
    for (size_t i = 0; i < training_data.size(); ++i)
            new_logprob += m.SequenceLogProbability(training_data[i]);

    int iteration = 0;
    double logeps = 0.3;
    while (new_logprob - old_logprob > logeps) {
    //for (size_t j = 0; j < 1000; ++j) {
        ++iteration;
        std::cout << iteration << std::endl;
        m.Train(training_data);
        old_logprob = new_logprob;
        new_logprob = 0;
        for (size_t i = 0; i < training_data.size(); ++i)
            new_logprob += m.SequenceLogProbability(training_data[i]);
    }
    out << "After " <<  iteration << " iterations of training reaching edge " << new_logprob - old_logprob << " probabilities are:" << endl;
    for (size_t i = 0; i < training_data.size(); ++i)
        out << m.SequenceLogProbability(training_data[i]) << " ";
    out << endl << endl;
}

// False epitope is the protein subsequence whose probability to be the epitope is bigger than one of the real epitope
void Test(HMM& m, const vector<vector<int>>& test, map<vector<int>, vector<int>>& real_ep, int border, ostream& out) {
    vector<int> sub;
    int mistakes = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        int false_ep = 0;
        for (size_t j = 0; j <= test[i].size() - (2 * border + 9); ++j) {
            sub = vector<int>(test[i].begin() + j, test[i].begin() + j + 2 * border + 9);
            if (m.SequenceLogProbability(sub) > m.SequenceLogProbability(real_ep[test[i]])) {
                ++false_ep;
                ++mistakes;
            }
        }
        out << false_ep << " false epitopes out of " << test[i].size() - (2 * border + 9) + 1 << " subsequences" << endl;
    }
    out << "Total amount of mistakes: " << mistakes << endl;
}

// Here we make sure that test and ptest relate to the same data
void CombineModelTest(HMM& m, HMM& pm, const vector<vector<int>>& test, const vector<vector<int>>& ptest,
                      map<vector<int>, vector<int>>& real_ep, map<vector<int>, vector<int>>& preal_ep, int border, ostream& out) {
    vector<int> sub, psub;
    int mistakes = 0;
    for (size_t i = 0; i < test.size(); ++i) {
        int false_ep = 0;
        for (size_t j = 0; j <= test[i].size() - (2 * border + 9); ++j) {
            sub = vector<int>(test[i].begin() + j, test[i].begin() + j + 2 * border + 9);
            psub = vector<int>(ptest[i].begin() + j, ptest[i].begin() + j + 2 * border + 9);
            if (m.SequenceLogProbability(sub) > m.SequenceLogProbability(real_ep[test[i]]) &&
                pm.SequenceLogProbability(psub) > pm.SequenceLogProbability(preal_ep[ptest[i]])) {
                ++false_ep;
                ++mistakes;
            }
        }
        out << false_ep << " false epitopes out of " << test[i].size() - (2 * border + 9) + 1 << " subsequences" << endl;
    }
    out << "Total amount of mistakes: " << mistakes << endl;
}

int main()
{
    map<vector<int>, vector<int>> prot_ep, properties_prot_ep;
    vector<vector<int>> epitopes, proteins, properties_epitopes, properties_proteins;
    ifstream f("t_cell_epitopes.fasta");
    ofstream out("output9.txt");
    int border = 9; // we consider not only the epitope sequence itself but border amino acids before and after the epitope
    ReadFastaFile(f, prot_ep, epitopes, proteins, properties_prot_ep, properties_epitopes, properties_proteins, border);

    vector<vector<int>> training_data(epitopes.begin(), epitopes.begin() + 1100);
    vector<vector<int>> test(proteins.begin() + 1100, proteins.end());
    vector<vector<int>> ptraining_data(properties_epitopes.begin(), properties_epitopes.begin() + 1100);
    vector<vector<int>> ptest(properties_proteins.begin() + 1100, properties_proteins.end());

// Sequence training
    int n = 7, m = 20;
    Matrix transition (n, vector<double>(n));
    Matrix observation (n, vector<double>(m));
    vector<double> initial_distribution(n);
    srand(time(NULL));
    randomize(initial_distribution);
    randomize(transition);
    randomize(observation);
    HMM model(transition, observation, initial_distribution);
    out << "SEQUENCE TRAINING" << endl;
    TrainModel(model, training_data, out);

//Amino acids' properties training
    int pn = 2, pm = 4;
    Matrix ptransition (pn, vector<double>(pn));
    Matrix pobservation (pn, vector<double>(pm));
    vector<double> pinitial_distribution(pn);
    srand(time(NULL));
    randomize(pinitial_distribution);
    randomize(ptransition);
    randomize(pobservation);
    HMM prop_model(ptransition, pobservation, pinitial_distribution);
    out << "AMINO PROPERTIES TRAINING" << endl;
    TrainModel(prop_model, ptraining_data, out);

/*    out << "SEQUENCE CROSS VALIDATION" << endl;
    Test(model, test, prot_ep, border, out);
    out << endl;
    out << "PROPERTIES CROSS VALIDATION" << endl;
    Test(prop_model, ptest, properties_prot_ep, border, out);
    out << endl;*/
    out << "COMBINE MODELS CROSS VALIDATION" << endl;
    CombineModelTest(model, prop_model, test, ptest, prot_ep, properties_prot_ep, border, out);
    out << endl;

    out << "NEW PARAMETERS SEQUENCE MODEL" << std::endl;
    model.ShowModelParameters(out);
    out << endl << "NEW PARAMETERS PROPERTIES MODEL" << std::endl;
    prop_model.ShowModelParameters(out);
}
