#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <set>

using namespace std;

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

void ReadFastaFile(ifstream& f, map<char, int>& epitope_amino, map<char, int>& protein_amino,
                   set<string>& proteins, set<string>& epitopes) {
    string s, ep, pr = "";
    if (f.is_open()) {
        while(!f.eof()) {
            getline(f, s);
            if(s[0] == '>') {
                if (!(pr.empty()) && pr.find(ep) != string::npos) {
                    proteins.insert(pr);
                    epitopes.insert(ep);
                    for (char c : pr)
                        ++protein_amino[c];
                    for (char c : ep)
                        ++epitope_amino[c];
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

int main()
{
    map<char, int> epitope_amino, protein_amino;
    set<string> proteins, epitopes;
    ifstream f("t_cell_epitopes.fasta");
    ofstream out("statistics.txt");
    ofstream prop("properties.txt");
    ReadFastaFile(f, epitope_amino, protein_amino, proteins, epitopes);

    out << "EPITOPE" << endl;
    for (auto p : epitope_amino)
        out << p.first << " : " << p.second << endl;
    out << endl << "PROTEIN" << endl;
    for (auto p : protein_amino)
        out << p.first << " : " << p.second << endl;
    out << proteins.size() << " " << epitopes.size() << endl;
}
