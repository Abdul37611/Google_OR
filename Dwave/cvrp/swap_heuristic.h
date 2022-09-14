/* COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
This software is D-Wave confidential and proprietary information. */

#include <unordered_map>
#include <vector>
#include <iostream>
#include <math.h>
#include "random.h"
#include <cstdint>


using namespace std;


struct Node {
    int prev, data, next;
};


class Permutation {
    private:
        int num_samples;
        int num_variables;
        vector<vector<Node>> state;
        vector<vector<double>> distance;
        vector<int> num_connected;

    public:
        Permutation(vector<vector<double>>, int);
        vector<pair<vector<int>, double>> GetState();
        void Swap(int a, int b, int sample);
        void Insert(int a, int b, int sample);
        double InsertDelta(int a, int b, int sample);
        void Remove(int a, int sample);
        double RemoveDelta(int a, int sample);
        double Delta(int a, int b, int sample);
        void SetState(vector<vector<int>>);
        vector<double> Cost();
        double Cost(int);
        void SA(double beta, double max_beta, double scale);
        void print_state(int sample);

        bool IsConnected(int a, int sample) {
            return state[sample][a].next != -1;
        };
        int NumConnected(int sample) {
            return num_connected[sample];
        };
        vector<vector<int>> GetStateExplicit();
};


void Permutation::SetState(vector<vector<int>> input) {
    num_samples = input.size();
    for (int sample = 0; sample < num_samples; sample++) {
        int n = input[sample].size();
        state[sample].resize(num_variables);
        num_connected[sample] = 0;
        for (int i = 0; i < n; i++) {
            int data = input[sample][i];
            int prev, next;
            if (i == 0) {
                prev = input[sample][n - 1];
            } else {
                prev = input[sample][(i - 1)];
            }
            next = input[sample][(i + 1) % n];
            state[sample][data].prev = prev;
            state[sample][data].data = data;
            state[sample][data].next = next;
            num_connected[sample] += 1;
        }
    }
}


vector<pair<vector<int>, double>> Permutation::GetState() {
    vector<pair<vector<int>, double>> solution;
    solution.resize(num_samples);
    for (int sample = 0; sample < num_samples; sample++){
        solution[sample].first.resize(state[sample].size());
        auto a = state[sample][0];
        for (int i = 0; i < state[sample].size(); i ++) {
            solution[sample].first[i] = a.data;
            a = state[sample][a.next];
        }
        solution[sample].second = this->Cost(sample);
    }
    return solution;
}


vector<vector<int>> Permutation::GetStateExplicit() {
    vector<vector<int>> solution;
    solution.resize(num_samples);
    for (int sample = 0; sample < num_samples; sample++){
        solution[sample].resize(state[sample].size());
        for (int i = 0; i < state[sample].size(); i ++) {
            auto a = state[sample][i];
            solution[sample][i] = a.next;
        }
    }
    return solution;
}


Permutation::Permutation(vector<vector<double>> distances, int _num_samples) {
    num_samples = _num_samples;
    num_variables = distances.size();
    num_connected.resize(num_samples);
    int n = distances.size();
    this->state.resize(num_samples);
    for (int sample = 0; sample < num_samples; sample++) {
        this->state[sample].resize(n);
        for (int i = 0; i < n; i++) {
            int a;
            if (i == 0) {
                a = n - 1;
            } else {
                a = (i - 1);
            }
            int b = (i + 1) % n;
            this->state[sample][i].prev = -1;
            this->state[sample][i].data = i;
            this->state[sample][i].next = -1;
        }
        num_connected[sample] = 0;
    }
    distance = distances;
}


void Permutation::Swap(int a, int b, int sample) {
    if (state[sample].size() <= 2) {
        return;
    }
    if (state[sample][a].next == -1) {
        return;
    }
    if (state[sample][b].next == -1) {
        return;
    }
    int prev_a = state[sample][a].prev;
    int next_a = state[sample][a].next;
    int prev_b = state[sample][b].prev;
    int next_b = state[sample][b].next;

    if (prev_a == b) {
        // prev_b -> b -> a -> next_a
        state[sample][a].prev = prev_b;
        state[sample][a].next = b;
        state[sample][next_a].prev = b;
        if (state[sample][next_a].next == b) {
            state[sample][next_a].next = a;
        }
        if (state[sample][prev_b].prev == a) {
            state[sample][prev_b].prev = b;
        }
        state[sample][prev_b].next = a;
        state[sample][b].prev = a;
        state[sample][b].next = next_a;
    } else if (prev_b == a) {
        // prev_a -> a -> b -> next_b
        this->Swap(b, a, sample);
    } else {
        // prev_a -> a -> next_a -> ... -> prev_b -> b -> next_b
        // prev_a -> b -> next_a -> ... -> prev_b -> a -> next_b
        state[sample][prev_a].next = b;
        state[sample][prev_b].next = a;

        state[sample][a].next = next_b;
        state[sample][a].prev = prev_b;
        state[sample][b].next = next_a;
        state[sample][b].prev = prev_a;

        state[sample][next_a].prev = b;
        state[sample][next_b].prev = a;
    }
}

void Permutation::Insert(int a, int b, int sample) {
    if (state[sample][b].next != -1) {
        return;
    }
    if (state[sample][a].next == -1) {
        if (num_connected[sample] == 0){
            state[sample][a].next = b;
            state[sample][a].prev = b;
            state[sample][b].next = a;
            state[sample][b].prev = a;
            num_connected[sample] += 2;
            return;
        }
        return;
    }
    int next_a = state[sample][a].next;

    state[sample][a].next = b;
    state[sample][b].prev = a;
    state[sample][b].next = next_a;
    state[sample][next_a].prev = b;
    num_connected[sample] += 1;
}

double Permutation::InsertDelta(int a, int b, int sample) {
    if (state[sample][b].next != -1) {
        return 0;
    }
    if (state[sample][a].next == -1) {
        if (num_connected[sample] == 0) {
            return distance[a][b] + distance[b][a];
        }
        return 0;
    }
    int next_a = state[sample][a].next;

    return distance[a][b] + distance[b][next_a] - distance[a][next_a];
}

void Permutation::Remove(int a, int sample) {
    if (state[sample][a].next == -1) {
        return;
    }
    int next_a = state[sample][a].next;
    int prev_a = state[sample][a].prev;

    state[sample][a].next = -1;
    state[sample][a].prev = -1;

    state[sample][prev_a].next = next_a;
    state[sample][next_a].prev = prev_a;

    if (prev_a == next_a){
        state[sample][next_a].next = -1;
        state[sample][next_a].prev = -1;
        num_connected[sample] -= 2;
        return;
    }
    num_connected[sample] -= 1;
}

double Permutation::RemoveDelta(int a, int sample) {
    if (state[sample][a].next == -1) {
        return 0;
    }
    int next_a = state[sample][a].next;
    int prev_a = state[sample][a].prev;

    if (num_connected[sample] == 2){
        return - distance[prev_a][a] - distance[a][next_a];
    }
    return - distance[prev_a][a] - distance[a][next_a] +
    distance[prev_a][next_a];
}

double Permutation::Delta(int a, int b, int sample) {
    if (state[sample].size() <= 2) {
        return 0;
    }
    if (state[sample][a].next == -1) {
        return 0;
    }
    if (state[sample][b].next == -1){
        return 0;
    }
    double delta = 0;
    int prev_a = state[sample][a].prev;
    int next_a = state[sample][a].next;
    int prev_b = state[sample][b].prev;
    int next_b = state[sample][b].next;

    if (prev_a == b) {
        // prev_b -> b -> a -> next_a
        delta += - distance[b][a] + distance[a][b];
        delta += - distance[prev_b][b] + distance[prev_b][a];
        delta += - distance[a][next_a] + distance[b][next_a];
    } else if (prev_b == a) {
        // prev_a -> a -> b -> next_b
        return this->Delta(b, a, sample);
    } else {
        // prev_a -> a -> next_a -> ... -> prev_b -> b -> next_b
        // prev_a -> b -> next_a -> ... -> prev_b -> a -> next_b
        delta += - distance[prev_a][a] + distance[prev_a][b];
        delta += - distance[a][next_a] + distance[b][next_a];
        delta += - distance[prev_b][b] + distance[prev_b][a];
        delta += - distance[b][next_b] + distance[a][next_b];
    }
    return delta;
}


vector<double> Permutation::Cost() {
    vector<double> costs;
    costs.resize(num_samples);

    #pragma omp parallel for
    for (int sample = 0; sample < num_samples; sample++){
        costs[sample] = this->Cost(sample);
    }
    return costs;
}


double Permutation::Cost(int sample) {
    double cost = 0;
    for (auto a : state[sample]) {
        cost += distance[a.data][a.next];
    }
    return cost;
}


void Permutation::SA(double beta, double max_beta, double scale){
    #pragma omp parallel for
    for (int sample = 0; sample < num_samples; sample++){
        Rng rng;
        int n = state[sample].size();
        double sample_beta = beta;
        while (sample_beta < max_beta) {
            for (int a=0; a < n; a++) {
                for (int b=0; b < n; b++) {
                    double delta = this->Delta(a, b, sample);
                    double prob = 1.0;
                    double rand = rng();
                    if (delta >= 0) {
                        prob = exp(- sample_beta * delta);
                    }
                    if (prob > rand) {
                        this->Swap(a, b, sample);
                    }
                }
            }
            sample_beta *= scale;
        }
    }
}


void Permutation::print_state(int sample) {
    for (auto a : state[sample]) {
        if (IsConnected(a.data, sample)){
            cout << a.data << " " << a.next << "   |";
        }
    }
    cout << endl;
}


vector<pair<vector<int>, double>> solve(
        vector<vector<double>> distance, vector<vector<int>> states,
        double beta, double max_beta, double scale, int n = 12) {
    Permutation permutation(distance, n);
    if (states.size() == n) {
        permutation.SetState(states);
    }
    permutation.SA(beta, max_beta, scale);
    return permutation.GetState();
}
