/* COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
This software is D-Wave confidential and proprietary information. */

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <set>
using namespace std;

#ifndef RANDOM_H
#define RANDOM_H

struct Rng {
    std::mt19937 engine;
    std::uniform_real_distribution<double> distribution;
    Rng() : distribution(0, 1) {
        engine.seed(rand());
    }
    double operator ()() {
        return distribution(engine);
    }
    void setSeed(unsigned int argSeed) {
	engine.seed(argSeed);
    }
};


vector<unsigned int> getRNGSeedsDevice(int num_samples, int num_attempts=10) {
  std::random_device rd;
  std::set<unsigned int> rngSeedSet;
  std::vector<unsigned int> vRngSeeds;
  vRngSeeds.reserve(num_samples);
  for(int sample = 0; sample < num_samples; sample++) {
	  while(1) {
		  unsigned int rand_num;
		  while(1) {
			  try {
				  rand_num = rd();
			  }
			  catch(...) {
				  if(num_attempts) {
					  num_attempts--;
					  continue;
				  }
				  else {
					  rand_num = rand();
					  break;
				  }
			  }
			  break;
		  }
		  auto result = rngSeedSet.insert(rand_num);
		  // If the seed has not been seen yet save it.
		  if((result.first != rngSeedSet.end()) && result.second) {
			  vRngSeeds.push_back(*result.first);
			  break;
		  }
	  }
	  //std::cout << "Seed for sample " << sample <<" " <<  vRngSeeds[sample] << std::endl;
  }
  assert(vRngSeeds.size() == num_samples);
  return vRngSeeds;
}

// A backup for random seed generator if random_device is not available at all.
vector<unsigned int> getRNGSeedsPseudo(int num_samples) {
  std::set<unsigned int> rngSeedSet;
  std::vector<unsigned int> vRngSeeds;
  vRngSeeds.reserve(num_samples);
  for(int sample = 0; sample < num_samples; sample++) {
	  while(1) {
		  auto result = rngSeedSet.insert(rand());
		  // If the seed has not been seen yet save it.
		  if((result.first != rngSeedSet.end()) && result.second) {
			  vRngSeeds.push_back(*result.first);
			  break;
		  }
	  }
	  //std::cout << "Seed for sample " << sample <<" " <<  vRngSeeds[sample] << std::endl;
  }
  assert(vRngSeeds.size() == num_samples);
  return vRngSeeds;
}

vector<unsigned int> getRNGSeeds(int num_samples) {
	vector<unsigned int> vSeeds;
	try {
	  vSeeds = getRNGSeedsDevice(num_samples);
	} catch (...) {
	  vSeeds = getRNGSeedsPseudo(num_samples);
	}
	return vSeeds;
}

vector<double> Softmax(const vector<double>& weights, double Beta, double min_weight) {
    vector<double> probs;
    double sum = 0;
    for(auto weight : weights) {
        double pr = std::exp((-weight + min_weight) * Beta);
        sum += pr;
        probs.push_back(pr);
    }
    for(auto& pr : probs) {
        pr /= sum;
    }
    return probs;
}

int StochasticSelection(const vector<double>& probs, Rng& rng) {
    const double point = rng();
    double cur_cutoff = 0;
    for(int i=0; i<probs.size()-1; ++i) {
        cur_cutoff += probs[i];
        if(point < cur_cutoff) return i;
    }
    int a = probs.size() - 1;
    return a;
}


#endif