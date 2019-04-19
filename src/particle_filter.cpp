/**
 * particle_filter.cpp
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#define EPS 0.00001

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;
using std::discrete_distribution;
using std::numeric_limits;
using std::uniform_real_distribution;
using std::uniform_int_distribution;
using std::mersenne_twister_engine;

// declare a random engine to be used across multiple and various method calls
 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized) {
	return;
  }
  num_particles = 100;  // TODO: Set the number of particles
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(y, std[2]);
  gen.seed(5);
  for (int i = 0; i < num_particles; i++) {
	  Particle p = Particle();
	  
	  p.x = dist_x(gen);
	  p.y = dist_y(gen);
	  p.theta = dist_theta(gen);
	  p.id = 1;
	  p.weight = 1.0;
	  particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	for (int i = 0; i < num_particles; i++) {
		
		if (fabs(yaw_rate) >EPS){
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
	int num_obs = observations.size();
	int num_landmarks = predicted.size();

	for (int i = 0; i < num_obs; ++i) {
		int closest_landmark = 0;
		int min_dist = 999999;
		int curr_dist;
		// Iterate through all landmarks to check which is closest
		for (int j = 0; j < num_landmarks; ++j) {
			// Calculate Euclidean distance
			//curr_dist = sqrt(pow(trans_obs[i].x - landmarks[j].x, 2)
			//	+ pow(trans_obs[i].y - landmarks[j].y, 2));
			curr_dist = dist(observations[i].x, predicted[j].x, observations[i].y, predicted[j].y);
			// Compare to min_dist and update if closest
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				closest_landmark = j;
			}
		}
		// Output the related association information
		//std::cout << "OBS" << observations[i].id << " associated to L"
		//	<< landmarks[closest_landmark].id << std::endl;
		observations[i].id = closest_landmark;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	for (int i = 0; i < num_particles; i++) {
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		//Create a vector to hold the map landmark locations predicted to be within sensor range of the particle
		vector<LandmarkObs> lmark_predictions;


		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			//Get id and x,y coordinates
			int lmark_id = map_landmarks.landmark_list[j].id_i;
			float lmark_x = map_landmarks.landmark_list[j].x_f;
			float lmark_y = map_landmarks.landmark_list[j].y_f;


			//Only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular region around the particle, this considers a rectangular region but is computationally faster)
			if (fabs(lmark_x - particle_x) <= sensor_range && fabs(lmark_y - particle_y) <= sensor_range) {
				lmark_predictions.push_back(LandmarkObs{ lmark_id, lmark_x, lmark_y });
			}
		}

		//Create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
		vector<LandmarkObs> transfrmd_obs;
		for (int j = 0; j < observations.size(); j++) {
			double t_x = cos(particle_theta)*observations[j].x - sin(particle_theta)*observations[j].y + particle_x;
			double t_y = sin(particle_theta)*observations[j].x + cos(particle_theta)*observations[j].y + particle_y;
			transfrmd_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		//Data association for the predictions and transformed observations on current particle
		dataAssociation(lmark_predictions, transfrmd_obs);
		particles[i].weight = 1.0;
		for (int j = 0; j < transfrmd_obs.size(); j++) {
			double o_x, o_y, pr_x, pr_y;
			o_x = transfrmd_obs[j].x;
			o_y = transfrmd_obs[j].y;
			int asso_prediction = transfrmd_obs[j].id;

			//x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < lmark_predictions.size(); k++) {
				if (lmark_predictions[k].id == asso_prediction) {
					pr_x = lmark_predictions[k].x;
					pr_y = lmark_predictions[k].y;
				}
			}

			//Weight for this observation with multivariate Gaussian
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			//double obs_w = (1 / (2 * M_PI*s_x*s_y)) * exp(-(pow(pr_x - o_x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - o_y, 2) / (2 * pow(s_y, 2)))));
			double obs_w = ParticleFilter::multiv_prob(sig_x, sig_y, pr_x, pr_y, o_x, o_y);
			//Product of this obersvation weight with total observations weight
			particles[i].weight *= obs_w;
		}
	}

}
double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y)
{
	// calculate normalization term
	double gauss_norm;
	gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

	// calculate exponent
	double exponent;
	exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
		+ (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
	// calculate weight using normalization terms and exponent
	double weight;
	weight = gauss_norm * exp(-exponent);
	return weight;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */


	// Find max weight mw= max(w)
	vector<double> weights;
	double maxWeight = std::numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > maxWeight) {
			maxWeight = particles[i].weight;
		}
	}

	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);
	int index = distInt(gen);
	double beta = 0.0;
	vector<Particle> resampledParticles;
	for (int i = 0; i < num_particles; i++) {
		beta += distDouble(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampledParticles.push_back(particles[index]);
	}

	particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}