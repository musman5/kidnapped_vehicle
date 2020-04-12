/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
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

#define SMALLEST_VALUE 0.0001

using std::string;
using std::vector;
using std::normal_distribution;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 125;  // TODO: Set the number of particles
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Initialize
  for (int i=0; i < num_particles; i++) {
    Particle init_particle;
    init_particle.x = dist_x(gen);
    init_particle.y = dist_y(gen);
    init_particle.theta = dist_theta(gen);
    init_particle.weight =  1.0;
    init_particle.id = i;
    
    // Save particle information to set of particles
    particles.push_back(init_particle);

  }
  
  is_initialized= true; 
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
  
  for (int i = 0 ; i < num_particles; i++) {
       
    // To avoid divide by zero, check yaw for smallest value
    if (fabs(yaw_rate) > SMALLEST_VALUE) {
      particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) -  cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
    } else {
      particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }

    // Add noise
    particles[i].x +=  dist_x(gen);
    particles[i].y +=  dist_y(gen);
    particles[i].theta +=  dist_theta(gen);
    
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

  // Using nearest neighbor algorithm associate observation with map co-ordinates.
  
  for (unsigned int i=0; i < observations.size(); i++) {
    // Initialize minimum distance with maximum value
    double minimum_distance = std::numeric_limits<double>::max();
    int saved_perdiction_id = -1;
    
    double curr_obs_x = observations[i].x;
    double curr_obs_y = observations[i].y;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      double curr_pred_x = predicted[j].x;
      double curr_pred_y = predicted[j].y;
      int curr_pred_id = predicted[j].id;
      double current_distance = dist(curr_obs_x, curr_obs_y, curr_pred_x, curr_pred_y);
      
      if (current_distance < minimum_distance) {
        minimum_distance = current_distance;
        saved_perdiction_id = curr_pred_id;
      }
    }
    observations[i].id = saved_perdiction_id;
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

  
  for (unsigned int i=0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    
    // Loop through landmark list
    vector<LandmarkObs> predicted_landmarks;
    for (unsigned int j =0; j < map_landmarks.landmark_list.size(); j++) {
      // Save information of each landmark into local variables
      float current_landmark_x = map_landmarks.landmark_list[j].x_f;
      float current_landmark_y = map_landmarks.landmark_list[j].y_f;
      int current_landmark_id = map_landmarks.landmark_list[j].id_i;

      // Save only those landmarks which are in sensor range
      if ((fabs(current_landmark_x - particle_x) <= sensor_range) && (fabs(current_landmark_y - particle_y) <= sensor_range)){
      
        // Add prediction to landmark vector
        predicted_landmarks.push_back(LandmarkObs{ current_landmark_id, current_landmark_x, current_landmark_y });
        
      }
    }
    
    // Transform vehicle co-ordinates to map co-ordinates
    vector<LandmarkObs> transformed_observations;
    
    for (unsigned int k=0; k < observations.size(); k++) {

      float transformed_x = particle_x + cos(particle_theta) * observations[k].x - sin(particle_theta) * observations[k].y;
      float transformed_y = particle_y + sin(particle_theta) * observations[k].x + cos(particle_theta) * observations[k].y;
      
      transformed_observations.push_back(LandmarkObs{observations[k].id, transformed_x, transformed_y});

    }
    // Associate observations with predicted landmarks
    dataAssociation(predicted_landmarks, transformed_observations);
    
    // Use Multiverse Gaussian Distribution to calculate the weight
    particles[i].weight = 1.0;
    
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];

    double gauss_norm = 1/(2 * M_PI * sigma_x * sigma_y);
    
    // Use multiverse Gaussian Probablity funtion to calculate wieght of particle
    for (unsigned int m = 0; m < transformed_observations.size(); m++) {
      double trans_obs_x = transformed_observations[m].x;
      double trans_obs_y = transformed_observations[m].y;
      double trans_obs_id = transformed_observations[m].id;
      double weight = 1.0;
      
      for (unsigned int n = 0; n < predicted_landmarks.size(); n++) {
        double pred_landmark_x = predicted_landmarks[n].x;
        double pred_landmark_y = predicted_landmarks[n].y;
        double pred_landmark_id = predicted_landmarks[n].id;
        
        if (trans_obs_id == pred_landmark_id) {
          double exponent = (pow((trans_obs_x - pred_landmark_x),2) / (2.0 * pow(sigma_x,2))) + (pow((trans_obs_y - pred_landmark_y), 2) / (2.0 * pow(sigma_y,2)));
          // Calculate weight using normalization terms and exponent as described in class
          weight = gauss_norm * exp(-exponent);
          if(weight !=0) particles[i].weight *= weight;
        }
      } // end loop n
    } // end loop m

  }  // end loop i

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled_particles;
  // set container size to number of particles
  resampled_particles.resize(num_particles);
  
  vector<double> particle_weights;
  // Get particle weights
  for(int i=0; i < num_particles; i++) {
    particle_weights.push_back(particles[i].weight);
  }
  
  // From discrete distribution reference calculate distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(particle_weights.begin(), particle_weights.end());
  
  // Resample based on discrete distribution
  for (int i=0; i < num_particles; i++) {
    resampled_particles[i] = particles[dist(gen)];
  }
  particles.clear();
  // Save new set of particles
  particles = resampled_particles;
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
  
  // Clear previous associations
  // particle.associations.clear();
  // particle.sense_x.clear();
  // particle.sense_y.clear();
  
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