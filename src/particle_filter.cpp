/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	//Set the number of particles.
	num_particles = 32;

	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// Creates a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Add random Gaussian noise to each particle.
	default_random_engine gen;

	//Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
		// cout << "p: id: " << p.id << "\tx:" << p.x << "\ty:" << p.y << "\tweight:" << p.weight << endl;
	}

	is_initialized = true;
	// cout << "init done" << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// cout << "prediction" << endl;
	//Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create random gen engine
	default_random_engine gen;

	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];
	// cout << "std_x: " << std_x << "\tstd_y:" << std_y << "\tstd_theta:" << std_theta << endl;

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p = particles.at(i);

		double x, y, theta;
		if (fabs(yaw_rate) > 0.000001)
		{
			x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			theta = p.theta + yaw_rate * delta_t;
		}
		else
		{
			cout << "going straight." << endl;
			x = p.x + velocity * delta_t * cos(p.theta);
			y = p.y + velocity * delta_t * sin(p.theta);
		}

		// Creates a normal (Gaussian) distribution for x, y, theta
		normal_distribution<double> dist_x(x, std_x);
		normal_distribution<double> dist_y(y, std_y);
		normal_distribution<double> dist_theta(theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		particles.at(i) = p;
		// cout << "prediction / particle x:" << p.x << "\ty:" << p.y << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// cout << "dataAssociation" << endl;

	for (int i = 0; i < observations.size(); i++)
	{
		LandmarkObs observation = observations.at(i);
		// cout << "pre-observation: " << observation.id << "\tx:" << observation.x << "\ty:" << observation.y << endl;

		int predicted_id = observation.id;
		double min_distance = DBL_MAX;
		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted.at(j);
			double distance = sqrt(pow(pred.x - observation.x, 2) + pow(pred.y - observation.y, 2));
			if (distance < min_distance)
			{
				min_distance = distance;
				predicted_id = pred.id;
			}
		}
		observation.id = predicted_id;
		// cout << "post-observation id:" << observation.id << "\tx:" << observation.x << "\ty:" << observation.y << "\tmin_distance:" << min_distance << endl;
		observations.at(i) = observation;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// cout << "updating weights for " << observations.size() << " observations" << endl;

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p = particles.at(i);
		// cout << "particle idx: " << i << "\tx:" << p.x << "\ty:" << p.y << endl;

		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++)
		{
			LandmarkObs observation = observations.at(j);
			LandmarkObs transformed_observation;

			// id remains the same
			transformed_observation.id = observation.id;

			// cout << "observation.x" << observation.x << "\tobservation.y" << observation.y << endl;
			// cout << "p.theta" << p.theta << endl;

			//transform to map x coordinate
			transformed_observation.x = p.x + (cos(p.theta) * observation.x) - (sin(p.theta) * observation.y);

			//transform to map y coordinate
			transformed_observation.y = p.y + (sin(p.theta) * observation.x) + (cos(p.theta) * observation.y);

			// cout << "transformed x:" << "\tx:" << transformed_observation.x << "\ty:" << transformed_observation.y << endl;
			transformed_observations.push_back(transformed_observation);
		}

		vector<LandmarkObs> predicted;
		for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
		{
			Map::single_landmark_s single_landmark_s = map_landmarks.landmark_list.at(k);
			// cout << "landmark idx" << k << "\tid:" << single_landmark_s.id_i << "\tx: " << single_landmark_s.x_f << "\ty: " << single_landmark_s.y_f << endl;
			double distance = sqrt(pow(single_landmark_s.x_f - p.x, 2) + pow(single_landmark_s.y_f - p.y, 2));
			if (distance <= sensor_range)
			{
				LandmarkObs landmark;
				landmark.id = single_landmark_s.id_i;
				landmark.x = single_landmark_s.x_f;
				landmark.y = single_landmark_s.y_f;

				// cout << "predicted id:" << landmark.id << "\tx:" << landmark.x << "\ty:" << landmark.y << "\tdistance:" << distance << endl;
				predicted.push_back(landmark);
			}
		}

		dataAssociation(predicted, transformed_observations);

		for (int l = 0; l < transformed_observations.size(); l++)
		{
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double x_obs = transformed_observations.at(l).x;
			double y_obs = transformed_observations.at(l).y;

			//TODO check if map_landmarks.landmark_list id's are in correct order
			int predicted_idx = transformed_observations.at(l).id - 1;
			// cout << "predicted_id id:" << predicted_idx << endl;

			double mu_x = map_landmarks.landmark_list.at(predicted_idx).x_f;
			double mu_y = map_landmarks.landmark_list.at(predicted_idx).y_f;

			//calculate normalization term
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
			// cout << "gauss_norm:" << gauss_norm << endl;

			//calculate exponent
			// cout << "x_obs:" << x_obs << "\tmu_x:" << mu_x << "\tsig_x:" << sig_x << endl;
			// cout << "y_obs:" << y_obs << "\tmu_y:" << mu_y << "\tsig_y:" << sig_y << endl;
			double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) + pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2));
			// double exponent = pow(x_obs - mu_x, 2) / (2 * sig_x) + pow(y_obs - mu_y, 2) / (2 * sig_y);
			// cout << "exponent:" << exponent << endl;

			//calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);
			// cout << "calc weight:" << weight << endl;
			p.weight = weight;
			//TODO: is this the correct way to assign a value?
			particles.at(i) = p;
			weights.at(i) = weight;
		}
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// cout << "resample" << endl;

	vector<Particle> resampled;
	resampled.clear();

	std::discrete_distribution<> d(weights.begin(), weights.end());
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i)
	{
		int idx = d(gen);
		resampled.push_back(particles.at(idx));
		// cout << "resampling: idx " << idx << "\tx:" << particles.at(idx).x << "\ty:" << particles.at(idx).y << endl;
	}

	particles.clear();
	particles = resampled;

	for (int i = 0; i < num_particles; ++i)
	{
		// cout << "after resample: idx " << i << "\tx:" << particles.at(i).x << "\ty:" << particles.at(i).y << endl;
	}
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
																				 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
