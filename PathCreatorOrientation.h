//
// Created by tomlucas on 14.05.20.
//

#ifndef CYLINDEREXAMPLE_PATHCREATOR_H
#define CYLINDEREXAMPLE_PATHCREATOR_H

#include <vector>
#include <Eigen/Core>
#include "types/SO3.h"

/**
 * Class to create the Ground truth.
 *
 * Creates a path  and stores the complete pose of it.
 */
class PathCreator {
public:
    //The positions alongside the path
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> path;
    //The orientations along the path
    std::vector<adekf::SO3d> orientations;

    /**
     * Sets up a path consisting of 2 straights and 2 180 degree curves.
     *
     * Creates a path that has a pose entry for every deltaT seconds.
     * Creates "factor" points between two entries for higher accuracy of the path without storing them.
     *
     * @param deltaT The time difference between two  measurements
     * @param factor A factor to determine number of samples in between points (to improve the numeric approximation of the curved path)
     */
    PathCreator(double deltaT,int factor) {
        deltaT = deltaT / factor;
        //std::cout << "DeltaT "<< deltaT<< std::endl;
        Eigen::Vector3d position{-100,0,0};
        adekf::SO3d orient{};
        Eigen::Vector3d body_velocity{10., 0, 0};

        Eigen::Vector3d ar{0, 0, M_PI /10.};
        size_t count = 0;
        //repeat 6 times (2 and a half round
        for(int i=0; i < 10; i ++) {
            //a straight part
            for (double t = 0; t < 6; t += deltaT) {
                position += orient * body_velocity * deltaT;
                count = (count + 1) % factor;
                if (count == 1) {
                    path.push_back(position);
                    orientations.push_back(orient);
                }
            }
            // a curved part
            for (double t = 0; t < 10.; t += deltaT) {
                orient = orient * adekf::SO3(ar * deltaT);
                position += orient * body_velocity * deltaT;
                count = (count + 1) % factor;
                if (count == 1) {
                    path.push_back(position);
                    orientations.push_back(orient);
                }
            }


        }

    }
};


#endif //CYLINDEREXAMPLE_PATHCREATOR_H
