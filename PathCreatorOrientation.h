//
// Created by tomlucas on 14.05.20.
//

#ifndef CYLINDEREXAMPLE_PATHCREATOR_H
#define CYLINDEREXAMPLE_PATHCREATOR_H

#include <vector>
#include <Eigen/Core>
#include "types/SO3.h"


class PathCreator {
public:
    std::vector<Eigen::Vector3d> path;
    std::vector<adekf::SO3d> orientations;


    PathCreator(double deltaT,int factor) {
        deltaT = deltaT / factor;
        //std::cout << "DeltaT "<< deltaT<< std::endl;
        Eigen::Vector3d position{-100,0,0};
        adekf::SO3d orient{};
        Eigen::Vector3d body_velocity{10., 0, 0};

        Eigen::Vector3d ar{0, 0, M_PI /10.};
        size_t count = 0;
        for(int i=0; i < 6; i ++) {
            for (double t = 0; t < 10; t += deltaT) {
                position += orient * body_velocity * deltaT;
                count = (count + 1) % factor;
                if (count == 1) {
                    path.push_back(position);
                    orientations.push_back(orient);
                }
            }

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
