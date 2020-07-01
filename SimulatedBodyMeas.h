//
// Created by tomlucas on 14.05.20.
//

#ifndef CYLINDEREXAMPLE_SIMULATEDRADAR_H
#define CYLINDEREXAMPLE_SIMULATEDRADAR_H

#include <Eigen/Core>
#include "JetNorm.h"
class SimulatedRadar{
public:
    Eigen::Vector3d position;
    SimulatedRadar(Eigen::Vector3d position):position(position){

    }

    bool isVisible(const Eigen::Vector3d & target_world_pos)const {
        //return true;
        return (position.segment<2>(0)-target_world_pos.segment<2>(0)).norm() <70;
    }


    template<typename T>
    Eigen::Matrix<T,3,1> getRadar(const Eigen::Matrix<T,3,1> & target_world_pos, adekf::SO3<T> transform_world_to_body )const {
        return transform_world_to_body*(position-target_world_pos);

        }

};

#endif //CYLINDEREXAMPLE_SIMULATEDRADAR_H
