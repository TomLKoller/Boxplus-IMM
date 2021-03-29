//
// Created by tomlucas on 14.05.20.
//

#ifndef SIMULATEDRADAR_H
#define SIMULATEDRADAR_H

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
    Eigen::Matrix<T,3,1> getRadar(const Eigen::Matrix<T,3,1> & target_world_pos)const {
        return (position-target_world_pos);

        }

};

#endif //SIMULATEDRADAR_H
