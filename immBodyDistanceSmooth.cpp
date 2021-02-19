#include "ADEKF.h"
#include "ManifoldCreator.h"
#include "types/SO3.h"
#include "adekf_viz.h"
#include "PoseRenderer.h"
#include "BPIMM.h"
#include "PathCreatorOrientation.h"
#include "SimulatedBodyMeas.h"
#include "GaussianNoiseVector.h"

#define GRAVITY_CONSTANT 9.81
#define SHOW_VIZ

//Declaration of State
ADEKF_MANIFOLD(CT_State, ((adekf::SO3, rotate_world_to_body)), (3, w_position), (3, w_velocity), (3, w_angular_rate))

//Declaration of measurement
ADEKF_MANIFOLD(Radar4, , (3, radar1), (3, radar2), (3, radar3), (3, radar4))


#include "NAIVE_IMM.h"
#include "rts-smoother.hpp"



/**
 * The constant turn model function.
 */
struct constant_turn_model {
    /**
     * Implements the constant turn dynamic.
     * @tparam T The Scalar type (required for auto diff)
     * @tparam Noise The type of the noise vector
     * @param state The state
     * @param noise The noise vector to apply the non additive noise
     * @param deltaT The time difference since the last measurement.
     */
    template<typename T, typename Noise>
    void operator()(CT_State<T> &state, const Noise &noise, double deltaT) {
        //pre calculate values
        T omega = norm(state.w_angular_rate);
        T c1 = omega == 0. ? -pow(deltaT, 2) * cos(omega * deltaT) / 2. : (cos(omega * deltaT) - 1.) / pow(omega, 2);
        T c2 = omega == 0. ? deltaT * cos(omega * deltaT) : sin(omega * deltaT) / omega;
        T c3 = omega == 0. ? -pow(deltaT, 3) * cos(omega * deltaT) / 6. : 1. / pow(omega, 2) * (sin(omega * deltaT) / omega - deltaT);
        T wx = state.w_angular_rate.x(), wy = state.w_angular_rate.y(), wz = state.w_angular_rate.z();
        T d1 = pow(wy, 2) + pow(wz, 2);
        T d2 = pow(wx, 2) + pow(wz, 2);
        T d3 = pow(wx, 2) + pow(wy, 2);

        //calcualte A and B Matrix according to [2]
        Eigen::Matrix<T, 3, 3> A, B;
        A << c1 * d1, -c2 * wz - c1 * wx * wy, c2 * wy - c1 * wx * wz,
                c2 * wz - c1 * wx * wy, c1 * d2, -c2 * wx - c1 * wy * wz,
                -c2 * wy - c1 * wx * wz, c2 * wx - c1 * wy * wz, c1 * d3;

        B << c3 * d1, c1 * wz - c3 * wx * wy, -c1 * wy - c3 * wx * wz,
                -c1 * wz - c3 * wx * wy, c3 * d2, c1 * wx - c3 * wy * wz,
                c1 * wy - c3 * wx * wz, -c1 * wx - c3 * wy * wz, c3 * d3;
        //Implement constant turn dynamic
        state.w_position += B * state.w_velocity + state.w_velocity * deltaT;
        state.w_velocity += A * state.w_velocity;
        state.rotate_world_to_body = state.rotate_world_to_body * adekf::SO3(state.w_angular_rate * deltaT).conjugate();
        state.w_angular_rate += noise * deltaT;
        

    };
    //Create static object
} constant_turn_model;

/**
 * Performs the simulation of a flying drone with camera.
 * @param argc argument counter
 * @param argv command line arguments
 * @return 0
 */
int main(int argc, char *argv[]) {
    constexpr double deltaT = 0.05;
    PathCreator path{deltaT, 10};



    //Setup covariance of constant turn model
    Eigen::Matrix<double, 3, 3> ct_sigma = ct_sigma.Identity()*0.1;

    //straight model
    auto straight_model = [](auto &state, auto noise, double deltaT) {
        state.w_velocity +=noise * deltaT;
        state.w_position += state.w_velocity * deltaT;
        //orientation and angular rate stay constant
    };
    //Setup covariance of straight model
    Eigen::Matrix<double, 3, 3> sm_sigma = sm_sigma.Identity()*10;


    auto free_model = [](auto &state, auto noise, double deltaT) {
        state.w_velocity +=NOISE(0,3) * deltaT;
        state.w_position += state.w_velocity * deltaT;
        state.w_angular_rate += NOISE(3,3) * deltaT;
        state.rotate_world_to_body = state.rotate_world_to_body * adekf::SO3(state.w_angular_rate * deltaT).conjugate();
       
    };
    Eigen::Matrix<double,6,1> fm_diag;
    fm_diag <<10.,10.,10.,.1,.1,.1;
    Eigen::Matrix<double, 6, 6> fm_sigma = fm_diag.asDiagonal();

    //Setup landmarks.
    SimulatedRadar radar1(Eigen::Vector3d(0, 40, -150)), radar2(Eigen::Vector3d(-120, 40, -150)), radar3(Eigen::Vector3d(-30, 0, -150)), radar4(
            Eigen::Vector3d(-90, 80, -150));
    //measurement model of 4 simultaneous landmark measurements
    auto radar_model = [](auto &state, const SimulatedRadar &radar1, const SimulatedRadar &radar2, const SimulatedRadar &radar3, const SimulatedRadar &radar4) {
        return Radar4{radar1.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                      radar2.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                      radar3.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                      radar4.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body)};
    };

    double rad_sigma = 1;
    GaussianNoiseVector radar_noise(0, rad_sigma, rad_sigma, rad_sigma);
    //Setup noise of measurement
    Eigen::Matrix<double, 12, 12> rm4_sigma = rm4_sigma.Zero();
    rm4_sigma.block<3, 3>(0, 0) =  rm4_sigma.block<3, 3>(3, 3) =rm4_sigma.block<3, 3>(6, 6) =    rm4_sigma.block<3, 3>(9, 9) = radar_noise.getCov();

    //Setup ekf
    adekf::ADEKF ekf{CT_State<double>(), Eigen::Matrix<double, 12, 12>::Identity()};
    ekf.mu.w_velocity.x() = 10.;
    ekf.mu.w_position.x() = -100;
    ekf.mu.w_angular_rate.z() = 0.0;
    //Setup Smoother
    adekf::RTS_Smoother smoother{ekf};

    //setup bp imm
    adekf::BPIMM imm{ekf.mu, ekf.sigma, {sm_sigma, ct_sigma}, straight_model, constant_turn_model};
    imm.addFilters({0, 1});

    //Setup of start conditions
    Eigen::Matrix<double, 2, 2> t_prob;
    t_prob << 0.95, 0.05,
            0.05, 0.95;
    imm.setTransitionProbabilities(t_prob);
    Eigen::Vector2d start_prob(0.5, 0.5);
    imm.setStartProbabilities(start_prob);

    //setup naive imm
    adekf::NAIVE_IMM bad_imm{ekf.mu, ekf.sigma, {sm_sigma, ct_sigma}, straight_model, constant_turn_model};
    bad_imm.addFilters({0, 1});

    //Setup of start conditions
    bad_imm.setTransitionProbabilities(t_prob);
    bad_imm.setStartProbabilities(start_prob);

    //Vectors to store path for evaluation
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > estimated_poses, imm_estimated_poses, bad_imm_estimated_poses, smoother_estimated_poses;
    double rmse_imm = 0, rmse_ekf = 0,rmse_smoother = 0, rmse_bad_imm = 0., rmse_bad_imm_to_imm=0.;
    for (size_t i = 0; i < path.path.size(); i++) {
        //read Ground truth path
        auto way_point = path.path[i];
        auto orient = path.orientations[i].conjugate();
        //perform interaction
        imm.interaction();
        bad_imm.interaction();
        //call predict methods
        ekf.predictWithNonAdditiveNoise(free_model, fm_sigma, deltaT);
        smoother.predictWithNonAdditiveNoise(free_model, fm_sigma, deltaT);
        imm.predictWithNonAdditiveNoise(deltaT);
        bad_imm.predictWithNonAdditiveNoise(deltaT);
        //filter updates
        Radar4<double> target{(radar1.getRadar(way_point, orient) + radar_noise.poll()),
                              (radar2.getRadar(way_point, orient) + radar_noise.poll()),
                              (radar3.getRadar(way_point, orient) + radar_noise.poll()),
                              (radar4.getRadar(way_point, orient) + radar_noise.poll())};
        imm.update(radar_model, rm4_sigma, target, radar1, radar2, radar3, radar4);
        bad_imm.update(radar_model, rm4_sigma, target, radar1, radar2, radar3, radar4);
        ekf.update(radar_model, rm4_sigma, target, radar1, radar2, radar3, radar4);
        smoother.update(radar_model, rm4_sigma, target, radar1, radar2, radar3, radar4);
        smoother.storeEstimation();


        imm.combination();
        bad_imm.combination();

        //calculate rmse in comparison to ground truth
        rmse_imm += pow((imm.mu.w_position - way_point).norm(), 2);
        rmse_ekf += pow((ekf.mu.w_position - way_point).norm(), 2);
        
        rmse_bad_imm += pow((bad_imm.mu.w_position - way_point).norm(), 2);
        rmse_bad_imm_to_imm+=pow((imm.mu.w_position - bad_imm.mu.w_position).norm(), 2);
        #ifdef SHOW_VIZ
        //store estimations for visualisation
        estimated_poses.push_back(ekf.mu.w_position);
        imm_estimated_poses.push_back(imm.mu.w_position);
        bad_imm_estimated_poses.push_back(bad_imm.mu.w_position);
        #endif //SHOW_VIZ
    }
    smoother.smoothAllWithNonAdditiveNoise(free_model, fm_sigma, deltaT);
    //smoother.smoothIntervalWithNonAdditiveNoise(2,-1,free_model, fm_sigma, deltaT);
    for (size_t i = 0; i < path.path.size(); i++) {
        if(smoother.smoothed_mus[i].w_position.allFinite()){
        rmse_smoother += pow((smoother.smoothed_mus[i].w_position - path.path[i]).norm(), 2);
        smoother_estimated_poses.push_back(smoother.smoothed_mus[i].w_position);
        }
    }

    std::cout << "EKF RMSE: " << sqrt(rmse_ekf / path.path.size()) << "\t Smoother RMSE: " << sqrt(rmse_smoother / smoother_estimated_poses.size()) << "\t IMM RMSE: " << sqrt(rmse_imm / path.path.size()) << "\t BAD IMM RMSE: "
              << sqrt(rmse_bad_imm / path.path.size()) << " \t ERROR DIFF: " << sqrt(rmse_imm / path.path.size()) - sqrt(rmse_bad_imm / path.path.size())<< "\t IMM-BAD IMM RMSE: " << sqrt(rmse_bad_imm_to_imm / path.path.size()) 
              << std::endl;

    
    #ifdef SHOW_VIZ
    //visualize paths
    adekf::viz::initGuis(argc, argv);
    adekf::viz::PoseRenderer::displayPath(path.path, "red");
    adekf::viz::PoseRenderer::displayPath(estimated_poses, "black");
    adekf::viz::PoseRenderer::displayPath(imm_estimated_poses, "green");
    adekf::viz::PoseRenderer::displayPath(bad_imm_estimated_poses, "blue");
    adekf::viz::PoseRenderer::displayPath(smoother_estimated_poses, "orange");
    adekf::viz::PoseRenderer::displayPoints({radar1.position, radar2.position, radar3.position, radar4.position}, "red", 5);
    adekf::viz::runGuis();
    #endif //SHOW_VIZ
    return 0;
}


