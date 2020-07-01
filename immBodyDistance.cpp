#include "ADEKF.h"
#include "ManifoldCreator.h"
#include "types/SO3.h"
#include <ADEKF/viz/adekf_viz.h>
#include <ADEKF/viz/PoseRenderer.h>
#include "ADIMM.h"
#include "PathCreatorOrientation.h"
#include "SimulatedBodyMeas.h"
#include "GaussianNoiseVector.h"

#define GRAVITY_CONSTANT 9.81


ADEKF_MANIFOLD(CT_State, ((adekf::SO3, rotate_world_to_body)), (3, w_position), (3, w_velocity), (3, w_angular_rate))
ADEKF_MANIFOLD(Radar4,,(3,radar1),(3,radar2),(3,radar3),(3,radar4))
ADEKF_MANIFOLD(Radar2,,(3,radar1),(3,radar2))


#include "NAIVE_IMM.h"


#define RADAR_ALWAYS_VISIBLE true


#define THIRD_TEST
#define DELTA_T_STEP 0.05
#define DELTA_T_MAX 0.08

struct constant_turn_model{
    template<typename T, typename Noise>
    void operator() (CT_State<T> &state,const Noise & noise, double deltaT) {

        T omega=norm(state.w_angular_rate);
        T c1=omega ==0.? -pow(deltaT,2)*cos(omega *deltaT)/2. : (cos(omega*deltaT)-1.)/pow(omega,2);
        T c2= omega ==0.? deltaT*cos(omega*deltaT) : sin(omega*deltaT)/omega;
        T c3= omega ==0.?-pow(deltaT,3)*cos(omega*deltaT)/6. : 1./pow(omega,2)*(sin(omega*deltaT)/omega-deltaT);
        T wx=state.w_angular_rate.x(),wy=state.w_angular_rate.y(),wz=state.w_angular_rate.z();
        T d1=pow(wy,2)+pow(wz,2);
        T d2=pow(wx,2)+pow(wz,2);
        T d3=pow(wx,2)+pow(wy,2);

        Eigen::Matrix<T,3,3> A,B;
        A <<    c1 *d1,            -c2*wz-c1*wx*wy,    c2*wy-c1*wx*wz,
                c2*wz-c1*wx*wy,     c1*d2,              -c2*wx-c1*wy*wz,
                -c2*wy-c1*wx*wz,    c2*wx-c1*wy*wz,     c1*d3;

        B <<    c3*d1,              c1*wz-c3*wx*wy,     -c1*wy-c3*wx*wz,
                -c1*wz-c3*wx*wy,    c3*d2,              c1*wx-c3*wy*wz,
                c1*wy-c3*wx*wz,     -c1*wx-c3*wy*wz,    c3*d3;

        state.w_position+=B*state.w_velocity+state.w_velocity*deltaT;
        state.w_velocity+=A*state.w_velocity;
        state.rotate_world_to_body=state.rotate_world_to_body*adekf::SO3(state.w_angular_rate*deltaT).conjugate();
        state.w_angular_rate+=NOISE(9,3)*deltaT;

    };
} constant_turn_model;


int main(int argc, char *argv[]) {
    adekf::viz::initGuis(argc, argv);

    for(double deltaT=0.05; deltaT < DELTA_T_MAX; deltaT+=DELTA_T_STEP) {
        std::cout << "Delta T: " << deltaT << std::endl;
        PathCreator path{deltaT, 10};
        //adekf::viz::PoseRenderer::displayPath(path.path, "red");




        Eigen::Matrix<double, 12, 12> ct_sigma = ct_sigma.Zero();

        ct_sigma(0, 0) = ct_sigma(1, 1) = ct_sigma(2, 2) = 0.0;
        ct_sigma(3, 3) = ct_sigma(4, 4) = ct_sigma(5, 5) = 0;
        ct_sigma(6, 6) = ct_sigma(7, 7) = ct_sigma(8, 8) = 0.;
        ct_sigma(9, 9) = ct_sigma(10, 10) = ct_sigma(11, 11) = 0.1;



        auto general_model = [](auto &state,auto noise, double deltaT) {
            state.w_velocity+=NOISE(6,3)*deltaT;
            //state.rotate_body_to_world=state.rotate_body_to_world*adekf::SO3(state.b_angular_rate*deltaT);
            state.w_position += state.w_velocity * deltaT+NOISE(3,3);
            //state.rotate_world_to_body=state.rotate_world_to_body*adekf::SO3(state.w_angular_rate*deltaT).conjugate();
            //state.w_angular_rate+=NOISE(9,3)*deltaT;

        };
        Eigen::Matrix<double, 12, 12> gm_sigma = gm_sigma.Zero();
        gm_sigma(0, 0) = gm_sigma(1, 1) = gm_sigma(2, 2) = 0.;
        gm_sigma(3, 3) = gm_sigma(4, 4) = gm_sigma(5, 5) = 0.;
        gm_sigma(6, 6) = gm_sigma(7, 7) = gm_sigma(8, 8) = 10;
        gm_sigma(9, 9) = gm_sigma(10, 10) = gm_sigma(11, 11) = 1;


        SimulatedRadar radar1(Eigen::Vector3d(0, 40, -150)), radar2(Eigen::Vector3d(-120, 40, -150)), radar3(Eigen::Vector3d(-30, 0, -150)), radar4(
                Eigen::Vector3d(-90, 80, -150));
        auto radar_model = [](auto &state, const SimulatedRadar& radar1, const SimulatedRadar& radar2, const SimulatedRadar& radar3,const SimulatedRadar& radar4,const Eigen::Vector3d & way_point) {
            return Radar4{(RADAR_ALWAYS_VISIBLE || radar1.isVisible(way_point) ? 1. : 0.) *
                          radar1.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                          (RADAR_ALWAYS_VISIBLE || radar2.isVisible(way_point) ? 1. : 0.) *
                          radar2.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                          (RADAR_ALWAYS_VISIBLE || radar3.isVisible(way_point) ? 1. : 0.) *
                          radar3.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body),
                          (RADAR_ALWAYS_VISIBLE || radar4.isVisible(way_point) ? 1. : 0.) *
                          radar4.getRadar<ScalarOf(state)>(state.w_position, state.rotate_world_to_body)};
        };

        //adekf::viz::PoseRenderer::displayPoints({radar1.position, radar2.position, radar3.position, radar4.position}, "red", 5);
        double rad_sigma = 1;
        GaussianNoiseVector radar_noise1(0, rad_sigma, rad_sigma, rad_sigma);
        GaussianNoiseVector radar_noise2(0, rad_sigma, rad_sigma, rad_sigma);
        GaussianNoiseVector radar_noise3(0, rad_sigma, rad_sigma, rad_sigma);
        GaussianNoiseVector radar_noise4(0, rad_sigma, rad_sigma, rad_sigma);
        auto rm_sigma = radar_noise1.getCov();
        Eigen::Matrix<double,12,12> rm4_sigma=rm4_sigma.Zero();
        rm4_sigma.block<3,3>(0,0)=rm_sigma;
        rm4_sigma.block<3,3>(3,3)=rm_sigma;
        rm4_sigma.block<3,3>(6,6)=rm_sigma;
        rm4_sigma.block<3,3>(9,9)=rm_sigma;
        std::cout << "Radar noise: \n" << rm_sigma << std::endl;
        adekf::ADEKF ekf{CT_State<double>(), Eigen::Matrix<double, 12, 12>::Identity()};
        ekf.mu.w_velocity.x() = 10.;
        ekf.mu.w_position.x() = -100;
        ekf.mu.w_angular_rate.z()=0.0;
        adekf::ADIMM imm{ekf.mu, ekf.sigma, {gm_sigma, ct_sigma}, general_model, constant_turn_model};
        imm.addModels({0, 1});

        //Setup of start conditions
        Eigen::Matrix<double, 2, 2> t_prob;
        t_prob << 0.95, 0.05,
                0.05, 0.95;
        imm.setTransitionProbabilities(t_prob);
        Eigen::Vector2d start_prob(0.5, 0.5);
        imm.setStartProbabilities(start_prob);

        adekf::NAIVE_IMM bad_imm{ekf.mu, ekf.sigma, {gm_sigma, ct_sigma}, general_model, constant_turn_model};
        bad_imm.addModels({0, 1});

        //Setup of start conditions
        bad_imm.setTransitionProbabilities(t_prob);
        bad_imm.setStartProbabilities(start_prob);

        std::vector<Eigen::Vector3d> estimated_poses, imm_estimated_poses, bad_imm_estimated_poses;
        double rmse_imm = 0, rmse_ekf = 0, rmse_bad_imm = 0.;
        for (size_t i = 0; i < path.path.size(); i++) {
            auto way_point = path.path[i];
            auto orient = path.orientations[i].conjugate();
            imm.interaction();
            bad_imm.interaction();
            ekf.predictWithNonAdditiveNoise(constant_turn_model, ct_sigma, deltaT);
            imm.predictWithNonAdditiveNoise(deltaT);
            bad_imm.predictWithNonAdditiveNoise(deltaT);
            //filter iteration
            double likelihood;
            Eigen::Matrix<double, 1, 1> num_visible{0};

            Radar4<double> target{(RADAR_ALWAYS_VISIBLE || radar1.isVisible(way_point) ?1.:0.)*(radar1.getRadar(way_point, orient) + radar_noise1.poll()),(RADAR_ALWAYS_VISIBLE || radar2.isVisible(way_point) ?1.:0.)*(radar2.getRadar(way_point, orient) + radar_noise2.poll()),(RADAR_ALWAYS_VISIBLE || radar3.isVisible(way_point) ?1.:0.)*(radar3.getRadar(way_point, orient) + radar_noise3.poll()),(RADAR_ALWAYS_VISIBLE || radar4.isVisible(way_point) ?1.:0.)*(radar4.getRadar(way_point, orient) + radar_noise4.poll())};
            imm.update(radar_model,rm4_sigma,target,radar1,radar2,radar3,radar4,way_point);
            bad_imm.update(radar_model,rm4_sigma,target,radar1,radar2,radar3,radar4,way_point);
            ekf.update(radar_model,rm4_sigma,target,radar1,radar2,radar3,radar4,way_point);

            imm.combination();
            bad_imm.combination();
            rmse_imm += pow((imm.mu.w_position - way_point).norm(), 2);
            rmse_ekf += pow((ekf.mu.w_position - way_point).norm(), 2);
            rmse_bad_imm += pow((bad_imm.mu.w_position - way_point).norm(), 2);

            estimated_poses.push_back(ekf.mu.w_position);
            imm_estimated_poses.push_back(imm.mu.w_position);
            bad_imm_estimated_poses.push_back(bad_imm.mu.w_position);
        }
        std::cout << "EKF RMSE: " << sqrt(rmse_ekf / path.path.size()) << "\t IMM RMSE: " << sqrt(rmse_imm / path.path.size()) << "\t BAD IMM RMSE: "
                  << sqrt(rmse_bad_imm / path.path.size()) << " \t ERROR DIFF: " << sqrt(rmse_imm / path.path.size()) - sqrt(rmse_bad_imm / path.path.size())
                  << std::endl;
        //adekf::viz::PoseRenderer::displayPath(estimated_poses, "black");
        //adekf::viz::PoseRenderer::displayPath(imm_estimated_poses, "red");
        //adekf::viz::PoseRenderer::displayPath(bad_imm_estimated_poses, "blue");
        Eigen::Vector4d rmses{0.,sqrt(rmse_imm / path.path.size()),sqrt(rmse_bad_imm / path.path.size()),sqrt(rmse_imm / path.path.size()) - sqrt(rmse_bad_imm / path.path.size())};
        adekf::viz::plotVector(rmses,"RMSE comparison",DELTA_T_MAX/DELTA_T_STEP,"eibd");
    }
    adekf::viz::runGuis();

    return 0;
}


