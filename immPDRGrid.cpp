//Better set: 	 Low Sigma: 0.1	 High Sigma: 8.4	 High to Low: 0.271	 Low to High: 0.271
//Better Error: 1.75361e+06
//Better set: 	 Low Sigma: 0.1	 High Sigma: 3.6	 High to Low: 0.213	 Low to High: 0.271
//Better Error: 1.04378e+06
//Better set: 	 Low Sigma: 0.1	 High Sigma: 2	 High to Low: 0.271	 Low to High: 0.271
//Better Error: 774065
//Current set: 	 Low Sigma: 0.02008	 High Sigma: 2.8	 High to Low: 0.213	 Low to High: 0.039



#include "ManifoldCreator.h"
#include "types/SO3.h"
// Declaration of State
ADEKF_MANIFOLD(XsensPose, ((adekf::SO3, orientation)), (3, position), (3, velocity), (3, acc_bias), (3, ar_bias))

#include "ADEKF.h"
#include "ADEKFUtils.h"
#include "adekf_viz.h"
#include "PoseRenderer.h"
#include "BPIMM.h"
#include "BPRTSIMM.h"
#include <numeric>

#define GRAVITY_CONSTANT 9.8029
//#define SHOW_VIZ

#include "rts-smoother.hpp"
#include "CSV_Reader.hpp"
#include <filesystem>
#include <regex>
//#define EKS
//#define MEKS
#define RTSIMMS
#define NIMMS
#define EKF_MODEL constant_turn_model
#define EKF_SIGMA ct_sigma
#define NOISE_ON_CONSTANT_TURN 0. // 10. for consistent evaluation // 0 for original implementation

struct imu_dynamic_model
{

    template <typename T, typename Noise>
    void operator()(XsensPose<T> &state, const Noise &noise, const Eigen::Vector3d &body_angular_rate, const Eigen::Vector3d &body_acceleration, double time_diff)
    {
        adekf::SO3<T> orient = state.orientation * adekf::SO3{(body_angular_rate + NOISE(0, 3)) * time_diff};
        state.orientation = orient;
        Eigen::Matrix<T, 3, 1> world_acceleration = (orient * (body_acceleration + NOISE(3, 3))) - Eigen::Vector3d(0, 0, GRAVITY_CONSTANT);
        state.position = state.position + state.velocity * time_diff + 0.5 * world_acceleration * pow(time_diff, 2);
        state.velocity = state.velocity + world_acceleration * time_diff;
    }
    // Create static object
} imu_dynamic_model_;

#define DATA_SIZE 9

namespace fs = std::filesystem;

std::list<fs::directory_entry> collectTrials(const fs::path &folder)
{
    std::list<fs::directory_entry> entries;
    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (entry.is_directory())
        {
            auto new_entries = collectTrials(entry);
            entries.insert(entries.end(), new_entries.begin(), new_entries.end());
        }
        else
        {
            std::smatch match;
            const std::string filename=(std::string)entry.path().filename();
            if (std::regex_search(filename,match, std::regex{".mat"}))
                entries.push_back(entry);
        }
    }
    return entries;
}

/**
 * @param argc argument counter
 * @param argv command line arguments
 * @return 0
 */
int main(int argc, char *argv[])
{
    constexpr double deltaT = 0.005;

    auto trials = collectTrials("/home/tkoller/repositories/pyshoe/data/vicon/processed/");
    std::cout << "Found number of Trials: " << trials.size() << std::endl;
    std::list<std::vector<double>> params;
    double steps = 10.;
    for (double sm_sig = 0.0001; sm_sig < 0.1; sm_sig += (0.1 - 0.0001) / steps)
    {
        for (double hi_sig = 2; hi_sig < 10; hi_sig += (10 - 2) / steps)
        {
            for (double high_to_low = 0.01; high_to_low < 0.3; high_to_low += (0.3 - 0.01) / steps)
            {
                for (double low_to_high = 0.01; low_to_high < 0.3; low_to_high += (0.3 - 0.01) / steps)
                {
                    std::vector<double> param_set;
                    param_set.push_back(sm_sig);
                    param_set.push_back(hi_sig);
                    param_set.push_back( high_to_low);
                    param_set.push_back(low_to_high);
                    params.push_front(param_set);
                }
            }
        }
    }
    double best_error=std::numeric_limits<double>::max();
    double best_set[4]={0,0,0,0};
    std::map<std::string,adekf::aligned_vector<Eigen::Matrix<double, DATA_SIZE, 1>>> datas;
    for (std::vector<double> param_set : params)
    {
        std::cout << "Current set: \t Low Sigma: " <<param_set[0] <<"\t High Sigma: " << param_set[1] << "\t High to Low: " << param_set[2] << "\t Low to High: " << param_set[3] << std::endl;
        double error_norm=0;
        for (auto trial : trials)
        {
            //std::cout << trial << std::endl;
            // std::cout << argv[1] << std::endl;

            if (datas.find(trial.path().string())==datas.end()){
            std::ifstream file{trial.path()};
            // read all positions from csv
            
            std::string line;
            getline(file, line);//Skip Header
            adekf::aligned_vector<Eigen::Matrix<double, DATA_SIZE, 1>> data;
            while (file.good())
            {
                Eigen::Matrix<double, DATA_SIZE, 1> imu_and_gt;
                if (zavi::csv_reader::readLineToEigen(file, imu_and_gt, ','))
                {
                    data.push_back(imu_and_gt);
                }
            }
            datas[trial.path().string()]=data;
            }
            adekf::aligned_vector<Eigen::Matrix<double, DATA_SIZE, 1>> data=datas[trial.path().string()];
            
            adekf::SquareMatrixType<double, 6> dyn_cov = dyn_cov.Zero();
            dyn_cov.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 2.89e-8; // Angular rate noise
            dyn_cov.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.0096;  // Acceleration noise

            // zupt model
            adekf::SquareMatrixType<double, 3> small_sigma = small_sigma.Identity() * param_set[0];
            adekf::SquareMatrixType<double, 3> high_sigma = high_sigma.Identity() * param_set[1];
            auto zupt_model = [](auto state)
            {
                return state.velocity;
            };
            auto step_model = [](auto state)
            {
                return state.velocity.norm();
            };

            XsensPose<double> start_state{};
            start_state.orientation = adekf::SO3{0.118385952267, 0.664553541062, 0.732492761361, -0.0883613071421}.conjugate();
            //if constexpr (DATA_SIZE == 10)
            XsensPoseCov start_cov = XsensPoseCov::Identity();
            start_cov.block<6, 6>(9, 9) = adekf::SquareMatrixType<double, 6>::Identity() * 0.01;
            // setup BP RTS IMM
            adekf::BPRTSIMM rts_imm{start_state, start_cov, {dyn_cov}, imu_dynamic_model_};
            rts_imm.addFilters({0, 0});

            // Setup of start conditions, each row has to sum up to 1.0, Eigen Matrices are Columnmajor
            Eigen::Matrix<double, 2, 2> t_prob;
            t_prob << 1.0 - param_set[3], param_set[2],
                param_set[3], 1.0 - param_set[2];
            rts_imm.setTransitionProbabilities(t_prob);
            Eigen::Vector2d start_prob(0.5, 0.5);
            rts_imm.setStartProbabilities(start_prob);

            for (auto data_point : data)
            {
                // std::cout << data_point.transpose() <<std::endl;
                // RTSIMM
                rts_imm.interaction();
                rts_imm.predictWithNonAdditiveNoise(data_point.segment<3>(3), data_point.segment<3>(0), deltaT);
                Eigen::Vector2d log_likelihood;
                rts_imm.getFilter(0).update(log_likelihood(0), zupt_model, small_sigma, Eigen::Vector3d::Zero());
                rts_imm.getFilter(1).update(log_likelihood(1), zupt_model, high_sigma, Eigen::Vector3d::Zero());
                Eigen::Vector2d likelihood = log_likelihood.unaryExpr([](auto elem)
                                                                      { return std::exp(elem); }); // calc probability
                rts_imm.passMeasurementProbabilities(likelihood);
                rts_imm.combination();

                error_norm+=std::abs(rts_imm.mu.position.norm() - (data_point.segment<3>(6)-data[0].segment<3>(6)).norm());
            }
        }

        std::cout << "Current Error: " << error_norm << std::endl;
        if (error_norm < best_error){
            best_error=error_norm;
            memcpy(best_set,&param_set[0],sizeof(double)*4);
            std::cout << "Better set: \t Low Sigma: " <<best_set[0] <<"\t High Sigma: " << best_set[1] << "\t High to Low: " << best_set[2] << "\t Low to High: " << best_set[3] << std::endl;
            std::cout << "Better Error: " << best_error << std::endl;

        }
    }
    std::cout << "Best set: \t Low Sigma: " <<best_set[0] <<"\t High Sigma: " << best_set[1] << "\t High to Low: " << best_set[2] << "\t Low to High: " << best_set[3] << std::endl;
    std::cout << "Best Error: " << best_error << std::endl;

    return 0;
}
