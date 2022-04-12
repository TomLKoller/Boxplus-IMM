#include "ManifoldCreator.h"
#include "types/SO3.h"
//Declaration of State
ADEKF_MANIFOLD(XsensPose,((adekf::SO3,orientation)),(3,position),(3,velocity),(3,acc_bias),(3,ar_bias))



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

//#define EKS
//#define MEKS
#define RTSIMMS
#define NIMMS
#define EKF_MODEL constant_turn_model
#define EKF_SIGMA ct_sigma
#define NOISE_ON_CONSTANT_TURN 0.//10. for consistent evaluation // 0 for original implementation



struct imu_dynamic_model
{
    
    template <typename T, typename Noise>
    void operator()(XsensPose<T> &state, const Noise &noise, const Eigen::Vector3d &body_angular_rate, const Eigen::Vector3d &body_acceleration, double time_diff)
    {   
    adekf::SO3<T> orient = state.orientation * adekf::SO3{(body_angular_rate+NOISE(0,3)) * time_diff};
    state.orientation = orient;
    Eigen::Matrix<T, 3, 1> world_acceleration = (orient * (body_acceleration+NOISE(3,3))) - Eigen::Vector3d(0,0,GRAVITY_CONSTANT);
    state.position = state.position + state.velocity * time_diff + 0.5 * world_acceleration * pow(time_diff, 2);
    state.velocity = state.velocity + world_acceleration * time_diff;
    }
    //Create static object
} imu_dynamic_model_;

#define DATA_SIZE 6

/**
 * @param argc argument counter
 * @param argv command line arguments
 * @return 0
 */
int main(int argc, char *argv[])
{



    constexpr double deltaT = 0.005;
    //std::cout << argv[1] << std::endl;
    std::ifstream file{"../pdr_data.csv"};
    // read all positions from csv
    adekf::aligned_vector<Eigen::Matrix<double,DATA_SIZE,1>> data;
    std::string line;
    getline(file,line);
    std::cout << line << std::endl;

    while (file.good())
    {
        Eigen::Matrix<double,DATA_SIZE,1> imu_and_gt;
        if (zavi::csv_reader::readLineToEigen(file, imu_and_gt, ','))
        {
           data.push_back(imu_and_gt);
        }
    }
    std::cout << "Data size is: " << data.size()<< std::endl;


    adekf::SquareMatrixType<double,6> dyn_cov=dyn_cov.Zero();
    dyn_cov.block<3,3>(0,0)=Eigen::Matrix3d::Identity()*2.89e-8; //Angular rate noise
    dyn_cov.block<3,3>(3,3)=Eigen::Matrix3d::Identity()*0.0096; //Acceleration noise
    
    //zupt model
    adekf::SquareMatrixType<double, 3> small_sigma = small_sigma.Identity() * 0.001;   
    adekf::SquareMatrixType<double, 3> high_sigma = high_sigma.Identity() * 3;
    auto zupt_model = [](auto state) {
        return state.velocity;
     };
  
  
        XsensPose<double> start_state{};
        start_state.orientation=adekf::SO3{0.118385952267,0.664553541062	,0.732492761361,	-0.0883613071421}.conjugate();
        if constexpr (DATA_SIZE==10)
            start_state.position=data[0].segment<3>(6);
        XsensPoseCov start_cov=XsensPoseCov::Identity();
        start_cov.block<6,6>(9,9)=adekf::SquareMatrixType<double,6>::Identity()*0.01;
        //setup BP RTS IMM
        adekf::BPRTSIMM rts_imm{start_state, start_cov, {dyn_cov}, imu_dynamic_model_};
        rts_imm.addFilters({0, 0});
        
    
        //Setup of start conditions
        Eigen::Matrix<double, 2, 2> t_prob;
        double transit=0.1;
        t_prob << 1.0-transit, transit,
           transit, 1.0-transit;
        rts_imm.setTransitionProbabilities(t_prob);
        Eigen::Vector2d start_prob(0.5, 0.5);
        rts_imm.setStartProbabilities(start_prob);
        adekf::viz::initGuis(argc, argv);
        adekf::viz::displayPose(&rts_imm,"red");
        std::ofstream output{"../trajectory_data.csv"};
        output << "px,py,pz,vx,vy,vz,roll,pitch,yaw" << std::endl;
        std::thread loop([&](){
        for(auto data_point: data){
            //std::cout << data_point.transpose() <<std::endl;
            //RTSIMM
            rts_imm.interaction();
            rts_imm.predictWithNonAdditiveNoise(data_point.segment<3>(3),data_point.segment<3>(0),deltaT);
            Eigen::Vector2d log_likelihood;
            rts_imm.getFilter(0).update(log_likelihood(0), zupt_model, small_sigma, Eigen::Vector3d::Zero()); 
            rts_imm.getFilter(1).update(log_likelihood(1), zupt_model, high_sigma, Eigen::Vector3d::Zero());
            Eigen::Vector2d likelihood=log_likelihood.unaryExpr([](auto elem) {return std::exp(elem);});//calc probability
            rts_imm.passMeasurementProbabilities(likelihood);
            rts_imm.combination();
            rts_imm.storeEstimation();
            auto euler_angles=rts_imm.mu.orientation.toRotationMatrix().eulerAngles(0,1,2);
            adekf::viz::plotVector(euler_angles,"orientation",data.size(),"rpy");
            adekf::viz::plotVector(data_point.segment<3>(0),"Angular Rate",data.size(),"xyz");
            //On purpose: Swapping of x and y. The pyShoe framework apparently uses a left hand system (inverted)
            output << rts_imm.mu.position.y()<< "," <<rts_imm.mu.position.x()<< ","<<rts_imm.mu.position.z()<< "," <<rts_imm.mu.velocity.y()<< "," <<rts_imm.mu.velocity.x()<< ","<<rts_imm.mu.velocity.z()<< ","
            <<euler_angles(0)<< ","<<euler_angles(1)<< ","<<euler_angles(2)<< std::endl;
            Eigen::Vector3d modes;
            if constexpr(DATA_SIZE==10){
                modes <<rts_imm.getModelProbabilities(),data_point[9];
            
                adekf::viz::plotVector(data_point.segment<3>(6),"GT Position",data.size(),"xyz");
                adekf::viz::plotVector(rts_imm.mu.position-data_point.segment<3>(6),"Position Error",data.size(),"xyz");
            
                }
            else{
                modes <<rts_imm.getModelProbabilities(),0;
            }
               adekf::viz::plotVector(modes,"Mode Probabilites",data.size(),"swd");
            adekf::viz::plotVector(rts_imm.mu.velocity,"Velocity",data.size(),"xyz");
            /*
            adekf::viz::plotVector(rts_imm.mu.position,"Position",data.size(),"xyz");
            adekf::viz::plotVector(rts_imm.mu.acc_bias,"Acc Bias",data.size(),"xyz");
            adekf::viz::plotVector(rts_imm.mu.ar_bias,"Ar Bias",data.size(),"xyz");
            
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));*/
           }
           output.close();
        });
       

       
     
    adekf::viz::runGuis();
    return 0;
}
