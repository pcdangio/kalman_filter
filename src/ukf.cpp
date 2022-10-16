#include <kalman_filter/ukf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukf_t::ukf_t(uint32_t n_variables, uint32_t n_observers)
    : base_t(n_variables, n_observers)
{
    // Calculate number of sigma points.
    ukf_t::n_s = 1 + 2*ukf_t::n_x;

    // Allocate weight vector.
    ukf_t::wj.setZero(ukf_t::n_s);

    // Allocate sigma matrices
    ukf_t::Xs.setZero(ukf_t::n_x, ukf_t::n_s);
    ukf_t::X.setZero(ukf_t::n_x, ukf_t::n_s);
    ukf_t::Z.setZero(ukf_t::n_z, ukf_t::n_s);

    // Allocate temporaries.
    ukf_t::t_xs.setZero(ukf_t::n_x, ukf_t::n_s);
    ukf_t::t_zs.setZero(ukf_t::n_z, ukf_t::n_s);

    // Set default parameters.
    ukf_t::wo = 0.1;
}

// FILTER METHODS
#include <iostream>
void ukf_t::iterate()
{
    // ---------- STEP 1: PREPARATION ----------

    // Calculate weight vector for mean and covariance averaging.
    ukf_t::wj.fill((1.0 - ukf_t::wo)/(2.0 * static_cast<double>(ukf_t::n_x)));
    ukf_t::wj[0] = ukf_t::wo;

    // ---------- STEP 2: PREDICT ----------

    // Populate previous state sigma matrix
    // Calculate square root of P using Cholseky Decomposition
    ukf_t::llt.compute(ukf_t::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
    {
        std::cout << ukf_t::P << std::endl;
        throw std::runtime_error("covariance matrix P is not positive semi definite (predict)");
    }
    // Reset first column of Xs.
    ukf_t::Xs.col(0).setZero();
    // Fill Xs with +sqrt(P)
    ukf_t::Xs.block(0,1,ukf_t::n_x,ukf_t::n_x) = ukf_t::llt.matrixL();
    // Fill Xs with -sqrt(P)
    ukf_t::Xs.block(0,1+ukf_t::n_x,ukf_t::n_x,ukf_t::n_x) = -1.0 * ukf_t::Xs.block(0,1,ukf_t::n_x,ukf_t::n_x);
    // Apply sqrt(n+lambda) to entire matrix.
    ukf_t::Xs *= std::sqrt(static_cast<double>(ukf_t::n_x) / (1.0 - ukf_t::wo));
    // Add mean to entire matrix.
    ukf_t::Xs += ukf_t::x.replicate(1,ukf_t::n_s);

    // Perform state transition.
    // Reset X output matrix.
    ukf_t::X.setZero();
    // Pass Xs through state transition function.
    for(uint32_t s = 0; s < ukf_t::n_s; ++s)
    {
        // Normalize prior state vector.
        ukf_t::normalize_state(ukf_t::Xs.col(s));
        // Evaluate state transition.
        state_transition(ukf_t::Xs.col(s), ukf_t::X.col(s));
    }

    // Calculate predicted state mean.
    ukf_t::x.noalias() = ukf_t::X * ukf_t::wj;
    // Normalize predicted state mean.
    ukf_t::normalize_state(ukf_t::x);

    // Calculate predicted state covariance.
    ukf_t::X -= ukf_t::x.replicate(1, ukf_t::n_s);
    ukf_t::t_xs.noalias() = ukf_t::X * ukf_t::wj.asDiagonal();
    ukf_t::P.noalias() = ukf_t::t_xs * ukf_t::X.transpose();
    ukf_t::P += ukf_t::Q;

    // Log predicted state.
    ukf_t::log_predicted_state();

    // ---------- STEP 3: UPDATE ----------

    // Check if update is necessary.
    if(ukf_t::has_observations())
    {
        // Populate predicted state sigma matrix.
        // Calculate square root of P using Cholseky Decomposition
        ukf_t::llt.compute(ukf_t::P);
        // Check if calculation succeeded (positive semi definite)
        if(ukf_t::llt.info() != Eigen::ComputationInfo::Success)
        {
            std::cout << ukf_t::P << std::endl;
            throw std::runtime_error("covariance matrix P is not positive semi definite (update)");
        }
        // Reset first column of Xs.
        ukf_t::Xs.col(0).setZero();
        // Fill Xs with +sqrt(P)
        ukf_t::Xs.block(0,1,ukf_t::n_x,ukf_t::n_x) = ukf_t::llt.matrixL();
        // Fill Xs with -sqrt(P)
        ukf_t::Xs.block(0,1+ukf_t::n_x,ukf_t::n_x,ukf_t::n_x) = -1.0 * ukf_t::Xs.block(0,1,ukf_t::n_x,ukf_t::n_x);
        // Apply sqrt(n+lambda) to entire matrix.
        ukf_t::Xs *= std::sqrt(static_cast<double>(ukf_t::n_x) / (1.0 - ukf_t::wo));
        // Add mean to entire matrix.
        ukf_t::Xs += ukf_t::x.replicate(1,ukf_t::n_s);

        // Calculate observation matrix.
        // Reset observation matrix.
        ukf_t::Z.setZero();
        // Pass sigma points through observation function.
        for(uint32_t s = 0; s < ukf_t::n_s; ++s)
        {
            // Normalize state vector.
            ukf_t::normalize_state(ukf_t::Xs.col(s));
            // Pass through observation function.
            observation(ukf_t::Xs.col(s), ukf_t::Z.col(s));
        }

        // Calculate predicted observation mean.
        ukf_t::z.noalias() = ukf_t::Z * ukf_t::wj;

        // Log predicted observation.
        ukf_t::log_observations();

        // Calculate predicted observation covariance.
        ukf_t::Z -= ukf_t::z.replicate(1, ukf_t::n_s);
        ukf_t::t_zs.noalias() = ukf_t::Z * ukf_t::wj.asDiagonal();
        ukf_t::S.noalias() = ukf_t::t_zs * ukf_t::Z.transpose();
        ukf_t::S += ukf_t::R;

        // Calculate predicted state/observation covariance.
        ukf_t::Xs -= ukf_t::x.replicate(1, ukf_t::n_s);
        ukf_t::t_xs.noalias() = ukf_t::Xs * ukf_t::wj.asDiagonal();
        ukf_t::C.noalias() = ukf_t::t_xs * ukf_t::Z.transpose();

        // Run masked Kalman update.
        ukf_t::masked_kalman_update();
    }
    else
    {
        // Log empty observations.
        ukf_t::log_observations(true);
    }

    // Log estimated state.
    ukf_t::log_estimated_state();
}