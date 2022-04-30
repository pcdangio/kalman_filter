#include <kalman_filter/ukf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
ukf::ukf(uint32_t n_variables, uint32_t n_observers)
    : base(n_variables, n_observers)
{
    // Calculate number of sigma points.
    ukf::n_s = 1 + 2*ukf::n_x;

    // Allocate weight vector.
    ukf::wj.setZero(ukf::n_s);

    // Allocate sigma matrices
    ukf::X.setZero(ukf::n_x, ukf::n_s);
    ukf::Z.setZero(ukf::n_z, ukf::n_s);

    // Allocate interface components.
    ukf::i_xp.setZero(ukf::n_x);
    ukf::i_x.setZero(ukf::n_x);
    ukf::i_z.setZero(ukf::n_z);

    // Allocate temporaries.
    ukf::t_xs.setZero(ukf::n_x, ukf::n_s);
    ukf::t_zs.setZero(ukf::n_z, ukf::n_s);

    // Set default parameters.
    ukf::wo = 0.1;
}

// FILTER METHODS
void ukf::iterate()
{
    // ---------- STEP 1: PREPARATION ----------

    // Calculate weight vector for mean and covariance averaging.
    ukf::wj.fill((1.0 - ukf::wo)/(2.0 * static_cast<double>(ukf::n_x)));
    ukf::wj[0] = ukf::wo;

    // ---------- STEP 2: PREDICT ----------

    // Populate previous state sigma matrix
    // Calculate square root of P using Cholseky Decomposition
    ukf::llt.compute(ukf::P);
    // Check if calculation succeeded (positive semi definite)
    if(ukf::llt.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error("covariance matrix P is not positive semi definite (predict)");
    }
    // Reset first column of X.
    ukf::X.col(0).setZero();
    // Fill X with +sqrt(P)
    ukf::X.block(0,1,ukf::n_x,ukf::n_x) = ukf::llt.matrixL();
    // Fill X with -sqrt(P)
    ukf::X.block(0,1+ukf::n_x,ukf::n_x,ukf::n_x) = -1.0 * ukf::X.block(0,1,ukf::n_x,ukf::n_x);
    // Apply sqrt(n+lambda) to entire matrix.
    ukf::X *= std::sqrt(static_cast<double>(ukf::n_x) / (1.0 - ukf::wo));
    // Add mean to entire matrix.
    ukf::X += ukf::x.replicate(1,ukf::n_s);

    // Pass previous X through state transition function.
    for(uint32_t s = 0; s < ukf::n_s; ++s)
    {
        // Populate interface vector.
        ukf::i_xp = ukf::X.col(s);
        ukf::i_x.setZero();
        // Evaluate state transition.
        state_transition(ukf::i_xp, ukf::i_x);
        // Store result back in X.
        ukf::X.col(s) = ukf::i_x;
    }

    // Calculate predicted state mean.
    ukf::x.noalias() = ukf::X * ukf::wj;

    // Calculate predicted state covariance.
    ukf::X -= ukf::x.replicate(1, ukf::n_s);
    ukf::t_xs.noalias() = ukf::X * ukf::wj.asDiagonal();
    ukf::P.noalias() = ukf::t_xs * ukf::X.transpose();
    ukf::P += ukf::Q;

    // Log predicted state.
    ukf::log_predicted_state();

    // ---------- STEP 3: UPDATE ----------

    // Check if update is necessary.
    if(ukf::has_observations())
    {
        // Populate predicted state sigma matrix.
        // Calculate square root of P using Cholseky Decomposition
        ukf::llt.compute(ukf::P);
        // Check if calculation succeeded (positive semi definite)
        if(ukf::llt.info() != Eigen::ComputationInfo::Success)
        {
            throw std::runtime_error("covariance matrix P is not positive semi definite (update)");
        }
        // Reset first column of X.
        ukf::X.col(0).setZero();
        // Fill X with +sqrt(P)
        ukf::X.block(0,1,ukf::n_x,ukf::n_x) = ukf::llt.matrixL();
        // Fill X with -sqrt(P)
        ukf::X.block(0,1+ukf::n_x,ukf::n_x,ukf::n_x) = -1.0 * ukf::X.block(0,1,ukf::n_x,ukf::n_x);
        // Apply sqrt(n+lambda) to entire matrix.
        ukf::X *= std::sqrt(static_cast<double>(ukf::n_x) / (1.0 - ukf::wo));
        // Add mean to entire matrix.
        ukf::X += ukf::x.replicate(1,ukf::n_s);

        // Pass predicted X through state transition function.
        for(uint32_t s = 0; s < ukf::n_s; ++s)
        {
            // Populate interface vector.
            ukf::i_x = ukf::X.col(s);
            ukf::i_z.setZero();
            // Evaluate state transition.
            observation(ukf::i_x, ukf::i_z);
            // Store result back in X.
            ukf::Z.col(s) = ukf::i_z;
        }

        // Calculate predicted observation mean.
        ukf::z.noalias() = ukf::Z * ukf::wj;

        // Log predicted observation.
        ukf::log_observations();

        // Calculate predicted observation covariance.
        ukf::Z -= ukf::z.replicate(1, ukf::n_s);
        ukf::t_zs.noalias() = ukf::Z * ukf::wj.asDiagonal();
        ukf::S.noalias() = ukf::t_zs * ukf::Z.transpose();
        ukf::S += ukf::R;

        // Calculate predicted state/observation covariance.
        ukf::X -= ukf::x.replicate(1, ukf::n_s);
        ukf::t_xs.noalias() = ukf::X * ukf::wj.asDiagonal();
        ukf::C.noalias() = ukf::t_xs * ukf::Z.transpose();

        // Run masked Kalman update.
        ukf::masked_kalman_update();
    }
    else
    {
        // Log empty observations.
        ukf::log_observations(true);
    }

    // Log estimated state.
    ukf::log_estimated_state();
}