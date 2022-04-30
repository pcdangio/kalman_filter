#include <kalman_filter/kf.hpp>

using namespace kalman_filter;

// CONSTRUCTORS
kf::kf(uint32_t n_variables, uint32_t n_inputs, uint32_t n_observers)
    : base(n_variables, n_observers)
{
    // Store dimensions.
    kf::n_u = n_inputs;

    // Initialize model matrices.
    kf::A.setIdentity(kf::n_x, kf::n_x);
    kf::B.setZero(kf::n_x, kf::n_u);
    kf::H.setZero(kf::n_z, kf::n_x);

    // Initialize input vector.
    kf::u.setZero(kf::n_u);

    // Initialize temporaries.
    kf::t_x.setZero(kf::n_x);
    kf::t_xx.setZero(kf::n_x, kf::n_x);
    kf::t_zx.setZero(kf::n_z, kf::n_x);
}

// FILTER METHODS
void kf::iterate()
{
    // ---------- STEP 1: PREDICT ----------

    // Predict state.
    kf::t_x.noalias() = kf::A * kf::x;
    kf::x.noalias() = kf::B * kf::u;
    kf::x += kf::t_x;

    // Log predicted state.
    kf::log_predicted_state();

    // Predict covariance.
    kf::t_xx.noalias() = kf::A * kf::P;
    kf::P.noalias() = kf::t_xx * kf::A.transpose();
    kf::P += kf::Q;

    // ---------- STEP 2: UPDATE ----------

    // Check if update is necessary.
    if(kf::has_observations())
    {
        // Calculate predicted observation.
        kf::z.noalias() = kf::H * kf::x;

        // Log observations.
        kf::log_observations();
        
        // Calculate predicted observation covariance.
        kf::t_zx.noalias() = kf::H * kf::P;
        kf::S.noalias() = kf::t_zx * kf::H.transpose();
        kf::S += kf::R;

        // Calculate predicted state/observation cross covariance.
        kf::C.noalias() = kf::P * kf::H.transpose();

        // Perform masked kalman update.
        kf::masked_kalman_update();
    }
    else
    {
        // Log empty observations.
        kf::log_observations(true);
    }

    // Log estimated state.
    kf::log_estimated_state();
}
void kf::new_input(uint32_t input_index, double_t input)
{
    // Verify index exists.
    if(!(input_index < kf::n_u))
    {
        throw std::runtime_error("failed to set new input (input_index out of range)");
    }

    // Store input.
    kf::u(input_index) = input;
}

// ACCESS
uint32_t kf::n_inputs() const
{
    return kf::n_u;
}