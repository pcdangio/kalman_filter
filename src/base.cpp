#include <kalman_filter/base.hpp>

#include <fstream>
#include <iomanip>

using namespace kalman_filter;

// CONSTRUCTORS
base::base(uint32_t n_variables, uint32_t n_observers)
{
    // Store dimension sizes.
    base::n_x = n_variables;
    base::n_z = n_observers;

    // Allocate prediction components.
    base::x.setZero(base::n_x);
    base::P.setIdentity(base::n_x, base::n_x);
    base::Q.setIdentity(base::n_x, base::n_x);

    // Allocate update components.
    base::R.setIdentity(base::n_z, base::n_z);
    base::z.setZero(base::n_z);
    base::S.setZero(base::n_z, base::n_z);
    base::C.setZero(base::n_x, base::n_z);

    // Allocate temporaries.
    base::t_xx.setZero(base::n_x, base::n_x);
}
base::~base()
{
    // Stop logging if running.
    base::stop_log();
}

// FILTER METHODS
void base::new_observation(uint32_t observer_index, double_t observation)
{
    // Verify index exists.
    if(!(observer_index < base::n_z))
    {
        throw std::runtime_error("failed to add new observation (observer_index out of range)");
    }
    
    // Store observation in the observations map.
    // NOTE: This adds or replaces the observation at the specified observer index.
    base::m_observations[observer_index] = observation;
}
bool base::has_observations() const
{
    return !base::m_observations.empty();
}
bool base::has_observation(uint32_t observer_index) const
{
    return base::m_observations.count(observer_index) != 0;
}
void base::masked_kalman_update()
{
    // Get number of observations.
    uint32_t n_o = base::m_observations.size();

    // Using number of observations, create masked versions of S and C.
    Eigen::MatrixXd S_m(n_o, n_o);
    Eigen::MatrixXd C_m(base::n_x, n_o);
    // Iterate over z indices.
    uint32_t m_i = 0;
    uint32_t m_j = 0;
    // Iterate column first.
    for(auto j = base::m_observations.begin(); j != base::m_observations.end(); ++j)
    {
        // Iterate over rows to populate S_m.
        for(auto i = base::m_observations.begin(); i != base::m_observations.end(); ++i)
        {
            // Copy the selected S element into S_m.
            S_m(m_i++, m_j) = base::S(i->first, j->first);
        }
        m_i = 0;

        // Copy the selected C column into C_m.
        C_m.col(m_j++) = base::C.col(j->first);
    }

    // Calculate inverse of masked S.
    Eigen::MatrixXd Si_m = S_m.inverse();
    
    // Calculate Kalman gain (masked by n observations).
    Eigen::MatrixXd K_m(base::n_x,n_o);
    K_m.noalias() = C_m * Si_m;

    // Create masked version of za-z.
    Eigen::VectorXd zd_m(n_o);
    m_i = 0;
    for(auto observation = base::m_observations.begin(); observation != base::m_observations.end(); ++observation)
    {
        zd_m(m_i++) = observation->second - base::z(observation->first);
    }

    // Update state.
    base::x.noalias() += K_m * zd_m;

    // Update covariance.
    // NOTE: Just use internal temporary since it's masked size.
    base::P.noalias() -= K_m * S_m * K_m.transpose();

    // Protect against non-positive definite covariance matrices.
    // Force symmetric matrix.
    base::t_xx = base::P.transpose();
    base::P += base::t_xx;
    base::P /= 2.0;
    // Force full rank and clean out small numbers.
    for(uint32_t i = 0; i < base::n_x; ++i)
    {
        double_t row_sum = 0;
        // Force symmetry of non-diagonals.
        for(uint32_t j = 0; j < base::n_x; ++j)
        {
            if(i!=j)
            {
                if(base::P(i,j) < 1E-3)
                {
                    base::P(i,j) = 0.0;
                }
                else
                {
                    row_sum += std::abs(base::P(i,j));
                }
            }
        }
        // Force PSD by controlling diagonal.
        if(base::P(i,i) <= row_sum)
        {
            base::P(i,i) = row_sum + 1E-3;
        }
    }

    // Reset observations.
    base::m_observations.clear();
}

// ACCESS
uint32_t base::n_variables() const
{
    return base::n_x;
}
uint32_t base::n_observers() const
{
    return base::n_z;
}
double_t base::state(uint32_t index) const
{
    // Check if index is valid.
    if(index >= base::n_x)
    {
        throw std::runtime_error("invalid state variable index");
    }

    return base::x(index);
}
void base::set_state(uint32_t index, double_t value)
{
    // Check if index is valid.
    if(index >= base::n_x)
    {
        throw std::runtime_error("invalid state variable index");
    }

    base::x(index) = value;
}
double_t base::covariance(uint32_t index_a, uint32_t index_b) const
{
    // Check if indices is valid.
    if(index_a >= base::n_x || index_b >= base::n_x)
    {
        throw std::runtime_error("invalid state variable index");
    }

    return base::P(index_a, index_b);
}
void base::set_covariance(uint32_t index_a, uint32_t index_b, double_t value)
{
    // Check if indices is valid.
    if(index_a >= base::n_x || index_b >= base::n_x)
    {
        throw std::runtime_error("invalid state variable index");
    }

    base::P(index_a, index_b) = value;
}

// LOGGING
bool base::start_log(const std::string& log_file, uint8_t precision)
{
    // Stop any existing log.
    base::stop_log();

    // Open the file for writing.
    base::m_log_file.open(log_file.c_str());

    // Verify that the file opened correctly.
    if(base::m_log_file.fail())
    {
        // Close the stream and clear flags.
        base::m_log_file.close();
        base::m_log_file.clear();

        return false;
    }

    // Set precision output for the file.
    base::m_log_file << std::fixed << std::setprecision(precision);

    // Write the header line.
    for(uint32_t i = 0; i < base::n_x; ++i)
    {
        base::m_log_file << "xp_" << i << ",";
    }
    for(uint32_t i = 0; i < base::n_z; ++i)
    {
        base::m_log_file << "zp_" << i << ",";
    }
    for(uint32_t i = 0; i < base::n_z; ++i)
    {
        base::m_log_file << "za_" << i << ",";
    }
    for(uint32_t i = 0; i < base::n_x; ++i)
    {
        base::m_log_file << "xe_" << i;
        if(i + 1 < base::n_x)
        {
            base::m_log_file << ",";
        }
    }
    base::m_log_file << std::endl;

    return true;
}
void base::stop_log()
{
    // Check if a log is running.
    if(base::m_log_file.is_open())
    {
        // Close the stream and reset flags.
        base::m_log_file.close();
        base::m_log_file.clear();
    }
}
void base::log_predicted_state()
{
    if(base::m_log_file.is_open())
    {
        for(uint32_t i = 0; i < base::n_x; ++i)
        {
            base::m_log_file << base::x(i) << ",";
        }
    }
}
void base::log_observations(bool empty)
{
    if(base::m_log_file.is_open())
    {
        if(empty)
        {
            for(uint32_t i = 0; i < 2*base::n_z; ++i)
            {
                base::m_log_file << ",";
            }
        }
        else
        {
            // Predicted observations.
            for(uint32_t i = 0; i < base::n_z; ++i)
            {
                base::m_log_file << base::z(i) << ",";
            }
            // Actual observations.
            for(uint32_t i = 0; i < base::n_z; ++i)
            {
                auto observation = base::m_observations.find(i);
                if(observation != base::m_observations.end())
                {
                    base::m_log_file << observation->second;
                }
                base::m_log_file << ",";
            }
        }
    }
}
void base::log_estimated_state()
{
    if(base::m_log_file.is_open())
    {
        for(uint32_t i = 0; i < base::n_x; ++i)
        {
            base::m_log_file << base::x(i);
            if(i + 1 < base::n_x)
            {
                base::m_log_file << ",";
            }
        }
        base::m_log_file << std::endl;
    }
}