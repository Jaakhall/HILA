////////////////////////////////////////////////////////////////////////////////
/// @file expanded_ensemble_multicanonical.cpp
/// @author Jaakko Hällfors
/// @brief Model agnostic implementation of various
///        multicanonical expanded ensemble methods
/// @details TBA
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <regex>
#include <cmath>
#include "hila.h"
#include "tools/expanded_ensemble_multicanonical.h"

using string = std::string;
using int_vector = std::vector<int>;
using vector = std::vector<double>;

struct direct_iteration;

typedef bool (* finish_condition_pointer)(std::vector<int> &visits);

////////////////////////////////////////////////////////////////////////////////
/// @struct direct_iteration
/// @brief An internal struct parametrising the direct weight iteration method.
///
/// @var string direct_iteration::finish_condition
/// @brief Determines the iteration condition for the iteration method.
/// @details Current options: "all_visited", "ends_visited". The weight
/// modification factor \f$C\f$ is decreased after set bins are visited. These
/// can include e.g. all bins, or just the ends (the two preset options).
/// Finishing conditions can also be determined by the user and passed to the
/// methods through muca::direct_iteration_finish_condition(&func_pointer)
///
/// @var int direct_iteration::sample_size
/// @brief Number of samples before weight function update
///
/// @var int direct_iteration::single_check_interval
/// @brief Determines how often the update condition is checked for the case
/// direct_iteration::sample_size = 1
///
/// @var double direct_iteration::C_init
/// @brief Initial magnitude of the weight update
///
/// @var double direct_iteration::C_min
/// @brief Minimum magnitude of the weight update
///
/// @var double direct_iteration::C
/// @brief Magnitude of the weight update
/// @details This entry is decreased through the direct iteration method until
/// \f$ C < C_\mathrm{min}\f$ at which point the iteration is considered
/// complete. The magnitude of the update is such that the mean weight
/// modification is \f$C\f$. That is, the update satisfies
/// \f{equation}{\sum_i \delta W_i = N C,\f}
/// where \f$N\f$. is the number of bins. For sample_size = 1 we simply add
/// \f$ C \f$ to the single relevant bin each iteration.
///////////////////////////////////////////////////////////////////////////////
struct direct_iteration
{
    string finish_condition;
    int sample_size;
    int single_check_interval;
    double C_init;
    double C_min;
    double C;
};

////////////////////////////////////////////////////////////////////////////////
/// @struct weight_iteration_parameters
/// @brief An internal struct parametrising multicanonical methods.
///
/// @var string weight_iteration_parameters::weight_loc
/// @brief Path to the weight function file
///
/// @var string weight_iteration_parameters::outfile_name_base
/// @brief Prefix for the saved weight function files
/// @details
///
/// @var string weight_iteration_parameters::method
/// @brief Name of the iteration method
/// @details Current options: "direct"
/// See the documentation for the details of different methods.
///
/// @var bool weight_iteration_parameters::visuals
/// @brief Whether to print histogram during iteration
///
/// @var bool weight_iteration_parameters::hard_walls
/// @brief Whether the weight outside max_OP and min_OP is infinite
/// @details If not, the weight is assigned through constant extrapolation of
/// the nearest bin (first or last). Please start your simulations so that the
/// first configuration is within the interval, or the iteration can get stuck.
///
/// @var double weight_iteration_parameters::max_OP
/// @brief Maximum order parameter value
/// @details Only matters when creating bins from given parameters. When the
/// details of the weights are read from file this parameter is not used.
///
/// @var double weight_iteration_parameters::min_OP
/// @brief Minimum order parameter value
/// @details Only matters when creating bins from given parameters. When the
/// details of the weights are read from file this parameter is not used.
///
/// @var int weight_iteration_parameters::bin_number
/// @brief Number of OP bins
/// @details Only matters when creating bins from given parameters. When the
/// details of the weights are read from file this parameter is not used.
///
/// @var bool weight_iteration_parameters::AR_iteration
/// @brief Whether to update the weights after each call to accept_reject
///
/// @var struct direct_iteration weight_iteration_parameters::DIP
/// @brief A substruct that contains method specific parameters
///////////////////////////////////////////////////////////////////////////////
struct weight_iteration_parameters
{
    string weight_loc;
    string outfile_name_base;
    string method;

    bool visuals;
    bool hard_walls;
    double max_OP;
    double min_OP;
    int bin_number;
    bool AR_iteration;
    struct direct_iteration DIP;
};

// File global parameter struct filled by read_weight_parameters
static weight_iteration_parameters g_WParam;

// Initialise some static vectors for this file only.
static std::vector<vector> g_OPBinLimits;
static std::vector<vector> g_OPValues;
static std::vector<vector> g_WValues;
static vector g_ChainWValues;
//static vector g_OPValues(1,0);
//static vector g_OPBinLimits(2,0);
//static vector g_WValues(1,0);
static std::vector<int_vector> g_N_OP_Bin;
static std::vector<int_vector> g_N_OP_BinTotal;
static int g_WeightIterationCount = 0;
static bool g_WeightIterationFlag = true;

namespace hila
{
namespace muca
{

// Pointers to the iteration functions
iteration_pointer iterate_weights;
chain_iteration_pointer iterate_chains;

// Pointer to the finish condition check
finish_condition_pointer finish_check;

////////////////////////////////////////////////////////////////////////////////
/// @brief Writes a variable to the file, given the format string.
///
/// @param output_file
/// @param fmt           format string corresponding to input_value
/// @param input_value   numerical value to write to output_file
////////////////////////////////////////////////////////////////////////////////
template <class K>
void to_file(std::ofstream &output_file, string fmt, K input_value)
{
    char buffer[1024];
    sprintf(buffer, fmt.c_str(), input_value);
    if (hila::myrank() == 0) output_file << string(buffer);
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Generates a time stamped and otherwise appropriate file name for the
///        saved weight function files.
///
/// @return      generated filename string
////////////////////////////////////////////////////////////////////////////////
string generate_outfile_name()
{
    string filename = g_WParam.outfile_name_base + "_weight_function_";

    // A terrible mess to get the datetime format nice
    std::stringstream ss;
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    ss << std::put_time(&tm, "created_%Y.%m.%d_%H:%M:%S");
    string date = ss.str();

    filename = filename + date;
    return filename;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Parses the weight parameter file and fills the g_WParam struct.
/// @details
///
/// @param parameter_file_name   parameter file name
////////////////////////////////////////////////////////////////////////////////
void read_weight_parameters(string parameter_file_name)
{
    // Open the weight parameter file and list through the parameters.
    // See the parameter file for the roles of the parameters.
    hila::input par(parameter_file_name);

    // Generic control parameters
    string output_loc           = par.get("output file location");
    string outfile_name_base    = par.get("output file name base");

    string weight_loc           = par.get("weight file location");
    string iter_method          = par.get("iteration method");
    string hard_walls           = par.get("hard walls");
    double max_OP               = par.get("max OP");
    double min_OP               = par.get("min OP");
    int bin_number              = par.get("bin number");
    string iter_vis             = par.get("iteration visuals");

    // Direct iteration parameters
    string finish_condition     = par.get("finish condition");
    int DIM_sample_size         = par.get("DIM sample size");
    int DIM_check_interval      = par.get("DIM visit check interval");
    double add_initial          = par.get("add initial");
    double add_minimum          = par.get("add minimum");

    // Canonical iteration parameters
    int CIM_sample_size         = par.get("CIM sample size");
    int initial_bin_hits        = par.get("initial bin hits");
    int OC_max_iter             = par.get("OC max iter");
    int OC_frequency            = par.get("OC frequency");

    par.close();

    struct direct_iteration DIP
    {
        finish_condition,
        DIM_sample_size,
        DIM_check_interval,
        add_initial,
        add_minimum,
        add_initial
    };

    bool AR_ITER = false;
    bool visuals;
    if (iter_vis.compare("YES") == 0)
        visuals = true;
    else visuals = false;

    bool hwalls;
    if (hard_walls.compare("YES") == 0)
        hwalls = true;
    else hwalls = false;

    g_WParam =
    {
        weight_loc,
        outfile_name_base,
        iter_method,
        visuals,
        hwalls,
        max_OP,
        min_OP,
        bin_number,
        AR_ITER,
        DIP
    };
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Reads a precomputed weight function from file.
/// @details
/// The input file is to have 2N + 1 rows with following contents:
/// row 1:         weights of the used chains in their respective order as
///                whitespace separated numerical values, N entries
///
/// With the rest 2N taking the form 0 < i <= N
///
/// row 2i:        bin edges, of at least length n_i >= 2.
/// row 2i + 1:    values of the weights for each bin, length = n_i - 1 
///
/// The header can be whatever* and is always skipped. Regex finds the
/// data by finding first row with the substring "OP_value" and
/// assumes that the following contains the data as specified above.
///
/// *Avoid substring "OP_value"
///
/// @param W_function_filename
////////////////////////////////////////////////////////////////////////////////
void read_weight_function(string W_function_filename)
{
    hila::out0 << "\nLoading the user supplied weight function.\n";
    // Compute first header length by counting lines until finding
    // the column header through regex.
    int N_chains;
    if (hila::myrank() == 0)
    {
        int header_length = 1, data_length = -1;
        std::ifstream W_file;
        W_file.open(W_function_filename.c_str());
        if (W_file.is_open())
        {
            string line;
            while(std::getline(W_file, line))
            {
                if (std::regex_match(line, std::regex(".*BEGIN_DATA.*")))
                    data_length = 0;

                if (data_length < 0)
                    header_length += 1;
                else
                    data_length += 1;
            }
        }
        W_file.close();
        W_file.open(W_function_filename.c_str());


        // Skip the header and sscanf the values and add them into the vectors.
        int count = - header_length;
        data_length -= 1;
        printf("Weight function has header length of %d rows.\n", 
                header_length);
        //printf("Weight function has %d chains.\n", N_chains);
        printf("Reading the weight function into the program.\n");

        // Reset current vectors
        std::vector<std::vector<double>> empty_vec_vec(0);
        std::vector<double> empty_vec(0);
        g_OPBinLimits  = empty_vec_vec;
        g_OPValues     = empty_vec_vec;
        g_WValues      = empty_vec_vec;
        g_ChainWValues = empty_vec;

        // Assume first row constains chain weights
        // and the following pairs the bin limits and weights respectively
        if (W_file.is_open())
        {
            string line;
            while (std::getline(W_file, line))
            {
                // Read line into a vector
                vector val_vec(0);
                if (count >= 0)
                {
                    double value;
                    std::istringstream edge_stream(line);
                    while (edge_stream >> value)
                    {
                        val_vec.push_back(value);
                        //hila::out0 << count << " " << value << "\n";
                    }
                    if ((count > 0) and (count > 2 * N_chains))
                    {
                        hila::out0 << "Reading more rows than expected!" << "\n";
                    }
                }

                // Decide where to put the vector if nonempty
                if (val_vec.size() > 0)
                {
                    if (count == 0)
                    {
                        //hila::out0 << count << " " << "Reading chain weights\n";
                        g_ChainWValues = val_vec;
                        N_chains = g_ChainWValues.size();
                        //hila::out0 << "Found N_chains "<< N_chains << "\n";
                    }
                    else if (count % 2 == 1)
                    {
                        //hila::out0 << count << " " << "Reading bin limits\n";
                        g_OPBinLimits.push_back(val_vec);
                        // Determine bin centres from the edges and save them
                        // for later use
                        vector centres(val_vec.size() - 1);
                        for (int i = 0; i < val_vec.size() - 1; i++)
                        {
                            centres[i] = (val_vec[i + 1] + val_vec[i]) / 2.0;
                        }
                        g_OPValues.push_back(centres);
                    }
                    else
                    {
                        //hila::out0 << count << " " << "Reading weight values\n";
                        g_WValues.push_back(val_vec);
                    }
                }
                count += 1;
            }
        }
        W_file.close();
    }
    bool terminate = false;
    if (hila::myrank() == 0)
    {
        // Check afterwards that the lengths are consistent
        int N_BLvec = g_OPBinLimits.size();
        int N_WLvec = g_WValues.size();
        if (N_chains != N_BLvec)
        {
            hila::out0 << "There are " << N_chains << " chains, but "
                       << N_BLvec << " sets of bin limits!\n";
            hila::out0 << "Check input file formatting. Terminating.\n";
            terminate = true;
        }
        if (N_chains != N_WLvec)
        {
            hila::out0 << "There are " << N_chains << " chains, but "
                       << N_WLvec << " sets of weights!\n";
            hila::out0 << "Check input file formatting. Terminating.\n";
            terminate = true;
        }

        for (int i = 0; i < N_chains; i++)
        {
            int NBL = g_OPBinLimits[i].size();
            int NWL = g_WValues[i].size();
            if (NBL != NWL + 1)
            {
                hila::out0 << "Weights and bins for chain " << i
                           << " are inconsistent.\n";
                hila::out0 << "There are " << NBL << " bin edges, and " << NWL
                           << " weights!\n";
                hila::out0 << "Check input file formatting. Terminating.\n";
                terminate = true;
            }
        }

        // Check that the bin edges form an increasing sequence
        for (int i = 0; i < N_chains; i++)
        {
            double prev = g_OPBinLimits[i][0];
            for (int j = 1; j < g_OPBinLimits[i].size(); j++)
            {
                if (g_OPBinLimits[i][j] <= prev)
                {
                    hila::out0 << "The bin limits for chain " << i
                               << " do not form an increasing sequence!\n"
                               << "Edge " << j << " is less than or equal to "
                               << "edge " << j - 1 << "\n"
                               << "Check provided bin edges.\n";
                    terminate = true;
                }
                prev = g_OPBinLimits[i][j];
            }
        }
    }
    hila::broadcast(terminate);
    if (terminate)
    {
        hila::out0 << "Check input file formatting. Terminating.\n";
        hila::finishrun();
    }

    // Prints the bin limits and weights.
    //for (int i = 0; i < N_chains; i++)
    //{
    //    for (int j = 0; j < g_OPBinLimits[i].size(); j++)
    //    {
    //        printf("%e\n", g_OPBinLimits[i][j]);
    //    }
    //    printf("\n");
    //    for (int j = 0; j < g_WValues[i].size(); j++)
    //    {
    //        printf("%e\n", g_WValues[i][j]);
    //    }
    //    printf("Chain weight = %e\n", g_ChainWValues[i]);
    //    printf("\n");
    //}

    hila::out0 << "\nSuccesfully loaded the user provided weight function.\n";
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Reads the precomputed weight function from run_parameters struct
///        and saves it into a file.
/// @details The printing happens in an format identical to what is expected
///          by the funciton read_weight_function. See its documentation for
///          details.
///          TBA: Add string input that can contain user specified header data.
///
/// @param W_function_filename
/// @param g_WParam                    struct of weight iteration parameters
////////////////////////////////////////////////////////////////////////////////
void write_weight_function(string W_function_filename, std::string header)
{
    if (hila::myrank() == 0)
    {
        std::ofstream W_file;
        //string filename = generate_outfile_name(RP);
        printf("Writing the current weight function into a file...\n");
        W_file.open(W_function_filename.c_str());

        // Append a user defined header
        W_file << header;

        // Start with the chain weights
        to_file(W_file, "BEGIN_DATA\n", 0);
        for (int j = 0; j < g_ChainWValues.size(); j++)
                to_file(W_file, "%e\t", g_ChainWValues[j]);
        to_file(W_file, "\n", 0);
        // Then bin limits and weight values chain by chain
        for (int i = 0; i < g_ChainWValues.size(); ++i)
        {
            for (int j = 0; j < g_OPBinLimits[i].size(); j++)
                to_file(W_file, "%e\t", g_OPBinLimits[i][j]);
            to_file(W_file, "\n", 0);

            
            for (int j = 0; j < g_WValues[i].size(); j++)
                to_file(W_file, "%e\t", g_WValues[i][j]);
            to_file(W_file, "\n", 0);
        }
        // Remember to write the last bin upper limit
        W_file.close();
        printf("Succesfully saved the weight function into file\n%s\n",
               W_function_filename.c_str());
    }
}

//////////////////////////////////////////////////////////////////////////////////
///// @brief Returns a weight associated to the used order parameter.
///// @details The function uses supplied pairs of points to linearly interpolate
/////          the function on the interval. This interpolant provides the
/////          requested weights to be used as the multicanonical weight.
/////
///// @param  OP   value of the order parameter
///// @return The value of the weight.
//////////////////////////////////////////////////////////////////////////////////
//double weight_function(const double OP, const int ci)
//{
//    double val;
//    // If out of range, constant extrapolation or for hard walls, num inf.
//    if ((g_WParam.hard_walls) and
//       ((OP < g_OPValues[ci].front()) or (OP > g_OPValues[ci].back())))
//    {
//        val = std::numeric_limits<double>::infinity();
//    }
//    else if (OP <= g_OPValues[ci].front())
//    {
//        val = g_WValues[ci].front();
//    }
//    else if (OP >= g_OPValues[ci].back())
//    {
//        val = g_WValues[ci].back();
//    }
//
//    // Otherwise find interval, calculate slope, base index, etc.
//    // Basic linear interpolation to obtain the weight value.
//    else
//    {
//        auto it = std::lower_bound(g_OPValues[ci].begin(),
//                                   g_OPValues[ci].end(), OP);
//        int j = std::distance(g_OPValues[ci].begin(), it) - 1;
//        double y_value = g_WValues[ci][j + 1] - g_WValues[ci][j];
//        double x_value = g_OPValues[ci][j + 1] - g_OPValues[ci][j];
//        double slope = y_value / x_value;
//
//        double xbase = g_OPValues[ci][j];
//        double ybase = g_WValues[ci][j];
//
//        double xdiff = OP - xbase;
//        val = ybase + xdiff * slope;
//    }
//
//    return val + g_ChainWValues[ci];
//}

////////////////////////////////////////////////////////////////////////////////
/// @brief Returns a weight associated to the used order parameter.
/// @details The function uses supplied pairs of points to linearly interpolate
///          the function on the interval. This interpolant provides the
///          requested weights to be used as the multicanonical weight.
///
/// @param  OP   value of the order parameter
/// @return The value of the weight.
////////////////////////////////////////////////////////////////////////////////
static double weight_function(const double OP, const int ci)
{
    double val, slope, xdiff, ybase;
    if (OP <= g_OPValues[ci].front())
    {
        slope = -10000;
        xdiff = OP - g_OPValues[ci].front();
        ybase = g_WValues[ci].front();
        val = ybase + xdiff * slope;
        //val = std::numeric_limits<double>::infinity();
    }
    else if (OP >= g_OPValues[ci].back())
    {
        slope = 10000;
        xdiff = OP - g_OPValues[ci].back();
        ybase = g_WValues[ci].back();
        val = ybase + xdiff * slope;
        //val = std::numeric_limits<double>::infinity();
    }
    // Otherwise find interval, calculate slope, base index, etc.
    // Basic linear interpolation to obtain the weight value.
    else
    {
        auto it = std::lower_bound(g_OPValues[ci].begin(),
                                   g_OPValues[ci].end(), OP);
        int j = std::distance(g_OPValues[ci].begin(), it) - 1;
        double y_value = g_WValues[ci][j + 1] - g_WValues[ci][j];
        double x_value = g_OPValues[ci][j + 1] - g_OPValues[ci][j];
        double slope = y_value / x_value;

        double xbase = g_OPValues[ci][j];
        double ybase = g_WValues[ci][j];

        double xdiff = OP - xbase;
        val = ybase + xdiff * slope;
    }
    return val + g_ChainWValues[ci];
}

////////////////////////////////////////////////////////////////////////////////
/// @brief process 0 interface to "weight function" for the user accessing
///        the weights.
////////////////////////////////////////////////////////////////////////////////
double weight(double OP, int chain_index)
{
    double val;
    if (hila::myrank() == 0)
    {
        val = weight_function(OP, chain_index);
    }
    hila::broadcast(val);
    return val;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Sets the static g_WeightIterationFlag to given boolean.
///
/// @param YN   boolean indicating whether the iteration is to continue
////////////////////////////////////////////////////////////////////////////////
void set_weight_iter_flag(bool YN)
{
    if (hila::myrank() == 0) g_WeightIterationFlag = YN;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Returns the value of the static g_WeightIterationFlag to user
///
/// @return State of g_WeighITerationFlag
////////////////////////////////////////////////////////////////////////////////
bool check_weight_iter_flag()
{
    bool flag;
    if (hila::myrank() == 0) flag = g_WeightIterationFlag;
    hila::broadcast(flag);
    return flag;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Accepts/rejects a multicanonical update.
/// @details Using the values of the old and new order parameters the muca
///          update is accepted with the logarithmic probability
///          log(P) = - (W(OP_new) - W(OP_old))
///
/// @param  OP_old        current order parameter
/// @param  OP_new        order parameter of proposed configuration
/// @param  chain_index   index of the current chain
/// @param  chain_index   index of the proposed chain
/// @return Boolean indicating whether the update was accepted (true) or
///         rejected (false).
////////////////////////////////////////////////////////////////////////////////
bool accept_reject(const double OP_old,
                   const double OP_new,
                   const int chain_index_old,
                   const int chain_index_new)
{
    bool update;
    bool AR_iterate;
    // Only compute on node 0, broadcast to others
    if (hila::myrank() == 0)
    {
        double W_new = weight_function(OP_new, chain_index_new);
        double W_old = weight_function(OP_old, chain_index_old);

        // This happens when hard walls are active and OP_new
        // is out of bounds
        if (W_new > std::numeric_limits<double>::max())
        {
            update = false;
        }
        else
        {
            // get log(exp(-delta(W))) = -delta(W)
            // (just like -delta(S) in Metropolis-Hastings)
            double log_P = - (W_new - W_old);

            // Get a random uniform from [0,1] and return a boolean indicating
            // whether the update is to be accepted.
            double rval = hila::random();
            if (::log(rval) < log_P)
            {
                update = true;
            }
            else
            {
                update = false;
            }
        }

        // Get value from process 0
        AR_iterate = g_WParam.AR_iteration;
    }

    // Broadcast the update status to other processes along with the
    // weight iteration parameter
    hila::broadcast(update);
    hila::broadcast(AR_iterate);

    // Check if iteration is enabled
    if (AR_iterate)
    {
        if (update) set_weight_iter_flag(iterate_weights(OP_new, chain_index_old));
        else set_weight_iter_flag(iterate_weights(OP_old, chain_index_old));
    }

    return update;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Finds the index of the correc order parameter bin.
/// @details Using the bin limit vector the correct order parameter bin is
///          determined through a simple standard library search of
///          g_OPBinLimits. When the value of the given order parameter is
///          out of range, -1 is returned.
///
/// @param OP   value of the order parameter
/// @param ci   index of the current chain
/// @return integer index for the vector g_N_OP_Bin
////////////////////////////////////////////////////////////////////////////////
static int find_OP_bin_index(const double OP, const int ci)
{
    // Return -1 when not in the interval
    if (OP <= g_OPBinLimits[ci].front())
    {
        return -1;
    }
    else if (OP >= g_OPBinLimits[ci].back())
    {
        return -1;
    }
    // Find index of minimum edge value such that edge < OP:
    auto it = std::lower_bound(g_OPBinLimits[ci].begin(),
                               g_OPBinLimits[ci].end(), OP);
    int lower_limit = std::distance(g_OPBinLimits[ci].begin(), it) - 1;
    return lower_limit;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Same as find_OP_bin_index, except uses the index to simply modify the
///        bin hit vector g_N_OP_Bin. Does not modify when outside of the range.
/// @details
///
/// @param OP   value of the order parameter
/// @param ci   index of the current chain
////////////////////////////////////////////////////////////////////////////////
static void bin_OP_value(const double OP, const int ci)
{
    // Don't bin visits outside of the binned areas
    if (OP <= g_OPBinLimits[ci].front())
    {
        return;
    }
    else if (OP >= g_OPBinLimits[ci].back())
    {
        return;
    }
    // Find index of minimum edge value such that edge < OP:
    auto it = std::lower_bound(g_OPBinLimits[ci].begin(),
                               g_OPBinLimits[ci].end(), OP);
    int lower_limit = std::distance(g_OPBinLimits[ci].begin(), it) - 1;
    g_N_OP_Bin[ci][lower_limit] += 1;
}


////////////////////////////////////////////////////////////////////////////////
/// @brief Checks if all the bins have been visited by.
/// @details Simply checks whether all bins have a nonzero number of entries
///
/// @param  visit   integer vector with values 1 corresponding to visits
/// @return a boolean indicating the statement
////////////////////////////////////////////////////////////////////////////////
static bool all_visited(int_vector &n)
{
    int len = n.size();
    for (int i = 0; i < len; ++i)
    {
        if (n[i] == 0) return false;
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Checks if the first and last bin have been visited
/// @details Simply checks whether all bins have a nonzero number of entries.
///
/// @param  visit   integer vector with values 1 corresponding to visits
/// @return a boolean indicating the statement
////////////////////////////////////////////////////////////////////////////////
static bool first_last_visited(int_vector &n)
{
    int len = n.size();
    if ((n[0] == 0) or (n[len - 1] == 0))
        return false;
    else
        return true;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Sets a user provided function to the check in the "direct iteration"
/// method.
/// @details The for a given magnitude of update the "direct iteration" method
/// periodically checks whether the MCMC chain has covered enough of the
/// desired order parameter range, before reducing the update magnitude. Some
/// preset methods exist (and should suffice) but when needed, a new condition
/// can be set through this function. The input is a vector of integers
/// indicating the number of visits to each bin, and the output is a boolean
/// telling whether the desired condition has been achieved.
///
/// @param fc_pointer   A function pointer to a suitable condition function
////////////////////////////////////////////////////////////////////////////////
void set_direct_iteration_FC(bool (* fc_pointer)(int_vector &n))
{
    finish_check = fc_pointer;
}

//////////////////////////////////////////////////////////////////////////////////
///// @brief Given an order parameter, iterates the weight function until the
/////        sampling becomes acceptably efficient. Save the weight function into
/////        a file for later use (NOT FUNCTIONAL).
///// @details
/////
///// @param F     struct of fields
///// @param FP    struct of field parameters
///// @param RP    struct of run parameters
///// @param g_WParam   struct of weight iteration parameters
//////////////////////////////////////////////////////////////////////////////////
//void iterate_weight_function_canonical(fields &F,
//                                       field_parameters &FP,
//                                       run_parameters &RP,
//                                       weight_iteration_parameters &g_WParam)
//{
//    int samples  = g_WParam.sample_size;
//    int n_sweeps = g_WParam.sample_steps;
//
//    double max = g_WParam.max_OP;
//    double min = g_WParam.min_OP;
//
//    int N = g_WParam.bin_number;
//    // Initialise the weight vector with the bin centres
//    // and get corresponding bin limits.
//    {
//        double dx = (max - min) / (N - 1);
//        for (int i = 0; i < N; ++i)
//        {
//            RP.OP_values[i] = min + i * dx;
//        }
//    }
//    vector limits = get_bin_limits(min, max, N);
//
//    // Initialise the storage vectors to zero:
//    int_vector n(N, 0), g_sum(N, 0), n_sum(N, 0);
//    vector g_log_h_sum(N, 0), log_h(N, 0), W_prev(N, 0);
//
//    // Get initial guesses
//    for (int i = 0; i < N; ++i)
//    {
//        n[i]     = 0;
//        g_sum[i] = g_WParam.initial_bin_hits;
//        log_h[i] = RP.W_values[i];
//    }
//
//    static int count = 1;
//    while (true)
//    {
//        float accept = 0;
//        float OP_sum = 0;
//        for (int i = 0; i < samples; i++)
//        {
//            accept += mc_update_sweeps(F, FP, RP, n_sweeps);
//            OP_sum += FP.OP_value;
//            bin_OP_values(n, limits, FP.OP_value);
//        }
//
//        for (int m = 0; m < N; m++)
//        {
//            n_sum[m] += n[m];
//        }
//
//        if (count % g_WParam.OC_frequency == 0 and count < g_WParam.OC_max_iter)
//        {
//            overcorrect(RP.W_values, n_sum);
//        }
//        else
//        {
//            recursive_weight_iteration(RP.W_values, n, g_sum, g_log_h_sum);
//        }
//
//
//        // Some mildly useful print out
//        int nmax = *std::max_element(n_sum.begin(), n_sum.end());
//        for (int m = 0; m < N; ++m)
//        {
//            std::string n_sum_log = "";
//            for (int i = 0; i < int(n_sum[m] * 50.0 / nmax); i++)
//            {
//                n_sum_log += "|";
//            }
//            if (hila::myrank() == 0)
//            {
//                printf("%5.3f\t%10.3f\t\t%d\t%s\n", limits[m],
//                        RP.W_values[m],
//                        n_sum[m], n_sum_log.c_str());
//            }
//            n[m] = 0;
//        }
//
//        count += 1;
//        if (count > g_WParam.max_iter) break;
//    }
//}

//////////////////////////////////////////////////////////////////////////////////
///// @brief Procures a vector containing equidistant bin edges.
///// @details
/////
///// @return vector containing the bin edges
//////////////////////////////////////////////////////////////////////////////////
//static vector get_equidistant_bin_limits()
//{
//    double min = g_WParam.min_OP;
//    double max = g_WParam.max_OP;
//    int N      = g_WParam.bin_number;
//    vector bin_edges(N + 1);
//    double diff = (max - min) / (N - 1);
//    for (int i = 0; i < N + 1; ++i)
//    {
//        bin_edges[i] = min - diff / 2.0 + diff * i;
//    }
//    return bin_edges;
//}
//
//////////////////////////////////////////////////////////////////////////////////
///// @brief Sets up the global vectors for bin limits and centres using
/////        get_equidistant_bin_limits.
///// @details
//////////////////////////////////////////////////////////////////////////////////
//static void setup_equidistant_bins()
//{
//    // Get bin limits so that centre of first bin is at min_OP and
//    // the last bin centre is at max_OP.
//    g_OPBinLimits = get_equidistant_bin_limits();
//    for (int i = 0; i < g_OPValues.size(); i++)
//    {
//        double centre = (g_OPBinLimits[i + 1] + g_OPBinLimits[i]) / 2.0;
//        g_OPValues[i] = centre;
//    }
//}

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialises the global vectors appropriately, setting up a binning
///        if not provided by the user.
/// @details The global vectors are initialised to correct dimensions as to
///          prevent indexing errors in the iteration methods.
////////////////////////////////////////////////////////////////////////////////
static void initialise_weight_vectors()
{
    std::vector<int_vector> GNOB;
    for (int i = 0; i < g_ChainWValues.size(); i++)
    {
        int N = g_WValues[i].size();
        GNOB.push_back(int_vector(N, 0));
    }
    GNOB.push_back(int_vector(g_ChainWValues.size(), 0));

    g_N_OP_Bin      = GNOB;
    g_N_OP_BinTotal = GNOB;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Given an order parameter, bins it to correct weight interval, and
///        periodically updates the weights accordingly.
/// @details This extremely simple update method
///
/// @param  OP   order parameter of the current configuration (user supplied)
/// @return boolean indicating whether the iteration is considered complete
////////////////////////////////////////////////////////////////////////////////
static bool iterate_weight_function_direct(const double OP, const int ci)
{
    bool continue_iteration;
    if (hila::myrank() == 0)
    {
        int samples = g_WParam.DIP.sample_size;
        int N       = g_WValues.size();

        bin_OP_value(OP, ci);
        g_WeightIterationCount += 1;

        if (g_WeightIterationCount >= samples)
        {
            for (int m = 0; m < N; m++)
            {
                g_WValues[ci][m] += g_WParam.DIP.C * g_N_OP_Bin[ci][m] * N / samples;
                g_N_OP_BinTotal[ci][m] += g_N_OP_Bin[ci][m];
            }

            if (g_WParam.visuals) print_iteration_histogram(ci);

            double base = *std::min_element(g_WValues[ci].begin(), g_WValues[ci].end());
            for (int m = 0; m < N; ++m)
            {
                // Always set minimum weight to zero. This is inconsequential
                // as only the differences matter.
                g_WValues[ci][m] -= base;
                g_N_OP_Bin[ci][m] = 0;
            }
            g_WeightIterationCount = 0;

            if (finish_check(g_N_OP_BinTotal[ci]))
            {
                for (int m = 0; m < N; m++)
                {
                    g_N_OP_BinTotal[ci][m] = 0;
                }

                g_WParam.DIP.C /= 1.5;
                hila::out0 << "Decreasing update size...\n";
                hila::out0 << "New update size C = " << g_WParam.DIP.C << "\n\n";
            }
            write_weight_function("intermediate_weight.dat");
            hila::out0 << "Update size C = " << g_WParam.DIP.C << "\n\n";
        }

        continue_iteration = true;
        if (g_WParam.DIP.C < g_WParam.DIP.C_min)
        {
            hila::out0 << "Reached minimum update size.\n";
            hila::out0 << "Weight iteration complete.\n";
            continue_iteration = false;
        }
    }
    hila::broadcast(continue_iteration);
    return continue_iteration;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Same as iterate_weight_function_direct for sample size 1.
/// @details In this special case the weights are modified after each new
///          value of the order parameter. Reduced internal complexity due to
///          this simplification.
///          Whether all bins have been visited is now checked only every
///          g_WParam.DIP.single_check_interval to prevent excessive checking
///          for the visits.
///
/// @param  OP   order parameter of the current configuration
///              (user supplied)
/// @param  ci   current chain index (user supplied)
/// @return boolean indicating whether the iteration is considered complete
////////////////////////////////////////////////////////////////////////////////
static bool iterate_weight_function_direct_single(const double OP,
                                                  const int ci)
{
    int continue_iteration;
    if (hila::myrank() == 0)
    {
        int samples = g_WParam.DIP.sample_size;
        int N       = g_WValues[ci].size();

        int bin_index = find_OP_bin_index(OP, ci);
        //hila::out0 << OP << "  " << bin_index  << " chain " << ci << "\n";
        // Only increment if on the min-max interval
        if (bin_index != -1)
            g_N_OP_BinTotal[ci][bin_index] += 1;

        g_WeightIterationCount += 1;
        g_WValues[ci][bin_index] += g_WParam.DIP.C;

        continue_iteration = true;

        if (g_WeightIterationCount % g_WParam.DIP.single_check_interval == 0)
        {

        double base = *std::min_element(g_WValues[ci].begin(), g_WValues[ci].end());
        for (int m = 0; m < N; ++m)
        {
            // Always set minimum weight to zero. This is inconsequential
            // as only the differences matter.
            g_WValues[ci][m] -= base;
            g_N_OP_Bin[ci][m] = 0;
        }

        // Visuals
        if (g_WParam.visuals) print_iteration_histogram(ci);

        // If condition satisfied, zero the totals and decrease C
        if (finish_check(g_N_OP_BinTotal[ci]))
        {
            for (int m = 0; m < N; m++)
            {
                g_N_OP_BinTotal[ci][m] = 0;
            }

            g_WParam.DIP.C /= 1.5;
            hila::out0 << "Decreasing update size...\n";
            hila::out0 << "New update size C = " << g_WParam.DIP.C << "\n\n";
        }

        if (g_WParam.DIP.C < g_WParam.DIP.C_min)
        {
            hila::out0 << "Reached minimum update size.\n";
            hila::out0 << "Weight iteration complete.\n";
            continue_iteration = false;
        }

        //write_weight_function("intermediate_weight.dat");
        hila::out0 << "Update size C = " << g_WParam.DIP.C << "\n\n";
        }
    }
    hila::broadcast(continue_iteration);
    return continue_iteration;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Same as iterate_weight_function_direct for sample size 1.
/// @details In this special case the weights are modified after each new
///          value of the order parameter. Reduced internal complexity due to
///          this simplification.
///          Whether all bins have been visited is now checked only every
///          g_WParam.DIP.single_check_interval to prevent excessive checking
///          for the visits.
///
/// @param  OP   order parameter of the current configuration
///              (user supplied)
/// @param  ci   current chain index (user supplied)
/// @return boolean indicating whether the iteration is considered complete
////////////////////////////////////////////////////////////////////////////////
static bool iterate_chains_direct_single(const double OP, const int ci)
{
    int continue_iteration;
    if (hila::myrank() == 0)
    {
        int N = g_ChainWValues.size();
        g_WeightIterationCount += 1;
        g_ChainWValues[ci] += g_WParam.DIP.C;

        g_N_OP_BinTotal[N][ci] += 1;
        continue_iteration = true;

        if (g_WeightIterationCount % g_WParam.DIP.single_check_interval == 0)
        {
            double base = *std::min_element(g_ChainWValues.begin(),
                                            g_ChainWValues.end());
            for (int m = 0; m < N; ++m)
            {
                // Always set minimum weight to zero. This is inconsequential
                // as only the differences matter.
                g_ChainWValues[m] -= base;
                g_N_OP_Bin[N][m] = 0;
            }
            
            for (int m = 0; m < N; ++m)
            {
                printf("%f\t", g_ChainWValues[m]);
            }
            printf("\n");

            for (int m = 0; m < N; ++m)
            {
                printf("%d\t", g_N_OP_BinTotal[N][m]);
            }
            printf("\n");

            // If condition satisfied, zero the totals and decrease C
            if (finish_check(g_N_OP_BinTotal[N]))
            {
                for (int m = 0; m < N; m++)
                {
                    g_N_OP_BinTotal[N][m] = 0;
                }

                g_WParam.DIP.C /= 1.5;
                hila::out0 << "Decreasing update size...\n";
                hila::out0 << "New update size C = " << g_WParam.DIP.C << "\n\n";
            }

            if (g_WParam.DIP.C < g_WParam.DIP.C_min)
            {
                hila::out0 << "Reached minimum update size.\n";
                hila::out0 << "Weight iteration complete.\n";
                continue_iteration = false;
            }

            write_weight_function("intermediate_weight.dat");
            hila::out0 << "Update size C = " << g_WParam.DIP.C << "\n\n";
        }
    }
    hila::broadcast(continue_iteration);
    return continue_iteration;
}
////////////////////////////////////////////////////////////////////////////////
/// @brief Prints out a crude horisontal histogram.
/// @details Procures a crude horisontal ASCII histogram based on the g_N_OP_Bin
///          vector. The histogram bin heights are proportional to g_N_OP_Bin
///          values. This is not very expressive for large N, as it won't fit
///          the height of the screen.
////////////////////////////////////////////////////////////////////////////////
static void print_iteration_histogram(const int ci)
{
    int samples = g_WParam.DIP.sample_size;
    int N       = g_WValues[ci].size();
    // Find maximum bin content for normalisation
    int nmax    = *std::max_element(g_N_OP_Bin[ci].begin(), g_N_OP_Bin[ci].end());
    // Write a column header
    hila::out0 << "Chain " << ci << "\n"; 
    printf("Order Parameter     Weight 	         Number of hits\n");

    for (int m = 0; m < N; ++m)
    {
        // For each bin get a number of "|":s proportional to the number of
        // hits to each bin and print it out along with relevant numerical
        // values
        std::string n_sum_hist = "";
        if (g_N_OP_BinTotal[ci][m] > 0) n_sum_hist += "O";
        for (int i = 0; i < int(g_N_OP_Bin[ci][m] * 200.0 / samples); i++)
        {
            n_sum_hist += "|";
        }
        printf("%-20.7f%-20.7f%d\t\t\t%s\n", g_OPValues[ci][m],
                g_WValues[ci][m], g_N_OP_Bin[ci][m], n_sum_hist.c_str());
    }
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Initialises all the variables needed for the weight iteration.
/// @details For the iteration the function pointer to iterate_weights must be
///          initialised. The condition for proceeding to the next step of the
///          iteration is also set through a function pointer.
///
///          The correct methods are chosen according to the
///          parameter file. Further, method specific variables that are
///          modified during the run are also set to the necessary values.
////////////////////////////////////////////////////////////////////////////////
static void setup_iteration()
{
    // Initialise iterate_weights by pointing it at the
    // correct method
    if (g_WParam.method.compare("direct") == 0)
    {
        if (g_WParam.DIP.sample_size > 1)
        {
            iterate_weights = &iterate_weight_function_direct;
            iterate_chains = &iterate_chains_direct_single;
        }
        else
        {
            iterate_weights = &iterate_weight_function_direct_single;
            iterate_chains = &iterate_chains_direct_single;
        }
        g_WParam.DIP.C = g_WParam.DIP.C_init;
    }
    else
    {
        iterate_weights = &iterate_weight_function_direct;
        iterate_chains = &iterate_chains_direct_single;
        g_WParam.DIP.C = g_WParam.DIP.C_init;
    }

    // Zero the iteration counter
    g_WeightIterationCount = 0;

    // Set up the finish condition pointer for the direct method.
    if (g_WParam.DIP.finish_condition.compare("all_visited") == 0)
    {
        finish_check = &all_visited;
    }
    else if (g_WParam.DIP.finish_condition.compare("ends_visited") == 0)
    {
        finish_check = &first_last_visited;
    }
    else
    {
        finish_check = &all_visited;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Enable/disable continuous weight iteration
/// @details Premits the user to enable/disable continuous weight iteration at
///          each call to accept_reject. Simply modifies a flag parameter
///          that is checked in accept_reject.
///
/// @param YN   enable (true) or disable (false) the iteration
////////////////////////////////////////////////////////////////////////////////
void set_continuous_iteration(bool YN)
{
    if (hila::myrank() == 0) g_WParam.AR_iteration = YN;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Enable/disable hard OP walls
/// @details Premits the user to enable/disable hard OP walls when evaluating
///          a weight value outside of the interval [min_OP, max_OP].
///
/// @param YN   enable (true) or disable (false) hard limits on OP
////////////////////////////////////////////////////////////////////////////////
void hard_walls(bool YN)
{
    if (hila::myrank() == 0) g_WParam.hard_walls = YN;
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Loads parameters and weights for the multicanonical computation.
/// @details Sets up iteration variables. Can be called multiple times and must
///          be called at least once before attempting to use any of the muca
///          methods.
///
/// @param wfile_name   path to the weight parameter file
////////////////////////////////////////////////////////////////////////////////
void initialise(const string wfile_name)
{
    // Read parameters into g_WParam struct
    read_weight_parameters(wfile_name);
    // This is fine to do just for process zer0
    if (hila::myrank() == 0)
    {
        // Read pre-existing weight if given
        if (g_WParam.weight_loc.compare("NONE") != 0)
        {
            read_weight_function(g_WParam.weight_loc);
	}

        // Initialise rest of the uninitialised vectors
        initialise_weight_vectors();
    }

    // Choose an iteration method (or the default)
    setup_iteration();
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Returns the value of g_WParam.OP_min
///
/// @return  value of OP_min
////////////////////////////////////////////////////////////////////////////////
void muca_min_OP(double &value, bool modify)
{
    if (modify)
    {
        if (hila::myrank() == 0) g_WParam.min_OP = value;
        hila::out0 << "min_OP set to new value " << g_WParam.min_OP << "\n";
    }
    else
    {
	value = g_WParam.min_OP;
	hila::broadcast(value);
    }
}

////////////////////////////////////////////////////////////////////////////////
/// @brief Returns the value of g_WParam.max_OP
///
/// @return  value of max_OP
////////////////////////////////////////////////////////////////////////////////
void muca_max_OP(double &value, bool modify)
{
    if (modify)
    {
        if (hila::myrank() == 0) g_WParam.max_OP = value;
        hila::out0 << "max_OP set to new value " << g_WParam.max_OP << "\n";
    }
    else
    {
        value = g_WParam.max_OP;
	hila::broadcast(value);
    }
}

void weight_iter_add(double &C, bool modify)
{
    if (modify)
        g_WParam.DIP.C = C;
    else
    {
        C = g_WParam.DIP.C;
	hila::broadcast(C);
    }
}

void set_weight_bin_edges(std::vector<std::vector<double>> edges)
{
    g_OPBinLimits = edges;
    std::vector<vector> new_centres;
    for (int i = 0; i < g_OPBinLimits.size(); i++)
    {
        vector nc;
        for (int j = 0; j < g_OPBinLimits[i].size() - 1; j++)
        {
            double cent = g_OPBinLimits[i][j + 1];
            cent += g_OPBinLimits[i][j];
            cent /= 2.0;

            nc.push_back(cent);
        }
        new_centres.push_back(nc);
    }
    g_OPValues = new_centres;
    //hila::out0 << g_OPValues.size() << "\n";
    //for (int i = 0; i < g_OPValues.size(); i++)
    //    hila::out0 << g_OPValues[i][2] << " ";
}

void set_weights(std::vector<std::vector<double>> weights)
{
    g_WValues = weights;
    initialise_weight_vectors();
}

void set_chain_weights(std::vector<double> chain_weights)
{
    g_ChainWValues = chain_weights;
    initialise_weight_vectors();
}

void add_to_chain(int chain_index, double C)
{
    g_ChainWValues[chain_index] += C;
}

}
}
