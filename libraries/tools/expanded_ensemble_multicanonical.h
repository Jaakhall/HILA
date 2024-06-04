////////////////////////////////////////////////////////////////////////////////
/// @file expanded_ensemble_multicanonical.h
/// @author Jaakko Hällfors
/// @brief Header for model agnostic implementation of
///        multicanonical expanded ensemble methods
////////////////////////////////////////////////////////////////////////////////
#ifndef WE_MULTICANONICAL_HEADER
#define WE_MULTICANONICAL_HEADER

// !OBS!
// Please see the doxygen documentation from github for further details. The
// corresponding .cpp file contains the actual implementations and also
// documents the internal functions (not included in doxygen) in relative
// detail.
// Furthermore, the repository will contain a short example program that
// elucidates the use of the various muca methods.

namespace hila
{
// Multicanonical methods are separated to their own namespace
namespace muca
{
// Function pointer to the iteration function
typedef bool (* iteration_pointer)(const double OP, const int chain_index);
typedef bool (* chain_iteration_pointer)(const double OP, const int chain_index);
extern iteration_pointer iterate_weights;
extern chain_iteration_pointer iterate_chains;

// Quick helper function for writing values to a file
template <class K>
void to_file(std::ofstream &output_file, std::string fmt, K input_value);

// Generates timestamped file names
std::string generate_outfile_name();

// Reads parameters for muca computations
void read_weight_parameters(std::string parameter_file_name);

// Reads weight functions from a file
void read_weight_function(std::string W_function_filename);

// Writes weight functions to a file
void write_weight_function(std::string filename);

// Gives the weight as a function of the order parameter
double weight(const double OP, const int chain_index);

// Accept/reject determination for pairs of order parameter values
bool accept_reject(const double OP_old, const double OP_new,
                   const int chain_index_old, const int chain_index_new);

// Set the direct iteration finish condition
void set_direct_iteration_FC(bool (* fc_pointer)(std::vector<int> &n));

// Set to perform the weight iteration at each call to accept_reject
void set_continuous_iteration(bool YN);

// For the continuous iteration the finish condition is tracked internally
// and can be checked and set using the two functions below
bool check_weight_iter_flag();
void set_weight_iter_flag(bool YN);

void set_weight_iter_add(double C);
void add_to_chain(int chain_index, double C);

void set_weight_bin_edges(std::vector<std::vector<double>> edges);
void set_weights(std::vector<std::vector<double>> weights);
void set_chain_weights(std::vector<double> chain_weights);

// Initialises the muca computations according to the weight parameter file.
// This function is to always be called before using any of the above functions
void initialise(const std::string wfile_name);

// Access some of the internal variables from the program
void muca_min_OP(double &value, bool modify = false);
void muca_max_OP(double &value, bool modify = false);

// enable/disable hard walls for the order parameter
void hard_walls(bool YN);
////////////////////////////////////////////////////////////////////////////////
// Static functions are internal to above methods. See .cpp file for details
////////////////////////////////////////////////////////////////////////////////
static double weight_function(const double OP, const int chain_index);

static void bin_OP_value(const double OP, const int chain_index);

static int find_OP_bin_index(const double OP, const int chain_index);

static bool all_visited(std::vector<int> &n);
static bool first_last_visited(std::vector<int> &n);

static void overcorrect(std::vector<double> &Weight, std::vector<int> &n_sum);

static void recursive_weight_iteration(std::vector<double> &Weight,
                                       std::vector<int> &n,
                                       std::vector<int> &g_sum,
                                       std::vector<double> &g_log_h_sum);

static void print_iteration_histogram(const int chain_index);

static std::vector<double> get_equidistant_bin_limits();

static void setup_equidistant_bins();

static void initialise_weight_vectors();

static bool iterate_weight_function_direct(const double OP, const int chain_index);

static bool iterate_weight_function_direct_single(const double OP, const int chain_index);

static bool iterate_chains_direct_single(const double OP, const int chain_index);

static void setup_iteration();

}
}

#endif
