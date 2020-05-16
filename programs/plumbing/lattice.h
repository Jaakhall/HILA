#ifndef LATTICE_H
#define LATTICE_H

#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#define SUBNODE_LAYOUT  // TEMP HERE TO HELP EDITOR!!!!

// TODO: assertion moved somewhere where basic params
#undef NDEBUG
#include <assert.h>
#include "../plumbing/defs.h"
#include "../plumbing/coordinates.h"
#include "../plumbing/inputs.h"

#ifdef SUBNODE_LAYOUT
#ifndef VECTOR_SIZE
#define VECTOR_SIZE (256/8)                    // this is for AVX2
#endif
// This is the vector size used to determine the layout
constexpr unsigned number_of_subnodes = VECTOR_SIZE/sizeof(float);
#endif

// list boundary conditions - used only if SPECIAL_BOUNDARY_CONDITIONS defined
enum class boundary_condition_t {PERIODIC, ANTIPERIODIC, FIXED};

void test_std_gathers();

struct node_info {
  coordinate_vector min,size;
  unsigned evensites, oddsites;
};

/* Some backends need specialized lattice data
 * in loops. Forward declaration here and
 * implementations in backend headers.
 * Loops generated by Transformer can access
 * this through lattice->backend_lattice.
 */
struct backend_lattice_struct;


class lattice_struct {
private:
 
  // Use ints instead of unsigned, just to avoid surprises in arithmetics
  // I shall assume here that int is 32 bits, and long long 64 bits.  I guess these are
  // pretty much standard for now
  // Alternative: int_32t and int_64t (or int_fast_32t  and int_fast_64t, even more generally) 
  coordinate_vector l_size;
  long long l_volume;

public:

  // Information about the node stored on this process
  struct node_struct {
    int rank;                           // rank of this node
    unsigned sites, evensites, oddsites;
    unsigned field_alloc_size;          // how many sites/node in allocations 
    coordinate_vector min, size;        // node local coordinate ranges
    unsigned nn[NDIRS];                 // nn-node of node down/up to dirs
    bool first_site_even;               // is location min even or odd?
    std::vector<coordinate_vector> coordinates;

    void setup(node_info & ni, lattice_struct & lattice);

#ifdef SUBNODE_LAYOUT
    // If we have vectorized-style layout, we introduce "subnodes"
    // size is this_node.size/subnodes.divisions, which is not
    // constant across nodes
    struct subnode_struct {
      coordinate_vector divisions,size;  // div to subnodes to directions, size
      coordinate_vector offset[number_of_subnodes];  // coord shift to subnodes
      unsigned sites,evensites,oddsites;   

      void setup(const node_struct & tn);
    } subnodes;
#endif

  } this_node;


  // information about all nodes
  struct allnodes {
    int number;                            // number of nodes
    unsigned n_divisions[NDIM];            // number of node divisions to dir
    // lattice division: div[d] will have num_dir[d]+1 elements, last size
    // TODO: is this needed at all?
    std::vector<unsigned> divisors[NDIM];
    std::vector<node_info> nodelist;

    unsigned * RESTRICT map_array;                  // mapping (optional)
    unsigned * RESTRICT map_inverse;                // inv of it
    
    void create_remap();                   // create remap_node
    unsigned remap(unsigned i);            // use remap
    unsigned inverse_remap(unsigned i);    // inverse remap
    
  } nodes;


  struct comm_node_struct {
    unsigned rank;                         // rank of communicated with node
    unsigned sites, evensites, oddsites;
    unsigned buffer;                       // offset from the start of field array
    unsigned * sitelist;

    // Get a vector containing the sites of parity par and number of elements
    const unsigned * RESTRICT get_sitelist(parity par, int & size) const {
      if (par == ALL) { 
        size = sites;
        return sitelist;
      } else if (par == EVEN) {
        size = evensites;
        return sitelist;
      } else {
        size = oddsites;
        return sitelist + evensites;
      }
    }

    // The number of sites that need to be communicated
    unsigned n_sites(parity par) const {
      if(par == ALL){
        return sites;
      } else if(par == EVEN){
        return evensites;
      } else {
        return oddsites;
      }
    }

    // The local index of a site that is sent to neighbour
    unsigned site_index(int site, parity par) const {
      if(par == ODD){
        return sitelist[evensites+site];
      } else {
        return sitelist[site];
      }
    }

    // The offset of the halo from the start of the field array
    unsigned offset(parity par) const {
      if(par == ODD){
        return buffer + evensites;
      } else {
        return buffer;
      }
    }
  };

  // nn-communication has only 1 node to talk to
  struct nn_comminfo_struct {
    unsigned * index;
    comm_node_struct from_node, to_node;
    unsigned receive_buf_size;                    // only for general gathers
  };

  // general communication
  struct gen_comminfo_struct {
    unsigned * index;
    std::vector<comm_node_struct> from_node;
    std::vector<comm_node_struct> to_node;
    unsigned receive_buf_size;     
  };

  // nearest neighbour comminfo struct
  std::array<nn_comminfo_struct,NDIRS> nn_comminfo;

  // Main neighbour index array
  unsigned * RESTRICT neighb[NDIRS];

  // implement waiting using mask_t - unsigned char is good for up to 4 dim. 
  dir_mask_t * RESTRICT wait_arr_;

#ifdef SPECIAL_BOUNDARY_CONDITIONS
  // special boundary pointers are needed only in cases neighbour
  // pointers must be modified (new halo elements). That is known only during runtime.
  // is_on_edget is the only "general" info element here, true if the node to direction
  // dir is on lattice edge.
  struct special_boundary_struct {
    unsigned * neighbours;
    unsigned * move_index;
    unsigned offset, n_even, n_odd, n_total;
    bool is_needed;
    bool is_on_edge;
  };
  // holder for nb ptr info
  special_boundary_struct special_boundaries[NDIRS];
#endif

#ifndef VANILLA
  backend_lattice_struct *backend_lattice;
#endif

  void setup(int siz[NDIM], int &argc, char **argv);
  void setup(input & inputs);
  void setup_layout();
  void setup_nodes();
  
  #if NDIM == 4
  void setup(int nx, int ny, int nz, int nt, int &argc, char **argv);
  #elif NDIM == 3  
  void setup(int nx, int ny, int nz, int &argc, char **argv);
  #elif NDIM == 2
  void setup(int nx, int ny, int &argc, char **argv);
  #elif NDIM == 1
  void setup(int nx, int &argc, char **argv); 
  #endif


  void teardown();

  // Std accessors:
  // volume
  long long volume() { return l_volume; }

  // size routines
  int size(direction d) { return l_size[d]; }
  int size(int d) { return l_size[d]; }
  coordinate_vector size() {return l_size;}

  coordinate_vector mod_size(const coordinate_vector & v) { return mod(v, l_size); }

  int local_size(int d) { return this_node.size[d]; }
  unsigned local_volume() {return this_node.sites;}

  int node_rank() { return this_node.rank; }
  int n_nodes() { return nodes.number; }
  std::vector<node_info> nodelist() { return nodes.nodelist; }
  coordinate_vector min_coordinate(){ return this_node.min; }
  
  bool is_on_node(const coordinate_vector & c);
  int  node_rank(const coordinate_vector & c);
  unsigned site_index(const coordinate_vector & c);
  unsigned site_index(const coordinate_vector & c, const unsigned node);
  const unsigned field_alloc_size() const {return this_node.field_alloc_size; }

  void create_std_gathers();
  gen_comminfo_struct create_general_gather( const coordinate_vector & r);
  std::vector<comm_node_struct> 
  create_comm_node_vector( coordinate_vector offset, unsigned * index, bool receive);

  
  bool first_site_even() { return this_node.first_site_even; };

#ifdef SPECIAL_BOUNDARY_CONDITIONS
  void init_special_boundaries();
  void setup_special_boundary_array(direction d);

  const unsigned * get_neighbour_array(direction d, boundary_condition_t bc);
#else
  const unsigned * get_neighbour_array(direction d, boundary_condition_t bc) { 
    return neighb[d];
  }
#endif

  unsigned remap_node(const unsigned i);
  
  #ifdef EVEN_SITES_FIRST
  int loop_begin( parity P) const {
    if(P==ODD){
      return this_node.evensites;
    } else {
      return 0;
    }
  }
  int loop_end( parity P) const {
    if(P==EVEN){
      return this_node.evensites;
    } else {
      return this_node.sites;
    }
  }
  #else
  
  int loop_begin( parity P) const {
    if(P==EVEN){
      return this_node.evensites;
    } else {
      return 0;
    }
  }
  int loop_end( parity P) const {
    if(P==ODD){
      return this_node.evensites;
    } else {
      return this_node.sites;
    }
  }
  #endif

#ifndef SUBNODE_LAYOUT

  inline const coordinate_vector & coordinates( unsigned idx ) const {
    return this_node.coordinates[idx];
  }

  inline int coordinate( direction d, unsigned idx ) const {
    return this_node.coordinates[idx][d];
  }


#else

  inline const coordinate_vector coordinates( unsigned idx ) const {

    return  this_node.coordinates[idx / number_of_subnodes]
            + this_node.subnodes.offset[idx % number_of_subnodes];
  }

  inline int coordinate( direction d, unsigned idx ) const {

    return  this_node.coordinates[idx / number_of_subnodes][d]
            + this_node.subnodes.offset[idx % number_of_subnodes][d];
  }

#endif

  inline parity site_parity( unsigned idx ) const {
  #ifdef EVEN_SITES_FIRST
    if (idx < this_node.evensites) return EVEN;
    else return ODD;
  #else 
    return coordinates(idx).parity();
  #endif
  }

  coordinate_vector local_coordinates( unsigned idx ) const {
    return coordinates(idx) - this_node.min;
  }

  lattice_struct::nn_comminfo_struct get_comminfo(int d){
    return nn_comminfo[d];
  }

  /* MPI functions and variables. Define here in lattice? */
  void initialize_wait_arrays();
  #ifdef USE_MPI
  MPI_Comm mpi_comm_lat;
  #endif

  template <typename T>
  void reduce_node_sum(T * value, int N, bool distribute);

  template <typename T>
  void reduce_node_product(T * value, int N, bool distribute);

  // Guarantee 64 bits for these - 32 can overflow!
  unsigned long long n_gather_done = 0, n_gather_avoided = 0;

};

/// global handle to lattice
extern lattice_struct * lattice;

// Keep track of defined lattices
extern std::vector<lattice_struct*> lattices;

// and the MPI tag generator
int get_next_msg_tag();





#ifdef VANILLA
#include "../plumbing/backend_cpu/lattice.h"
#elif defined(CUDA)
#include "../plumbing/backend_cuda/lattice.h"
#elif defined(VECTORIZED)
#include "../plumbing/backend_vector/lattice_vector.h"
#endif




#endif
