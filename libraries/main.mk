# This is the main makefile of hila applications
#  - to be called in application makefiles 
#  - calls platform specific makefiles in directory "platforms" 
#

# If "make clean", don't worry about targets, platforms and options
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),cleanall)

# PLATFORM needs to be defined. Check.
ifndef PLATFORM
  $(error "make <goal> PLATFORM=<name>" required.  See directory "platforms" at top level)
endif

# allow only 0 or 1 goals in make
ifneq ($(words $(MAKECMDGOALS)), 1)
ifneq ($(words $(MAKECMDGOALS)), 0)
  $(error Use (at most) one goal in make)
endif
endif

.PRECIOUS: build/%.cpt build/%.o


## hilapp binary (TOP_DIR defined in calling Makefile)

HILAPP := $(TOP_DIR)/hilapp/build/hilapp

################

LIBRARIES_DIR := $(TOP_DIR)/libraries
PLATFORM_DIR := $(LIBRARIES_DIR)/platforms
HILA_INCLUDE_DIR := $(TOP_DIR)/libraries

HILAPP_DIR := $(dir $(HILAPP))

HILA_OBJECTS = \
  build/initialize.o \
  build/param_input.o \
  build/mersenne_inline.o \
  build/lattice.o \
  build/map_node_layout_trivial.o \
  build/memalloc.o \
  build/timing.o \
  build/test_gathers.o \
  build/com_mpi.o \
  build/com_single.o

# com_mpi / com_single could be moved to platforms, but they're protected by USE_MPI guards

# Read in the appropriate platform bits and perhaps extra objects
include $(PLATFORM_DIR)/$(PLATFORM).mk

# Define LAYOUT_VECTOR if vector (SUBNODE) layout is desired
ifdef LAYOUT_VECTOR
  HILA_OBJECTS += build/setup_layout_vector.o
	HILA_OPTS += -DSUBNODE_LAYOUT
else
	HILA_OBJECTS += build/setup_layout_generic.o
endif

# To force a full remake when changing platforms or targets
LASTMAKE := build/_lastmake.${MAKECMDGOALS}.${PLATFORM}

$(LASTMAKE): $(MAKEFILE_LIST)
	-mkdir -p build
	-rm -f build/_lastmake.*
	make clean
	touch ${LASTMAKE}



# Use all headers inside libraries for dependencies
HILA_HEADERS := $(wildcard $(TOP_DIR)/libraries/*/*.h) $(wildcard $(TOP_DIR)/libraries/*/*/*.h)

ALL_DEPEND := $(LASTMAKE) $(HILA_HEADERS)

HILA_OPTS += -I$(HILA_INCLUDE_DIR)

# Add the (possible) std. includes for hilapp
HILAPP_OPTS += -I$(HILAPP_DIR)/clang_include $(CUSTOM_HILAPP_OPTS)

#
#  GIT VERSION: tricks to get correct git version and build date
#  on the file

GIT_SHA := $(shell git rev-parse --short=8 HEAD)

ifneq "$(GIT_SHA)" "" 
HILA_OPTS += -DGIT_SHA_VALUE=$(GIT_SHA)
GIT_SHA_FILE := build/_git_sha_$(GIT_SHA)

# Force recompilation if git number has changed

$(GIT_SHA_FILE):
	-rm -f build/_git_sha_*
	touch $(GIT_SHA_FILE)

ALL_DEPEND += $(GIT_SHA_FILE)
	
endif

# Standard rules for creating and building cpdt files. These
# build .o files in the build folder by first running them
# through the 


build/%.cpt: %.cpp Makefile $(MAKEFILE_LIST) $(ALL_DEPEND) $(APP_HEADERS)
	mkdir -p build
	$(HILAPP) $(APP_OPTS) $(HILA_OPTS) $(HILAPP_OPTS) $< -o $@

build/%.o : build/%.cpt
	$(CC) $(CXXFLAGS) $(APP_OPTS) $(HILA_OPTS) $< -c -o $@

build/%.cpt: $(LIBRARIES_DIR)/plumbing/%.cpp $(ALL_DEPEND) $(HILA_HEADERS)
	mkdir -p build
	$(HILAPP) $(APP_OPTS) $(HILA_OPTS) $(HILAPP_OPTS) $< -o $@

	
# This one triggers only for cuda targets
build/%.cpt: $(LIBRARIES_DIR)/plumbing/backend_cuda/%.cpp $(ALL_DEPEND) $(HILA_HEADERS)
	mkdir -p build
	$(HILAPP) $(APP_OPTS) $(HILA_OPTS) $(HILAPP_OPTS) $< -o $@

	

endif
endif   # close the "clean" bracket

.PHONY: clean cleanall

clean:
	-rm -f build/*.o build/*.cpt build/_lastmake*

cleanall:
	-rm -f build/*
