BUILD_DIR := ../build
BINARY_DIR := ../bin
SRC_DIRS := .

#----------------------------------------------------------------------------------------------------------------------

# Find all the C++ files we want to compile
# I'd use a bash command but not all client have bash
SRCS := \
	BitBoardDefine.cpp \
	BoardState.cpp \
	EvalCache.cpp \
	EvalNet.cpp \
	main.cpp \
	Move.cpp \
	MoveGeneration.cpp \
	MoveList.cpp \
	GameState.cpp \
	Search.cpp \
	SearchData.cpp \
	SearchLimits.cpp \
	StagedMoveGenerator.cpp \
	TimeManage.cpp \
	TranspositionTable.cpp \
	TTEntry.cpp \
	Zobrist.cpp \
	Pyrrhic/tbprobe.cpp \
	td-leaf/HalogenNetwork.cpp \
	td-leaf/td-leaf-learn.cpp \
	td-leaf/TrainableNetwork.cpp

# String substitution for every C/C++ file.
# As an example, hello.cpp turns into ./build/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)

#----------------------------------------------------------------------------------------------------------------------

WFLAGS = -Wall -Wextra -Wshadow -std=c++17
CFLAGS = -O3 $(WFLAGS) -g -ffast-math -DNDEBUG -flto -march=native 
RFLAGS = -O3 $(WFLAGS) -g -ffast-math -DNDEBUG -flto

POPCNTFLAGS = -DUSE_POPCNT -msse3 -mpopcnt
PEXTFLAGS   = $(POPCNTFLAGS) -DUSE_PEXT -mbmi2
AVX2FLAGS   = -DUSE_AVX2 -msse -msse3 -mpopcnt -mavx2 -msse4.1 -mssse3 -msse2

LDFLAGS   = -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -lm
CXXFLAGS := $(INC_FLAGS) -MMD -MP

ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    detected_OS := Windows
else
    detected_OS := $(shell uname)
endif

$(info detected OS is [${detected_OS}])

# Many users on windows use a environment to build so we must link statically
ifeq ($(detected_OS),Windows)
    LDFLAGS += -static
endif

$(info LDFLAGS = [${LDFLAGS}])

# Each target sets particular flags, and then builds the corresponding exe file.

# for OpenBench support, we don't set a EXE name for the default (native) builds
.PHONY: native
native: CXXFLAGS += $(CFLAGS) $(POPCNTFLAGS)
native: LDFLAGS += -flto
native: EXE = $(BINARY_DIR)/Halogen-native.exe
native: binary

.PHONY: debug
debug: CXXFLAGS += $(WFLAGS) -g
debug: EXE = $(BINARY_DIR)/Halogen-debug.exe
debug: binary

.PHONY: sanitize
sanitize: CXXFLAGS += $(WFLAGS) -g -fsanitize=undefined -fsanitize=address
sanitize: LDFLAGS += -fsanitize=undefined -fsanitize=address
sanitize: EXE = $(BINARY_DIR)/Halogen-sanitize.exe
sanitize: binary

.PHONY: nopopcnt
nopopcnt: CXXFLAGS += $(RFLAGS)
nopopcnt: LDFLAGS += -flto -static 
nopopcnt: EXE = $(BINARY_DIR)/Halogen-nopopcnt.exe
nopopcnt: binary

.PHONY: popcnt
popcnt: CXXFLAGS += $(RFLAGS) $(POPCNTFLAGS)
popcnt: LDFLAGS += -flto -static 
popcnt: EXE = $(BINARY_DIR)/Halogen-popcnt.exe
popcnt: binary

.PHONY: pext
pext: CXXFLAGS += $(RFLAGS) $(PEXTFLAGS)
pext: LDFLAGS += -flto -static 
pext: EXE = $(BINARY_DIR)/Halogen-pext.exe
pext: binary

.PHONY: popcnt-avx2
popcnt-avx2: CXXFLAGS += $(RFLAGS) $(AVX2FLAGS) $(POPCNTFLAGS)
popcnt-avx2: LDFLAGS += -flto -static 
popcnt-avx2: EXE = $(BINARY_DIR)/Halogen-popcnt-avx2.exe
popcnt-avx2: binary

.PHONY: pext-avx2
pext-avx2: CXXFLAGS += $(RFLAGS) $(AVX2FLAGS) $(PEXTFLAGS)
pext-avx2: LDFLAGS += -flto -static 
pext-avx2: EXE = $(BINARY_DIR)/Halogen-pext-avx2.exe
pext-avx2: binary

.PHONY: release
release:
	$(MAKE) native
	$(MAKE) nopopcnt
	$(MAKE) popcnt
	$(MAKE) pext
	$(MAKE) popcnt-avx2
	$(MAKE) pext-avx2

#----------------------------------------------------------------------------------------------------------------------

# The final build step.
.PHONY: binary
binary: $(OBJS)
	mkdir -p $(dir $(EXE))
	$(CXX) $(OBJS) -o $(EXE) $(LDFLAGS)

# Build step for C++ source
$(BUILD_DIR)/%.cpp.o: %.cpp clean_build
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean_build
clean_build:
	rm -rf $(BUILD_DIR)

.PHONY: clean
clean: clean_build
	rm -rf $(BINARY_DIR)

# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)