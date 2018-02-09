INC_DIR = include
VPATH = src $(sort $(dir $(wildcard src/*/)))
EXE = main
SRC_FILES = $(wildcard src/*.cpp) $(wildcard src/*/*.cpp)
OBJ_DIR = obj
SRC_DIR = src
OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES)) 
CXX := g++ 
LINK_FLAGS =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
CXXFLAGS := -std=c++11 -fopenmp -O3 -I${INC_DIR} -mavx2 -mbmi2  -I${MKLROOT}/include #-ftree-vectorizer-verbose=1
$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(EXE) $(OBJ) ${LINK_FLAGS} 
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	g++ $(CXXFLAGS) -c -o $@ $<
clean:
	rm $(wildcard $(OBJ_DIR)/*.o) 
	rm $(wildcard $(OBJ_DIR)/*/*.o)
	rm main 
