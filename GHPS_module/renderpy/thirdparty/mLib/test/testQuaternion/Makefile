.SUFFIXES:

CXX = g++
FLAGS = -g -std=c++14
# FLAGS += -I "src"
FLAGS += -I "../../include"
FLAGS += -I "../../src"
LFLAGS = -g

SRC = main.cpp
OBJS = $(SRC:.cpp=.o)
EXECUTABLE = testQuat

.PHONY:	all clean

all:	$(EXECUTABLE)

%.o:	%.cpp
	$(CXX) $(FLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf *.o
	rm -rf $(EXECUTABLE)
