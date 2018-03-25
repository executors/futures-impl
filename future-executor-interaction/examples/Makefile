EXAMPLES = \
	SimpleAsync \
	SimpleContinuations \
	SimpleSemiFutureGet \
	SimpleViaGet

TESTS = $(addsuffix .test, $(EXAMPLES))

CXXFLAGS = -std=c++14 -pthread -Wall -Wextra -I../include -I../../../executors-impl/include

.PHONY: all clean

all: $(EXAMPLES)

test: $(TESTS)

clean:
	rm -f $(EXAMPLES)

$(EXAMPLES): %: %.cpp TestHelper.h
	$(CXX) $(CXXFLAGS) -o$@ $<

%.test: %
	./$(basename $<)
