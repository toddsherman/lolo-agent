CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
UNAME_S := $(shell uname -s)
LDLIBS ?=
ifeq ($(UNAME_S),Linux)
LDLIBS += -ldl
endif
TARGET := ../build/lolo-libretro-host

.PHONY: all clean

all: $(TARGET)

$(TARGET): libretro_host.cpp
	@mkdir -p ../build
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	@if [ -f $(TARGET) ]; then mv $(TARGET) /tmp/lolo-libretro-host.previous; fi
