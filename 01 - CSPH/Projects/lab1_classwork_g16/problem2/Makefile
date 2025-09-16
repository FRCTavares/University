
CXX=g++ -m64
CXXFLAGS=-Icommon/ -Iobjs/ -O3 -std=c++11 -Wall

APP_NAME=problem2
OBJDIR=objs
COMMONDIR=common


default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/problem2.o $(OBJDIR)/evalSerial.o $(OBJDIR)/evalThread.o 

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) -lm -lpthread

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/problem2.o: $(COMMONDIR)/CycleTimer.h

