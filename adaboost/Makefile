all: learner

learner: dbg/utils.o dbg/lbp.o dbg/lbph.o dbg/weakctor.o dbg/evolution.o src/main.cpp dbg/adaboost.o 
	g++ -g --std=c++0x -O2 dbg/utils.o dbg/lbp.o dbg/lbph.o dbg/weakctor.o dbg/adaboost.o dbg/evolution.o src/main.cpp -L./libs -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lflandmark_shared -lopencv_video -lgomp -fopenmp -o learner

dbg/lbp.o: src/lbp.cpp include/lbp.hpp
	g++ --std=c++0x -c -O2 src/lbp.cpp -fopenmp -o dbg/lbp.o

dbg/lbph.o: src/lbph.cpp include/lbph.hpp
	g++ --std=c++0x -c -O2 src/lbph.cpp -fopenmp -o dbg/lbph.o

dbg/weakctor.o: src/weakctor.cpp include/weakctor.hpp
	g++ --std=c++0x -c -O2 src/weakctor.cpp -fopenmp -o dbg/weakctor.o

dbg/adaboost.o: src/adaboost.cpp include/adaboost.hpp
	g++ --std=c++0x -c -O2 src/adaboost.cpp -fopenmp -o dbg/adaboost.o

dbg/evolution.o: src/evolution.cpp include/evolution.hpp
	g++ --std=c++0x -c -O2 src/evolution.cpp -fopenmp -o dbg/evolution.o

dbg/utils.o: src/utils.cpp include/utils.hpp
	g++ --std=c++0x -c -O2 src/utils.cpp -fopenmp -o dbg/utils.o

clean:
	rm -rf learner dbg/*.o
