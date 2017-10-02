CXXC = g++
CFLAGS = -Wall -O3 -fopenmp

all: svm-train svm-predict update

update: update.c
	$(CXXC) $(CFLAGS) update.c  -o update -lm
svm-predict: svm-predict.c svm.o
	$(CXXC) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXXC) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
svm.o: svm.cpp svm.h
	$(CXXC) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-train svm-predic update
