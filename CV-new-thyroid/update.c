#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

#define Error(s) {perror(s); exit(-1);}
#define MAX_T 3000
void train(void)
{
	pid_t pid;
	int status;
	if((pid = fork()) < 0){
		Error("train fork error");
	}
	else if(pid == 0){ //child
		if(execl("./svm-train", "./svm-train", "-s", "5", "-l", "3.cost", "../data/seg.lite", NULL) < 0){
			Error("train exec error");
		}
	}
	else{
		if(wait(&status) != pid){
			Error("wait train error");
		}
		return;
	}
}
void predict(void)
{
	pid_t pid;
	int status;
	if((pid = fork()) < 0){
		Error("predict fork error");
	}
	else if(pid == 0){ //child
		if(execl("./svm-predict", "./svm-predict", "-l", "3.cost", "../data/seg.t", "seg.lite.model", "x", NULL) < 0){
			Error("predict exec error");
		}
	}
	else{
		if(wait(&status) != pid){
			Error("wait predict error");
		}
		return;
	}
}

void print_times(int i)
{
	FILE *fp = fopen("conf_record", "a+");
	FILE *fp2 = fopen("cost_record", "a+");
	if(fp == NULL){
	    fprintf(stderr,"can't open conf_record\n");		    exit(1);
	}
	if(fp2 == NULL){
	    fprintf(stderr,"can't open cost_record\n");		    exit(1);
	}
	fprintf(fp, "update times = %d\n", i);
	fprintf(fp2, "update times = %d\n", i);
	fclose(fp);
	fclose(fp2);
}


int main(void)
{
	/*
	// construct input for train and predict 
	char pool1[100][100]={"./svm-train", "-s", "5", "-l", "3.cost", "usps"};
	char pool2[100][100]={"./svm-predict", "-l", "3.cost", "usps.t", "usps.model", "x"};
	int argc_train = 6;
	char **argv_train = (char **)malloc((argc_train + 1) * sizeof(char *));
	int argc_predict = 6;
	char **argv_predict = (char **)malloc((argc_train + 1) * sizeof(char *));;
	for(int i = 0; i < argc_train; i++)
		argv_train[i] = pool1[i];
	argv_train[argc_train] = NULL;
	for(int i = 0; i < argc_predict; i++)
		argv_predict[i] = pool2[i];
	argv_predict[argc_predict] = NULL;


	int update_times;
	scanf("%d", &update_times);
	for(int i = 0; i < update_times; i++){
		printf("update_times = %d", i);
		train_main(argc_train, argv_train);
		predict_main(argc_predict, argv_predict);
	}


	*/
	int update_times = MAX_T;
	for(int i = 0; i < update_times; i++){
		printf("update_times = %d\n", i);
		print_times(i);
		train();
		predict();
	}
}

