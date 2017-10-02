#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "svm.h"

#define cost_update_amount 1.1

char* line;
int max_line_len = 1024;
struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;
int dense = 0;
COST_TYPE loss_type = UNDEFINED;
char* loss_file = NULL;
double* mat = NULL;
FILE* fp = NULL;

void print_confusionMat(); //fprintf confusionMat to conf_record
void do_update_cost(); //increase cost
int **confusionMat;
int *pdata;
int max_error_row = -1;
int max_error_col = -1;
int error_num;

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

	int svm_type=svm_get_svm_type(model);
	double nloss = 0;
	int nr_class=svm_get_nr_class(model);
	int *labels=(int *) malloc(nr_class*sizeof(int));
	double *prob_estimates=NULL;
	int j;

	if (loss_type != UNDEFINED){
		if (loss_type == MATRIX_COST || loss_type == EXAMPLE_COST){
			fp = fopen(loss_file,"r");

			if(fp == NULL)
			{
				fprintf(stderr,"can't open loss file %s\n",loss_file);
				exit(1);
			}
		}
		if (loss_type == MATRIX_COST){
			mat = (double*) malloc(nr_class * nr_class*sizeof(double));
			for(int kk = 0; kk < nr_class*nr_class; ++kk){
				if (!fscanf(fp, "%lf", &(mat[kk]))){
					fprintf(stderr,
					"matrix loss %s read error\n", loss_file);
					exit(1);
				}
			}
			fclose(fp);
		}
	}

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
		}
	}

	while(1)
	{
		double target,v;

		if(dense){
			int i = 0;
			int index = 1;
			int c;

			while(1)
			{
				double tmp;

				if (fscanf(input, "%lf", &tmp) == EOF)
					goto show;
	
				c = fgetc(input);
				//read in consecutive spaces until the last one or newline
				while (isspace(c) && c != '\n'){
					int cnext = fgetc(input);
					if (!isspace(cnext)){
						ungetc(cnext, input);
						break;
					}
					c = cnext;				
				}

				//newline indicates label
				if (c == '\n' || c == EOF){
					target = tmp;
					x[i++].index = -1;
					break;
				}
				else if (isspace(c)){
					if (tmp != 0){
						if(i>=max_nr_attr-1)	// need one more for index = -1
						{
							max_nr_attr *= 2;
							x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
						}
						x[i].index = index;
						x[i].value = tmp;
						++i;
					}
					index++;
				}										
				else{
					fprintf(stderr, "unknown character in format: %c\n", c);
					exit(1);
				}
			}
		}
		else{
			int i = 0;
			int c;
	
			if (fscanf(input,"%lf",&target)==EOF)
				break;
	
			while(1)
			{
				if(i>=max_nr_attr-1)	// need one more for index = -1
				{
					max_nr_attr *= 2;
					x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
				}

				do {
					c = getc(input);
					if(c=='\n' || c==EOF) goto out2;
				} while(isspace(c));
				ungetc(c,input);
				fscanf(input,"%d:%lf",&x[i].index,&x[i].value);
				++i;
			}	

	out2:
			x[i++].index = -1;
		}

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			v = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g ",v);
			for(j=0;j<nr_class;j++)
				fprintf(output,"%g ",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			v = svm_predict(model,x);
			fprintf(output,"%g\n",v);
		}

		if(v == target)
			++correct;
		error += (v-target)*(v-target);
		sumv += v;
		sumy += target;
		sumvv += v*v;
		sumyy += target*target;
		sumvy += v*target;
		++total;

		confusionMat[(int)target - 1][(int)v - 1]++; //row is true, column is predict
		
		if (loss_type != UNDEFINED){
		    switch(loss_type){
		    case CLASS_COST:
			nloss += (v == target ? 0 : 1);
			break;
		    case ABS_COST:
			nloss += fabs(v-target);
			break;
		    case MATRIX_COST:
			nloss += mat[((int)target-1)*nr_class+(int)v-1];
			break;
		    case EXAMPLE_COST:
			double loss;
			int k;
			for(k=0;k<v;k++)
			    fscanf(fp, "%lf", &loss);
			nloss += loss;
			for(k=(int)v;k<nr_class;k++)
			    fscanf(fp, "%lf", &loss);
			break;
		    default:
			fprintf(stderr,"Error: unrecognized loss settings for C_SVC\n");
			exit(1);
		    }
		}
	}
show:
#if 0
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
		       ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
		       ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))
		       );
	}
	else
#endif
		printf("classification loss = %g (%d/%d) (classification)\n",
		       1.0 - (double)correct/total,total-correct,total);

	if (loss_type != UNDEFINED)
	    printf("Mean cost-sensitive loss = %g \n",nloss/total);	

	if(predict_probability)
	{
		free(prob_estimates);
		free(labels);
	}
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
 	"-D dense: whether to use dense formate in files, 0 or 1 (default 0)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
    srand(time(NULL));

	FILE *input, *output;
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'D':
				dense = atoi(argv[i]);
				break;
			case 'l':
				loss_type = (COST_TYPE)atoi(argv[i]);
				if (loss_type == MATRIX_COST || loss_type == EXAMPLE_COST){
					if (argv[i][1] != '.'){
						fprintf(stderr, 
							"loss argument format: %d.filename\n",
							loss_type);
						exit(1);
					}
					loss_file = (char*)malloc((strlen(argv[i]+2)+1)*sizeof(char));
					strcpy(loss_file, argv[i]+2);
				}
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}
	if(i>=argc)
		exit_with_help();
	
	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}
	
	line = (char *) malloc(max_line_len*sizeof(char));
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability)
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
		
		
	//confusion matrix
	int nr_class=svm_get_nr_class(model);
	confusionMat = (int **)malloc(nr_class * sizeof(int *));
	pdata = (int *)malloc(nr_class * nr_class * sizeof(int));
	for (int i = 0; i < nr_class; i++, pdata += nr_class)
		confusionMat[i] = pdata;
	memset(confusionMat[0], 0, nr_class * nr_class * sizeof(int));
	
	predict(input,output);//set confusionMat entry
	
	print_confusionMat();
	do_update_cost();
	
	svm_destroy_model(model);
	
	free(loss_file);
	free(confusionMat[0]);
	free(confusionMat);
	
	free(line);
	free(x);
	fclose(input);
	fclose(output);
	return 0;
}



void print_confusionMat(){
	int nr_class=svm_get_nr_class(model);
	FILE *fp = fopen("conf_record", "a+");
	if(fp == NULL){
		fprintf(stderr,"can't open conf_record\n");		    exit(1);
	}
	int max_error = 0;
	int total_correct = 0;
	int sum = 0;
	double g_mean = 1;
	for(int i = 0; i < nr_class; i++){
		int row_sum = 0;
		double recall = 0;
		for(int j = 0; j < nr_class; j++){
			if(i == j)
				total_correct += confusionMat[i][j];
			if(i != j && confusionMat[i][j] > max_error){
				max_error = confusionMat[i][j];
				max_error_row = i;
				max_error_col = j;
			}
			fprintf(fp, "%4d ", confusionMat[i][j]);
			printf("%4d ", confusionMat[i][j]);
			sum += confusionMat[i][j];
			row_sum += confusionMat[i][j];
		}
		if(row_sum > 0)
			recall = (double)confusionMat[i][i] / row_sum;
		g_mean *= recall;
		fprintf(fp, "\n");
		printf("\n");
	}
	g_mean = pow(g_mean, 1 / (double)nr_class);
	if(max_error == 0){
		max_error_row = -1;
		max_error_col = -1;
	}
	double cur_cost_error = 1.0 - (double)total_correct/sum;
	error_num = sum - total_correct;
	fprintf(fp, "(%d, %d), total=%d\n",max_error_row, max_error_col, sum);
	fprintf(fp, "classification loss = %g\n", cur_cost_error);
	fprintf(fp, "geometric mean = %g\n", g_mean);
	printf("geometric mean = %g\n", g_mean);
	fclose(fp);
}



void do_update_cost(){
	if(max_error_row == -1 || max_error_col == -1)
		return;
	FILE *fp;
	FILE *fp2;
	int max_class=svm_get_nr_class(model);

	fp = fopen(loss_file,"r");	
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}
	double *mat = (double *)malloc(max_class * max_class * sizeof(double));
	for(int kk = 0; kk < max_class*max_class; ++kk)
		if (!fscanf(fp, "%lf", &(mat[kk]))){
		fprintf(stderr,
			"matrix loss %s read error\n", loss_file);
		exit(1);
		}				
	fclose(fp);
	
	fp = fopen(loss_file,"w");	
	fp2 = fopen("cost_record", "a+");
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}
	if(fp2 == NULL){
	    fprintf(stderr,"can't open cost_record\n");		    exit(1);
	}
	
	for(int i = 0; i < max_class; i++){
		for(int j = 0; j < max_class; j++){
			if(i != j)//proportionally adjust cost
				mat[i * max_class + j] += (double)(max_class * (max_class - 1)) * confusionMat[i][j] / error_num;
			fprintf(fp, "%lf ", mat[i * max_class + j]);
			fprintf(fp2, "%6.2lf ", mat[i * max_class + j]);
		}
		fprintf(fp, "\n");
		fprintf(fp2, "\n");
	}
	fprintf(fp2, "\n");
	fclose(fp);
	fclose(fp2);
	free(mat);
}
