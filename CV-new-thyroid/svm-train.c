#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define cost_update_amount 10


void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC\n"
	"	1 -- nu-SVC\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR\n"
	"	4 -- nu-SVR\n"
	"       5 -- CSOVO-SVC\n"
	"       6 -- WAP-SVC\n"
	"       7 -- CSOVA-SVC\n"
	"       8 -- CSPCR-ESVR\n"
	"       9 -- CSOSR\n"
	"      10 -- CSTREE\n"
	"      11 -- CSFT\n"
	"      12 -- CSAPFT\n"
	"      13 -- CSSECOC\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"       0 -- linear: u'*v\n"
	"       1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"       2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"       3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
 	"       4 -- stump: -|u-v|_1 + coef0\n"
 	"       5 -- perceptron: -|u-v|_2 + coef0\n"
 	"       6 -- laplacian: exp(-gamma*|u-v|_1)\n"
 	"       7 -- exponential: exp(-gamma*|u-v|_2)\n"
	"       8 -- precomputed kernel (kernel values in training_set_file)\n"	
	"-l loss : cost-sensitive loss of cost-sensitive formulations (default 1)\n"
	"       1 -- classification [y == f(x)] \n"
	"       2 -- absolute | y - f(x) | \n"
	"       3.matrix_file -- a max_class by max_class matrix C_{y, k}\n"
	"       4.example_file -- a prob.l by max_class matrix C_{i, k}\n"
	"       (the labels are assumed to be within 1,2,...,max_class)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/k)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of *-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-D dense: whether to use dense formate in files, 0 or 1 (default 0)\n"
	"-u n: number of update_cost, use confusionMat to adjust cost-matrix (only support cost-matrix and cross_validation)"
	);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem_dense(const char *filename);
void read_problem(const char *filename);
void do_cross_validation();
void construct_loss();


void confusionMat_CV(); //use confusionMat to adjust cost-matrix and reduce error by cross_validation
void print_confusionMat(int i); //fprintf confusionMat to conf_record
void do_update_cost(int i);//increase cost
void init_cost(); //init cost, set 1 on diagonal, 0 on off-diagonal
void copy_min_cost(int i); //copy the cost-matrix at ith-time update loop to min_cost
void copy_max_g_mean(int i); //copy the cost-matrix at ith-time update loop to min_cost



struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;
int dense = 0;
COST_TYPE loss_type;
char* loss_file;


int **confusionMat;
int *pdata;
int update_cost;
int nr_update_cost;
int max_error_row = -1;
int max_error_col = -1;
double min_cost_error = 1;
double cur_cost_error;
double max_g_mean = 0;
double cur_g_mean;

int error_num;

int main(int argc, char **argv)
{
    srand(time(NULL));

	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
 	if (dense)
 		read_problem_dense(input_file_name);
 	else
		read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}
	
	if (loss_type != UNDEFINED){
		construct_loss();
	}
	
	if(update_cost){	
		confusionMat_CV();
	}
	else if(cross_validation){
		do_cross_validation();
	}
	else{
		model = svm_train(&prob,&param);
		svm_save_model(model_file_name,model);
		svm_destroy_model(model);
	}
	
	svm_destroy_param(&param);

	if (prob.cost){
		free(prob.cost);
	}
	free(prob.y);
	free(prob.x);
	free(x_space);

	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double nloss = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else {
	    if (prob.cost){ //classification, but with the loss
		for(i=0;i<prob.l;i++){
		    if(target[i] == prob.y[i]){
				++total_correct;
			}
		    nloss += prob.cost[i*prob.max_class+(int)target[i]-1];
			if(update_cost)
				confusionMat[(int)prob.y[i] - 1][(int)target[i] - 1]++; // row is true, column is predict
		}
		printf("Cross Validation Mean classification loss = %g \n",1.0 - (double)total_correct/prob.l);
		printf("Cross Validation Mean cost-sensitive loss = %g \n",nloss/prob.l);		

	    }
		
	    else
	    {
		for(i=0;i<prob.l;i++)
		    if(target[i] == prob.y[i])
			++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	    }
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	prob.cost = NULL;
	cross_validation = 0;
	loss_type = CLASS_COST;
	loss_file = NULL;
	
	update_cost = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
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
					loss_file = Malloc(char, strlen(argv[i]+2)+1);
					strcpy(loss_file, argv[i]+2);
				}
				break;
			case 'u':
				update_cost = 1;
				nr_update_cost = atoi(argv[i]);
				if(nr_update_cost < 1)
				{
					fprintf(stderr,"number of update_cost: n must >= 1\n");
					exit_with_help();
				}
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}

//read in problem (in dense format)
void read_problem_dense(const char*filename)
{
	int elements, max_index, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		double tmp;

		fscanf(fp, "%lf", &tmp);

		int c = fgetc(fp);
		//read in consecutive spaces until the last one or newline
		while (isspace(c) && c != '\n'){
			int cnext = fgetc(fp);
			if (!isspace(cnext)){
				ungetc(cnext, fp);
				break;
			}
			c = cnext;				
		}

		//newline indicates label
		if (c == '\n'){
			++prob.l;
			++elements; //-1
		}
		else if (c == EOF)
			break;
		else if (isspace(c)){
			if (tmp != 0)
				++elements;
		}
		else{
			fprintf(stderr, "unknown character in format: %c\n", c);
			exit(1);
		}
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		int index = (param.kernel_type == PRECOMPUTED ? 0 : 1);

		prob.x[i] = &x_space[j];
		while(1)
		{
			double tmp;
	
			fscanf(fp, "%lf", &tmp);

			int c = fgetc(fp);

			//read in consecutive spaces until the last one or newline
			while (isspace(c) && c != '\n'){
				int cnext = fgetc(fp);
				if (!isspace(cnext)){
					ungetc(cnext, fp);
					break;
				}
				c = cnext;				
			}

			//newline indicates label
			if (c == '\n'){
				prob.y[i] = tmp;
				x_space[j].index = -1;
				--index;
				if (index > max_index)
					max_index = index;
				++j;
				break;
			}
			else if (isspace(c)){
				if (tmp != 0){
					x_space[j].index = index;
					x_space[j].value = tmp;
					++j;
				}
				++index;
			}
		}	
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

// read in a problem (in svmlight format)
void read_problem(const char *filename)
{
	int elements, max_index, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++prob.l;
				// fall through,
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		double label;
		prob.x[i] = &x_space[j];
		fscanf(fp,"%lf",&label);
		prob.y[i] = label;

		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			if (fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value)) < 2)
			{
				fprintf(stderr,"Wrong input format at line %d\n", i+1);
				exit(1);
			}
			++j;
		}	
out2:
		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}

void construct_loss(){
	int max_class = 0;
	int l = prob.l;
	FILE *fp = NULL;
	double *mat = NULL;	    

	{//scan for max_class
	    for(int i = 0; i < l; i++)
		if ((int)prob.y[i] > max_class)
		    max_class = (int)prob.y[i];
	    int *count = Malloc(int,max_class);
	    for(int k = 0; k < max_class; k++)
		count[k] = 0;	
	    for(int i = 0; i < l; i++){
		int y = (int)prob.y[i];
		if (y >= 1)
		    count[y-1]++;
		else{
		    fprintf(stderr,"label %d is not valid for loss setting\n", y);
		    exit(1);
		}
	    }

	    for(int k = 0; k < max_class; k++){
		if (count[k] == 0){
		    fprintf(stderr,"Warning: no examples in label %d", k+1);
		}
	    }	    
	    free(count);
	    prob.max_class = max_class;
	}

	double *weight = Malloc(double,max_class);
	{//set up per-class weights
	    for(int k = 0; k < max_class; k++)
		weight[k] = 1;	
	    
	    for(int i=0;i<param.nr_weight;i++){
		int label = param.weight_label[i];
		if (label >= 1 && label <= max_class)
		    weight[label-1] *= param.weight[i];
		else
		    fprintf(stderr,"Warning: class label %d specified in weight is not found\n", label);
	    }
	    if (param.nr_weight > 0){
		free(param.weight_label);
		free(param.weight);
		param.nr_weight = 0;
	    }
	}
	
	{//set up cost
	    prob.cost = Malloc(double,l*max_class);
	    
	    switch(loss_type){
	    case CLASS_COST:
		for(int i = 0; i < l; ++i){
		    int y = (int)prob.y[i];
		    for(int k=0;k<max_class; ++k)
			prob.cost[i*max_class+k] = (y == k+1 ? 0 : 1);
		}
		break;

	    case ABS_COST:
		for(int i = 0; i < l; ++i){
		    int y = (int)prob.y[i];
		    for(int k=0;k<max_class; ++k)
			prob.cost[i*max_class+k] = abs(k+1-y);
		}
		break;
	    case MATRIX_COST:
		fp = fopen(loss_file,"r");
		
		if(fp == NULL){
		    fprintf(stderr,"can't open loss file %s\n",loss_file);
		    exit(1);
		}
		
		mat = Malloc(double, max_class * max_class);
		for(int kk = 0; kk < max_class*max_class; ++kk)
		    if (!fscanf(fp, "%lf", &(mat[kk]))){
			fprintf(stderr,
				"matrix loss %s read error\n", loss_file);
			exit(1);
		    }				
		
		for(int i = 0; i < l; ++i){
		    int y = (int)prob.y[i];
		    for(int k=0;k<max_class; ++k)
			prob.cost[i*max_class+k] = mat[(y-1)*max_class+k];
		}
		fclose(fp);
		free(mat);
		break;
		
	    case EXAMPLE_COST:
		fp = fopen(loss_file,"r");
		
		if(fp == NULL){
		    fprintf(stderr,"can't open loss file %s\n",loss_file);
		    exit(1);
		}
		
		for(int i = 0; i < l; ++i)
		    for(int k = 0; k < max_class; ++k)
			if (!fscanf(fp, "%lf", &(prob.cost[i*max_class+k]))){
			    fprintf(stderr,
				    "example loss %s read error\n", loss_file);
			    exit(1);
			}				
		fclose(fp);
		free(loss_file);
		break;
		
	    default:
		fprintf(stderr,"Error: unrecognized loss settings\n");
		exit(1);
	    }
	}

	{//scale by per class weight
	    int mat_index = 0;
	    for(int i=0;i<l;i++)
		for(int k=0;k<max_class;k++)
		    prob.cost[mat_index++] *= weight[(int)prob.y[i]-1];

	    free(weight);
	}
	
}

void confusionMat_CV(){
	int nr_class = prob.max_class;
	confusionMat = (int **)malloc(nr_class * sizeof(int *));
	pdata = (int *)calloc(nr_class * nr_class, sizeof(int));
	for (int i = 0; i < nr_class; i++, pdata += nr_class)
		confusionMat[i] = pdata;
	
	for(int i = 0; i < nr_update_cost; i++){
		fprintf(stderr, "update times = %d\n", i);
		if(i != 0){
			do_update_cost(i);
			memset(confusionMat[0], 0, nr_class * nr_class * sizeof(int));
		}
		do_cross_validation(); //set confusionMat entry
		print_confusionMat(i);
		if(cur_cost_error < min_cost_error){
			min_cost_error = cur_cost_error;
			copy_min_cost(i);
		}
		if(cur_g_mean > max_g_mean){
			max_g_mean = cur_g_mean;
			copy_max_g_mean(i);
		}
		
	}
	free(loss_file);//others except matrix_cost had been free
	free(confusionMat[0]);
	free(confusionMat);
}

void print_confusionMat(int i){
	int nr_class = prob.max_class;
	FILE *fp = fopen("conf_record", "a+");
	if(fp == NULL){
		fprintf(stderr,"can't open conf_record\n");		    exit(1);
	}
	fprintf(fp, "update times = %d\n", i);
	int max_error = 0;
	int total_correct = 0;
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
			row_sum += confusionMat[i][j];
		}
		if(row_sum > 0)
			recall = (double)confusionMat[i][i] / row_sum;
		g_mean *= recall;
		fprintf(fp, "\n");
	}
	g_mean = pow(g_mean, 1 / (double)nr_class);
	if(max_error == 0){
		max_error_row = -1;
		max_error_col = -1;
	}
	cur_cost_error = 1.0 - (double)total_correct/prob.l;
	cur_g_mean = g_mean;
	error_num = prob.l - total_correct;
	fprintf(fp, "(%d, %d), total=%d\n",max_error_row, max_error_col, prob.l);
	fprintf(fp, "classification loss = %g\n", cur_cost_error);
	fprintf(fp, "geometric mean = %g\n", cur_g_mean);
	fclose(fp);
}


void do_update_cost(int i){
	if(max_error_row == -1 || max_error_col == -1)
		return;
	FILE *fp;
	FILE *fp2;
	int max_class = prob.max_class;
	int l = prob.l;
	
	fp = fopen(loss_file,"r");	
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}
	double *mat = Malloc(double, max_class * max_class);
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
	fprintf(fp2, "update times = %d\n", i); 
	
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
	
	//set up cost
	switch(loss_type){
		case MATRIX_COST:	
		for(int i = 0; i < l; ++i){
		    int y = (int)prob.y[i];
		    for(int k=0;k<max_class; ++k)
			prob.cost[i*max_class+k] = mat[(y-1)*max_class+k];
		}
		break;
		default:
		fprintf(stderr,"Error: only support cost-matrix update settings\n");
		exit(1);
	}
	free(mat);
}

void init_cost(){
	FILE *fp;
	int max_class = prob.max_class;
	fp = fopen(loss_file,"w");	
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}	
	for(int i = 0; i < max_class; i++){
		for(int j = 0; j < max_class; j++){
			if(i == j){ 
				fprintf(fp, "%lf ", (double)0);
			}
			else{
				fprintf(fp, "%lf ", (double)1);
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void copy_min_cost(int i){
	
	FILE *fp;
	FILE *fp2;
	char c;
	fp = fopen(loss_file,"r");
	fp2 = fopen("min_cost", "w");
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}
	if(fp2 == NULL){
		fprintf(stderr,"can't open min_cost\n");		    exit(1);
	}
	while((c = fgetc(fp)) != EOF){
		fputc(c, fp2);
	}
	fprintf(fp2, "min_error = %lf at %d update\n", min_cost_error, i);
	fclose(fp);
	fclose(fp2);
}

void copy_max_g_mean(int i){
	
	FILE *fp;
	FILE *fp2;
	char c;
	fp = fopen(loss_file,"r");
	fp2 = fopen("max_g_mean", "w");
	if(fp == NULL){
	    fprintf(stderr,"can't open loss file %s\n",loss_file);		    exit(1);
	}
	if(fp2 == NULL){
		fprintf(stderr,"can't open max_g_mean\n");		    exit(1);
	}
	while((c = fgetc(fp)) != EOF){
		fputc(c, fp2);
	}
	fprintf(fp2, "max_g_mean = %lf at %d update\n", max_g_mean, i);
	fclose(fp);
	fclose(fp2);
}
