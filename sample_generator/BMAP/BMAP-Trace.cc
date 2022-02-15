#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "rvgs.h"
#include "rngs.h"
#include "BMAP-Trace.h"
#include "BMAP.h"


void Usage()
{
  fprintf(stderr, "                                           \n\
    Usage:                                                    \n\
            trace map-inputfile rand-seed\n\
    \n");
}

void Welford_alg(double & mean, double & std, double req, long long int s) 
{
    std     = std  + pow(req - mean, 2) * (s - 1) / s;
    mean    = mean + (req - mean) / s;
}

/*****************************************************************
            MAIN: create the trace from the BMAP distribution
			06-06-2006 ningfang mi
*****************************************************************/
int main(int argn, char* argv[])
{
    FILE*   input;      //input: MAP file for the interarrival time

    FILE*   f_trace;   //output: file trace -- arrival time   file size  file id

    BMAP*	arr_process;
    char    filename[100];  //temp string

    int i, index, ind;

    
    if (argn != 3) {
        Usage();
        return -1;
    }
    
    PlantSeeds(atol(argv[2]));

    /******************  Build MAP states ****************************/
    input   	= fopen(argv[1], "r"); assert(input);
  
    sprintf(filename, "traces/trace-%s", argv[1]);
    f_trace   = fopen(filename, "w"); assert(f_trace);
    
    index   = 0;
   
    arr_process   = new BMAP;
    assert(arr_process->get_input(input, index) > 0);
    fclose(input);

    /***************** Simulation   *********************************/
    double real_arrival=0.0;
    int    real_a_bulk;
    double r_mean;
    double r_svar;
	
	
    ind = 0;

    while (ind < NUM) {
	  //real_arrival += arr_process->gen_interval(real_a_bulk, r_mean, r_svar); //arrival times
        	real_arrival = arr_process->gen_interval(real_a_bulk, r_mean, r_svar);  //interarrival time or service time
            for (i = 0; i < real_a_bulk; i++) {
				fprintf(f_trace, "%10.6lf \n", real_arrival);
               	fflush(f_trace);
            } // for
            ind += real_a_bulk;           
    } // while
    printf("mean: %10.6lf, svar: %10.6lf\n", r_mean, r_svar);
     
    free(arr_process);
    fclose(f_trace);
    return 0;
 
} // main
