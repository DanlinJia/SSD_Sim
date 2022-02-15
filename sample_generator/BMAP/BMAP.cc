#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "rngs.h"
#include "rvgs.h"
#include "BMAP.h"
#include "BMAP-Trace.h"

/*****************************************
    Constructer
*****************************************/
BMAP::BMAP() 
{
    states       = NULL;
    numState    = 0;
    numBulk     = 0;
    curr_ind    = 0;
    mean        = 0.0;
    svar        = 0.0;
    number      = 0;
}

/******************************************
    Deconstructer
******************************************/
BMAP::~BMAP()
{
    if (states) {
      for (int i = 0; i < numState; i++)
	free(states[i].p);
      free(states);
    }

}

/************************************************
    PURPOSE:    Build BMAP states
    INPUT:      input   -- input file
                index   -- random stream index
    RETURN:     >0  : success
                else: fail
*************************************************/
int BMAP::get_input(FILE* input, int& index)
{
        int err;
        int i, j, k;
        
        err = fscanf(input, "%d %d", &numState, &numBulk);
        
        /***        INITIALIZATION      ***/
        states = new struct STATE[numState];
        for (i = 0; i < numState; i++) {
            states[i].mean       = 0.0;
            states[i].rand_ind   = index++;
            states[i].rand_trans = index++;
            states[i].during     = 0.0;
            states[i].p          = new double    [(numBulk+1) * numState];
        }
        
        /***        INPUT               ***/
        for (j = 0; j < numBulk + 1 && err >= 0; j++) {
            for (i = 0; i < numState && err >= 0; i++) {
                for (k = 0; k < numState && err >= 0; k++) {
                    err = fscanf(input, "%lf", &states[i].p[j*numState+k]);
 		    printf("j:%d, i:%d, k:%d, states: %10.6lf\n", j, i, k, states[i].p[j*numState+k]);
                    if (states[i].p[j*numState+k] < 0.0) states[i].p[j*numState+k] = 0.0;
                    states[i].mean += states[i].p[j*numState+k];
                } // k
		printf("i: %d mean: %10.6lf\n", i, states[i].mean);
            } // i
        } // j

        for (j = 0; j < numBulk + 1 && err >= 0; j++) {
            for (i = 0; i < numState && err >= 0; i++) {
                for (k = 0; k < numState && err >= 0; k++) {
                    states[i].p[j*numState+k] = (states[i].mean == 0) ? 
                                                0 : states[i].p[j*numState+k] / states[i].mean;
		    printf("2 - j:%d, i:%d, k:%d, states: %10.6lf\n", j, i, k, states[i].p[j*numState+k]);
                    if (j*numState+k > 0)
                        states[i].p[j*numState+k] += states[i].p[j*numState+k-1];
		    printf("3 - j:%d, i:%d, k:%d, states: %10.6lf\n", j, i, k, states[i].p[j*numState+k]);
                } // k
            } // i
        } // j
		

        print_P();

        printf("err = %d\n", err); fflush(stdout);
        return err;
}


/*************************************************************************
 * PURPOSE:     Generate interarrival time for BMAP
 * RETURN:      the number which follows BMAP distribution
                bulk -- number of arrivals
 ************************************************************************/
double BMAP::gen_interval(int & bulk, double & r_mean, double & r_svar)
{
    double  interval    = 0.0;
    double  theo_mean   = 0.0;
    double  prob;
    int i;
    
    SelectStream(states[curr_ind].rand_ind);
    theo_mean = states[curr_ind].mean;
    (theo_mean < 0.000001) ? states[curr_ind].during = INF
                        : states[curr_ind].during = Exponential(1/theo_mean);
    
    interval += states[curr_ind].during;
    if (interval == INF) 
        return interval;
    
    SelectStream(states[curr_ind].rand_trans);
    prob = Uniform(0, 1);
    for (i = 0; i < numState*(numBulk+1); i++)
        if (prob <= states[curr_ind].p[i]) break;
    assert(i < numState*(numBulk+1));
    
    bulk= i / numState;
    i   = i % numState;
    curr_ind = i;
    
    if (bulk == 0)
        interval += gen_interval(bulk, r_mean, r_svar);
        
    number  ++;
    Welford_alg(mean, svar, interval, number);
    for (i = 0; i < bulk-1; i++) {
        number ++;
        Welford_alg(mean, svar, 0, number);
    }
    r_mean = mean;
    r_svar = svar;
    return interval;
}


/********************************************************
 Print out transmission probabilities
********************************************************/
void BMAP::print_P()
{
    int i, j;
    for (i = 0; i < numState; i++) {
        printf("\n--------------------- State %d ----------------------\n", i);
        for (j = 0; j < numState * (numBulk+1); j++) {
            printf("%10.6lf\t", states[i].p[j]);
        } // for j
        printf("\n");
    } // for j

    fflush(stdout);
}
