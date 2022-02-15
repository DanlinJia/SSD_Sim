/* -------------------------------------------------------------------------  
 * This program is based on a one-pass algorithm for the calculation of an  
 * array of autocorrelations r[1], r[2], ... r[K].  The key feature of this 
 * algorithm is the circular array 'hold' which stores the (K + 1) most 
 * recent data points and the associated index 'p' which points to the 
 * (rotating) head of the array. 
 * 
 * Data is read from a text file in the format 1-data-point-per-line (with 
 * no blank lines).  Similar to programs UVS and BVS, this program is
 * designed to be used with OS redirection. 
 * 
 * NOTE: the constant K (maximum lag) MUST be smaller than the # of data 
 * points in the text file, n.  Moreover, if the autocorrelations are to be 
 * statistically meaningful, K should be MUCH smaller than n. 
 *
 * Name              : acs.c  (AutoCorrelation Statistics) 
 * Author            : Steve Park & Dave Geyer 
 * Language          : ANSI C
 * Latest Revision   : 2-10-97 
 * Compile with      : gcc -lm acs.c
 * Execute with      : a.out < acs.dat
 * ------------------------------------------------------------------------- 
 */

#include <stdio.h>
#include <math.h>

#define K    1                             /* K is the maximum lag */
#define SIZE (K + 1)

  int main(void)
{
  long   i = 0;                   /* data point index              */
  double x;                       /* current x[i] data point       */
  double sum = 0.0;               /* sums x[i]                     */
  long   n;                       /* number of data points         */
  long   j;                       /* lag index                     */
  double hold[SIZE];              /* K + 1 most recent data points */
  long   p = 0;                   /* points to the head of 'hold'  */
  double cosum[SIZE] = {0.0};     /* cosum[j] sums x[i] * x[i+j]   */
  double mean;

  while (i < SIZE) {              /* initialize the hold array with */
    scanf("%lf\n", &x);           /* the first K + 1 data values    */
    sum     += x;
    hold[i]  = x;
    i++;
  }

  while (!feof(stdin)) {
    for (j = 0; j < SIZE; j++)
      cosum[j] += hold[p] * hold[(p + j) % SIZE];
    scanf("%lf\n", &x);
    sum    += x;
    hold[p] = x;
    p       = (p + 1) % SIZE;
    i++;
  }
  n = i;

  while (i < n + SIZE) {         /* empty the circular array */
    for (j = 0; j < SIZE; j++)
      cosum[j] += hold[p] * hold[(p + j) % SIZE];
    hold[p] = 0.0;
    p       = (p + 1) % SIZE;
    i++;
  } 

  mean = sum / n;
  for (j = 0; j <= K; j++)
    cosum[j] = (cosum[j] / (n - j)) - (mean * mean);

  printf("for %ld data points\n", n);
  printf("the mean is ... %8.6f\n", mean);
  printf("the stdev is .. %8.6f\n\n", sqrt(cosum[0]));
  printf("  j (lag)   r[j] (autocorrelation)\n");
  for (j = 1; j < SIZE; j++)
    printf("%3ld  %11.6f\n", j, cosum[j] / cosum[0]);

  return (0);
}
