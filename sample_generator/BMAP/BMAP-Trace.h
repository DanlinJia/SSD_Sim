#define INF             999999999   // Infinity number
#define NUM             100000 //10000000  //1215070   //1606433    // sample space
#define STA_INTERVAL    1           // statistics interval
#define LEV             3           // tieres of the system

#define FIFO            0           
#define PS              1
#define SJF		2
#define DELAY		3
#define INFSERV		4
#define SUPERPOS	5
#define SPLIT		6
#define ADT_SHF		7
#define	SRPT		8

#define TS              0.1         // time slice
#define DEBUG_QLEN      1
#define DEBUG_DEPT      2
#define DEBUG_ARRV      4
#define DEBUG_SERV      8
#define DEBUG_RESP      16

void Welford_alg(double & mean, double & std, double req, long long int s);
