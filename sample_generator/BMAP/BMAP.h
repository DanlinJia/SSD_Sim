/*******************************************************
    STATE: states in irreducible infinitesimal generator
********************************************************/
struct STATE {
    double  mean;            // mean service time
    double* p;               /* transmission probabilities
                                p[i*#states+j] is the probability of
                                transmission from this state to state j
                                in D_i */
    int      rand_ind;        // rand stream index of service time
    int      rand_trans;      // rand stream index of state transmission
    double   during;          // during time in this state
};

/*******************************************************
    STATE: object that generates BMAP distribution
********************************************************/
class BMAP  {

private:
        struct STATE*   states;     // states in BMAP
        int             numState;   // number of states
        int             numBulk;    // number of bulk arrivals
        int             curr_ind;   // index of current state
        double          mean;       // mean
        double          svar;       // \sum (x_i - mean)
        long long int   number;     // number of intervals generated

public:
        BMAP();
        ~BMAP();
        int get_input(FILE* input, int& index);
        double gen_interval(int& bulk, double & r_mean, double & r_svar);
        void print_P();
};
