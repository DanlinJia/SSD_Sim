#!/bin/bash

# usage: ./clear_log.sh $stat $exp
# e.g., ./clear_log.sh cache_test/write_1000ns_32sct_122gbps test
stat=$1
exp=$2

if [ -z "$stat" ]; then
    if [ -z "$exp"]; then
        rm response
        rm wc_queue_*
        rm *_tracker
        rm waiting 
        rm workload_scenario_1.xml 
    fi
    else

        des=$stat/$exp
        mkdir -p $des

        mv response $des
        mv wc_queue_* $des
        mv *_tracker $des
        mv waiting $des
        mv workload_scenario_1.xml $des
fi

