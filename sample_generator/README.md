# Sample Generator:

Sample Generator generates and simulates a trace in the following steps:
  
  1. running BMAP-Trace to generate samples.
  2. merge and select samples to an MQSim trace.
  3. simulate the MQSim trace and generate a SNIS(io_sim) trace (.trace) and an output table (.csv).

Definitions:

  1. MQSim trace contains five columns, including ArrivalTime, Size, IOType, Offset, and VolumnID.
  2. SNIS(io_sim) trace contains the following columns: 
  "RequestID,ArrivalTime,IOType,Size,VolumeID,Offset,InitiatorID,TargetID"
  3. the output table contas following columns:
  "RequestID,ArrivalTime,DelayTime,FinishTime,InitiatorID,TargetID,IOType,Size,VolumeID,Offset
"
