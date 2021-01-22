import pandas as pd
import numpy as np
import argparse
#This script is to be used when adding new measurements. 
#It adds three columns: hardware peak performance, hardware bandwidth and total number of operations of a neural network in GOPs
#You can fill these 3 columns by hand, but if you have a lot of measurements, these script automatizes the process.
#Please verify the column names and if the values are correct for the 3 columns, for all hardware platforms, and for all CNNs

def main():
  #Variables - please alter these according to your file
  hardware_peak_performance_column_name = 'hw_peak_perf'
  hardware_bandwidth_column_name = "hw_bandwidth"
  number_operations_CNNs_gops = 'nn_total_operations'


  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i', '--input', required=True,
                      help='File to be processed. Three more columns will be added to the file')
  
  args = parser.parse_args()

  df = pd.read_csv(args.input)
  #Alter the numbers below if they are not correct. (eg.: 4,6; 6,71; 30.7; ...)
  df[hardware_peak_performance_column_name]=df.apply(lambda r: 0.96 if r.HWType == 'Ultra96-DPU' and r.Datatype == 'INT8'  else 
                                           ( 4.6  if r.HWType == 'ZCU104-DPU' and r.Datatype == 'INT8' else 
                                           ( 6.71 if r.HWType == 'ZCU102-DPU'   and r.Datatype == 'INT8' else 
                                           ( 30.7 if r.HWType == 'ZCU104-FINN'  and r.Datatype == 'INT2' else 
                                           ( 8.8  if r.HWType == 'ZCU104-FINN'  and r.Datatype == 'INT4' else 
                                           ( 30.7 if r.HWType == 'ZCU104-Bismo' and r.Datatype == 'INT2' else 
                                           ( 8.8  if r.HWType == 'ZCU104-Bismo' and r.Datatype == 'INT4' else 
                                           ( 1.33 if r.HWType == 'TX2' and r['Op mode'] =='maxn' and r.Datatype == 'FP16' else 
                                           ( 0.67 if r.HWType == 'TX2' and r['Op mode'] =='maxn' and r.Datatype == 'FP32' else 
                                           ( 1.15 if r.HWType == 'TX2' and r['Op mode'] =='maxp' and r.Datatype == 'FP16' else 
                                           ( 0.57 if r.HWType == 'TX2' and r['Op mode'] =='maxp' and r.Datatype == 'FP32' else 
                                           ( 0.87 if r.HWType == 'TX2' and r['Op mode'] =='maxq' and r.Datatype == 'FP16' else 
                                           ( 0.44 if r.HWType == 'TX2' and r['Op mode'] =='maxq' and r.Datatype == 'FP32' else 
                                           ( 4    if r.HWType == 'EdgeTPU' and r['Op mode'] =='fast'and r.Datatype == 'INT8' else 
                                           ( 2    if r.HWType == 'EdgeTPU' and r['Op mode'] =='slow'and r.Datatype == 'INT8' else 
                                           ( 1    if r.HWType == 'NCS2' and r.Datatype == 'INT8' else 
                                           ( 0.5  if r.HWType == 'NCS2' and r.Datatype == 'FP16' else 
                                           ( 0.192 if r.HWType == 'U96-Quadcore-A53' and r.Datatype == 'INT2' else 
                                           ( 0.192 if r.HWType == 'U96-Quadcore-A53' and r.Datatype == 'INT4' else 
                                           ( 0.192 if r.HWType == 'U96-Quadcore-A53' and r.Datatype == 'INT8'  else None ))))))))))))))))))) , axis=1)


  df[hardware_bandwidth_column_name] = df.apply(lambda r: 4.26 if r.HWType == 'Ultra96-DPU'  else 
                                          ( 19.2 if r.HWType == 'ZCU104-DPU'  else 
                                          ( 19.2 if r.HWType == 'ZCU102-DPU' else 
                                          ( 19.2 if r.HWType == 'ZCU104-FINN'  else 
                                          ( 19.2 if r.HWType == 'ZCU104-Bismo' else 
                                          ( 59.7 if r.HWType == 'TX2'  else 
                                          ( 25.6 if r.HWType == 'EdgeTPU'  else 
                                          ( 12.8 if r.HWType == 'NCS2' else 
                                          ( 4.26 if r.HWType == 'U96-Quadcore-A53'  else None )))))))) , axis=1)

  df[number_operations_CNNs_gops] = df.apply(lambda r: 3.1 if r.NN_Topology == 'GoogLeNetv1'  else 
                                          ( 1.1  if r.NN_Topology == 'MobileNetv1'  else 
                                          ( 7.7  if r.NN_Topology == 'ResNet-50' and r.PruningFactor == '100%' else 
                                          ( 6.5  if r.NN_Topology == 'ResNet-50' and r.PruningFactor == '80%' else 
                                          ( 3.8  if r.NN_Topology == 'ResNet-50' and r.PruningFactor == '50%' else 
                                          ( 2.5  if r.NN_Topology == 'ResNet-50' and r.PruningFactor == '30.00%' else 
                                          ( 0.47 if r.NN_Topology == 'CNV'  and r.PruningFactor == '100%' else
                                          ( 0.12 if r.NN_Topology == 'CNV'  and r.PruningFactor == '50%' else 
                                          ( 0.03 if r.NN_Topology == 'CNV'  and r.PruningFactor == '25%' else 
                                          ( 0.01 if r.NN_Topology == 'CNV'  and r.PruningFactor == '12.50%' else 
                                          ( 0.02 if r.NN_Topology == 'MLP'and r.PruningFactor == '100%' else 
                                          ( 0.00582 if r.NN_Topology == 'MLP'and r.PruningFactor == '50%' else 
                                          ( 0.0019  if r.NN_Topology == 'MLP'and r.PruningFactor == '25%' else 
                                          ( 0.0007  if r.NN_Topology == 'MLP'and r.PruningFactor == '12.50%'  else None ))))))))))))) , axis=1)
  file_name = "processed_" + args.input
  df.to_csv(file_name, index=False)

if __name__ == '__main__':
  main()