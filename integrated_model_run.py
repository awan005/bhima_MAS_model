__author__ = 'ankun wang'

from datetime import datetime
from integrated_model import IntegratedModelFrame
from integrated_model_stage import IntegratedModel
import sys, getopt

time_counter_main = 0
years_to_run = 3

def main(argv):
    inputfile = 0
    name = ''
    opts, args = getopt.getopt(argv,"hp:n:",["pfile=","nfile="])
    for opt, arg in opts:
        if opt == '-h':
            print ('integrated_model_run.py -p <inputfile>')
            sys.exit()
        elif opt in ("-p", "--pfile"):
            inputfile = arg
        elif opt in ("-n", "--nfile"):
            name = arg
        else: 
            assert False, "unhandled option"
    
    
    start_time = datetime.now().time()
    
    # set up simulation model
    TestModel = IntegratedModel(int(inputfile)-1)
    TestModelFrame = IntegratedModelFrame(TestModel, firstTimestep=time_counter_main, lastTimeStep=years_to_run*12, parameter=int(inputfile)-1,Name=name)
    TestModelFrame.run()
    
    end_time = datetime.now().time()
    
    print("\nStart: " + start_time.isoformat())
    print("End:   " + end_time.isoformat())
    print(inputfile)

if __name__ == "__main__":
   main(sys.argv[1:])

































