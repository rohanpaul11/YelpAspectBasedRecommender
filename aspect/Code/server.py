import subprocess
import sys
import os
import time

# starts the CoreNLP server
def start_corenlp_server():
    # stop the server if it is already running
    stop_corenlp_server()

    print('Starting the corenlp server...')

    # assuming stanford parser folder is in current directory
    exec_dir = os.path.abspath('./stanford-corenlp-full-2018-10-05')
    # command that starts the server with preloaded annotations
    command = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &'

    # store the outputs in logfile
    with open('../Logs/corenlp.log', 'w') as logfile:
        p = subprocess.Popen(command, cwd=exec_dir, shell=True, stdout=logfile, stderr=logfile)
        p.wait()
        expired = True
        # keep checking for upto 3 minutes to see if server is up
        end = time.time() + 3 * 60
        while time.time() < end:
            try:
                retcode = subprocess.check_call('wget "localhost:9000/ready" -O -', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                expired = False
                break
            except subprocess.CalledProcessError as e:
                time.sleep(5)
                pass
                
        if expired:
            print('Server failed to start in 3 minutes!')
        
        # print(p.returncode)
        return p.returncode

# stops the CoreNLP server
def stop_corenlp_server():
    print('Stopping the corenlp server if it is running...')
    # command that initiates shutdown
    command = 'wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -'
    # append shutdown logs
    with open('../Logs/corenlp.log', 'a+') as logfile:
        p = subprocess.Popen(command, shell=True, stdout=logfile, stderr=logfile)        
        p.wait()        
        return p.returncode
    return 1

if __name__ == '__main__':
    start_corenlp_server()    
    stop_corenlp_server()
