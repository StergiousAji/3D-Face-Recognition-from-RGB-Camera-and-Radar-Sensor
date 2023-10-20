import sys
import time
import socket

from soli_logging import SoliLogParser

if __name__ == "__main__":
   # if len(sys.argv) != 2:
   #     print('teststream.py <Soli logger IP/hostname>')
   #     sys.exit(0)

    parser = SoliLogParser()
    for i in range(3):
        sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Connecting...')
        sd.connect(('127.0.0.1', 22388))

        for j in range(25):
            data = sd.recv(32768)
            # print('Received {} bytes'.format(len(data)))
            if len(data) < 12000: # sometimes get an invalid packet on startup...
                continue

            params, burst = parser.parse_burst(data, clear_params=(j==0))
            if j == 0:
                print(params)
            print('> Received burst ID: {}'.format(burst.burst_id))

        print('Disconnecting...\n')
        sd.close()
        
        time.sleep(1)
