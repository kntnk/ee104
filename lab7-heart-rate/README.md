# FFT/IFFT Noise Cancelling and Heart Rate Analysis  

## Setup of the environment on Mac  
```
python3 -m venv lab7-env  
cd lab7-env  
source bin/activate  
```


## Installation of "numpy" on Mac  
In order to pip-install numpy on M1 mac, the architecture should be changed from arm to x86_64 by following the commands below.  
```
uname -m  
arch -x86_64 zsh  
pip install numpy  
```


## Additional required package  
```
pip install argparse  
pip install matplotlib  
pip install scipy  
pip install pandas    
pip install wave  
pip install heartpy  
pip install librosa  
```


## Execution of each program  
- Execute the FFT/IFFT Audio Signal Processing  
```
python3 lab7_main.py -f NC  
```

- Execute the Heart Rate Analysis - Time Domain Measurements  
```
python3 lab7_main.py -f HR  
```

- Execute the Heart Rate Diagnostic Analysis - Biotechnology  
```
python3 lab7_main.py -f HR2  
```