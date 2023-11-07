# Spike Sorting with A Dynamical Control Systems Approach
* Developing spike sorting method via a dynamical control systems approach.  
* System identification for electrical imagings and artifact
* Designing unknown input observer (UIO) to detect the spikes from the measurements
* The code requires on python >= 3.9.0. It can be installed on virtual environment "spike_venv" on the root folder.


## Package installations
```bash
pip3 install -r requirements.txt
```

## Running the procedure
```bash
python3 main.py [options]
```

options:
* --nsysid: running the neural system identification
* --asysid: running the artifact system identification
* --uiod: designing UIO
* --uncmask: determine the uncertain data mask
* --spikesorting: running the spike sorting
* --spikedetection: detecting the spike activation curve
Without options, the code executes all stages of the procedure.

## Visualization
```bash
python3 visualize.py
```
