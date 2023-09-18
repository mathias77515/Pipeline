# Pipeline

Repository that contain End-2-End pipeline to perform QUBIC frequency map-making, power spectrum estimation and cosmological analysis. 

# Description

First step is compute frequency observations based on 3 way to do it :

* Usual map-making by deconvolving multiple peaks (large memory requirement)
* Fake frequency map-making using PySM python package using instrumental noise description
* Spectrum-based map-making assuming perfect or gaussian foregrounds (idealistic model)

# Run 

The code can be run locally but more efficient in Computing Cluster using SLURM system. To send jobs on computing clusters with SLURM system, use the command :

```
sbatch main.sh
```
