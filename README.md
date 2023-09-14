# Pipeline

Repository that contain End-2-End pipeline to perform QUBIC frequency map-making, power spectrum estimation and cosmological analysis. There is 3 paths for data :

* Usual map-making by deconvolving multiple peaks
* Fake frequency map-making using PySM python package
* Spectrum-based map-making assuming perfect or gaussian foregrounds

To send jobs on computing clusters with SLURM system, use the command :

```
sbatch main.sh
```