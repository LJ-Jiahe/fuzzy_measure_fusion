# Fusion with Choquet Integral

### Prerequisites

**Software**

```
Conda(python)
```

**Python packages**
Command may vary according to your environment.
```
conda install -c conda-forge cvxopt qpsolvers matplotlib tqdm
conda install -c anaconda numpy scipy
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### Usage
Ignore ipynb files for now, these are files I used to build up the final code. They will be cleaned up in future versions.

**Command**
```
python Fusion\ 1\ -\ All\ In\ One\ -\ MP.py <repetition> <max_Num_Sources> <multi_process>
```
<p>The command takes 2 or 3 parameters besides the .py file.</p>

<p><repetition> integer, how many times you want to run for the test.</p>

<p><max_Num_Sources> integer, has to be bigger than 3.</p>

<p><multi_process> optional, integer, how many processes you want to use. The code will utilize all available processes if not explicitly assigned.</p>
