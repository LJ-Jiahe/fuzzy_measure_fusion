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
python fusion-MP.py <repetition> <max_Num_Sources> <multi_process>
```
<p>The command takes 2 or 3 parameters besides the .py file.</p>

| Parameter | Description |
|--- | --- |
| &lt;repetition> | integer, how many times you want to run for the test. |
| &lt;max_Num_Sources> | integer, has to be bigger than 3. |
| &lt;multi_process> | optional, integer, how many processes you want to use. The code will utilize all available processes if not explicitly assigned. |

For test run after environmnet setup, startwith
```
<repetition> = 10
<max_Num_Sources> = 3
<multi_process> = 10
```

If that works, now use
```
<repetition> = 100
<max_Num_Sources> = 8
<multi_process> = 100
```
