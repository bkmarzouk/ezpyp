# Easy Python Pipelines / _ezpyp_

Ezpyp enables rapid pipeline orchestration by translating functions into
pipeline steps and deriving the flow of dependencies.

### Quick start tutorial

Suppose that we have a set of functions `alpha`, `beta` and `gamma`,
where `alpha` and `beta` and independent, but `gamma` takes their outputs
as inputs.

```python
def alpha(x: int):
    return x ** 2


def beta(x: int):
    return x ** 3


def gamma(a, b):
    return a / b
```

We could turn this workflow into a pipeline with `ezpyp`

```python
from ezpyp import Pipeline, as_pickle_step
from ezpyp.steps import PlaceHolder

# Define a pipeline instance to link together functions as steps
pipeline = Pipeline("/home/pipeline_cache", "my_pipeline")


# Apply decorator to link alpha to pipeline
@as_pickle_step(pipeline)
def alpha(x):
    return x ** 2


# Not evaluated, yet
step_alpha = alpha(3)


# Apply decorator to link beta to pipeline
@as_pickle_step(pipeline)
def beta(x):
    return x ** 3


# Not evaluated, yet
step_beta = beta(2)


# For the final step, we define the dependencies of gamma to be other steps
@as_pickle_step(pipeline, depends_on=[step_alpha, step_beta])
def gamma(a, b):
    return a / b


# Still not evaluated, and we can use PlaceHolder to indicate
# that the gamma values explicitly depend on the outputs from alpha and beta
step_gamma = gamma(PlaceHolder(step_alpha), PlaceHolder(step_beta))

# Initialize the schema for our workflow (steps cannot be changed now)
pipeline.initialize_schema()

# Run the pipeline steps
pipeline.execute()

# Results from cache can be loaded with, e.g.
print(f"Pipeline result for gamma: {pipeline.get_result('gamma')}")
```

This will generate an output of the form

```text
[MPI process 00] [Running component of phase 00] [Executing Pickle Step Object 'alpha']
  ╰─> [PASS] 'alpha'
[MPI process 00] [Running component of phase 00] [Executing Pickle Step Object 'beta']
  ╰─> [PASS] 'beta'
[MPI process 00] [Running component of phase 01] [Executing Pickle Step Object 'gamma']
  ╰─> [PASS] 'gamma'
Pipeline result for gamma: 1.125
```

If exceptions are raised during in any of the steps, downstream steps whom
have an associated dependency will be skipped. A high-level results summary
for the pipline execution will be written to `pipeline.summary`

```text
Pipeline Completed @ /home/kareem/software/ezpyp/examples/simple

name            args            kwargs          phase           status          result
alpha           [3]             {}              0               0               /home/kareem/software/ezpyp/examples/simple/alpha.step
beta            [2]             {}              0               0               /home/kareem/software/ezpyp/examples/simple/beta.step
gamma           [9, 8]          {}              1               0               /home/kareem/software/ezpyp/examples/simple/gamma.step

```

- `name` This is the name of the step, i.e., the function name.
- `args` These are the values of arguments passed to the step function. 
- `kwargs` These are the keyword arguments passed to the step function.
- `phase` This is the phase of the pipeline the step belongs to.
- `status` This is the exit status from the pipeline step:
  - `0` no error raised during execution.
  - `1` error raised during execution.
  - `2` skipped due to error raised in previous pipeline phase.

See "The pipeline execution model" for more details.

### Schema initialization

The initialization of the pipeline schema will generate a
file `pipeline.schema`
in the cache location that has been defined. This cache schema is used to
ensure that multiple pipeline instances are not written to the same location:
pipelines should be reproducible and not overwrite existing results.

```text
{                  
  "0": [
    {
      "args": "[3]",
      "cache_location": "/home/kareem/software/ezpyp/examples/simple",
      "depends_on": "[]",
      "function_bytes": "b'\\x97\\x00|\\x00d\\x01z\\x08\\x00\\x00S\\x00'",
      "function_name": "alpha",
      "kwargs": "{}",
      "name": "alpha",
      "status_ext": "status",
      "step_ext": "step"
    },
    {
      "args": "[2]",
      "cache_location": "/home/kareem/software/ezpyp/examples/simple",
      "depends_on": "[]",
      "function_bytes": "b'\\x97\\x00|\\x00d\\x01z\\x08\\x00\\x00S\\x00'",
      "function_name": "beta",
      "kwargs": "{}",
      "name": "beta",
      "status_ext": "status",
      "step_ext": "step"
    }
  ],
  "1": [
    {
      "args": "[TMP for alpha, TMP for beta]",
      "cache_location": "/home/kareem/software/ezpyp/examples/simple",
      "depends_on": "[Pickle Step Object 'alpha', Pickle Step Object 'beta']",
      "function_bytes": "b'\\x97\\x00|\\x00|\\x01z\\x0b\\x00\\x00S\\x00'",
      "function_name": "gamma",
      "kwargs": "{}",
      "name": "gamma",
      "status_ext": "status",
      "step_ext": "step"
    }
  ]
}
```

### The pipeline execution model

After calling the schema validation, dependencies are expanded and simplified
in order to generate a collection of pipeline "phases", which represent a
hierarchy of execution order.

1. Simplify dependencies and build phases
2. Write schema
3. Execute pipeline
    - Iterate over phases
      - Compute steps
4. Tidy up and write results

In the simple example provided, the pipeline steps `step_alpha` and `step_beta`
belong to phase `0`, since these steps carry no dependency on other steps. By 
comparison, `step_gamma` belongs to phase `1`, since it depends on the 
completion of steps in phase `0`.

Dependency chains are automatically managed, such that, if another function 
`delta(g)` that was dependent on the completion of `step_gamma`, users do not
need to specify `depends_on=[step_gamma, step_alpha, step_beta`, but rather,
the highest-order step, i.e. `depends_on=[step_gamma]`.

### MPI 

Pipelines can be deployed in parallel, e.g., with

```shell
mpirun -n 4 python path/to/pipeline_script.py
```

The parallelism distributes pipeline steps belonging to the current phase of
the pipline to cores/cpus over MPI.