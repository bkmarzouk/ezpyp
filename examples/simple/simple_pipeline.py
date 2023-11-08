from ezpyp import Pipeline, as_pickle_step
from ezpyp.steps import PlaceHolder

# Define a pipeline instance to link together functions as steps
pipeline = Pipeline(".", "my_pipeline")


# Apply decorator to link alpha to pipeline
@as_pickle_step(pipeline)
def alpha(x):
    return x**2


# Not evaluated, yet
step_alpha = alpha(3)


# Apply decorator to link beta to pipeline
@as_pickle_step(pipeline)
def beta(x):
    return x**3


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

# Load result from cache
print(f"Pipeline result for gamma: {pipeline.get_result('gamma')}")
