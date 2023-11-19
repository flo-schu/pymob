- ***The goal for the framework is to provide a principled and guided modeling workflow, which can be adapted to the needs of the model***
- ## Simulation
- The modelling toolkit offers a variety of algorithms that can be used on a large number of models. Recognizing that simulation, optimization/calibration, parameter inference, sensitivity analysis, validation, etc. require similar workflows and have common input and output datastreams opens the door for building a generic Simulation class, which can be reused for the listed purposes. On the other hand, each model or simulation is unique.
- This creates the need to create a platform, which provides the modeler with tools to skip the repetitive parts of the process of scientific programming to focus on the more important tasks, developing, testing models, and analyzing them.
- ***The model always comes first, the platform comes second. Therefore, when developing this framework, the idea was to allow the user as much freedom as possible and to enforce rules in only truly universal aspects of the modeling process.***
- ### Simulation components
  - Any simulation has recurring components. Facilitating transfer of information between those components is key to analyze a described model with different  tools. In this framework, simulations are defined as classes which require the definition of methods that define the simulation.
  - ```python
    from timepath import SimulationBase
    
    class Simulation(SimulationBase):
      def parameterize(self, input):
          # Initial conditions and parameters
    
      def set_coordinates(self, input):
          # the model is integrated over the dimension time
    
      def observations(self, input):
          # (optional) Defines the observations of the model
          # Needed for calibration, inference and validation tasks
    
      def run(self):
          # describe the model, solve it and return the solution
    
    ```
  - #### Input dimensions $$X$$ `self.coordinates`
    id:: 6516b82e-5e8d-4633-9549-224dfd1b9e18
    Model input can be in the easiest case just one value e.g. $$x = 0.5$$ if the model only needs to be solved for a single value. Often, we are interested in more than only one solution or want to solve the model for more than one dimension.
  - #### Free model parameters $$\theta$$  `self.free_model_parameters`
    id:: 6519c5af-1d32-45ee-8b60-1e48bd0e29f0
    These parameters are controlled by the `settings.cfg` file and in contrast to the model_parameters, they are allowed to vary between evaluations. Either systematically, for parameter studies, or randomly for bayesian approaches or manually in interactive simulations.
    ```conf
    [model-parameters]
    alpha.value = 0.1
    alpha.min = 0 
    alpha.max = 1
    alpha.prior = lognormal(scale=1,loc=0.1)
    ```
  - #### Model parameters `self.model_parameters`
    id:: 6516a746-d219-4246-ad63-a2ee40408541
    Model parameters are a critical part of the model. Changing model parameters changes the output of a model. Parameters are involved in many typical tasks of model development:
    + parameter calibration (or optimization)
    + parameter inference (slightly different operation)
    + senstivity analysis
    - In addition to the model parameters, the `parameterize` method also returns  the initial conditions of the simulation for initial value problems.
      - ```python
        def parameterize(self, input):
          # Initial conditions and parameters
          y0 = ...
          parameters = ...
          return parameters, y0
        ```
      - The input argument contains a list of file-paths. If any files were provided in the simulation `settings.cfg` configuration file. These files can then be  read and parameter names and values imported. This is particularly useful if parameters are provided in parallelization environments on computing clusters. For storing parameters, the use of `JSON` files is recommended. Those can be directly parsed as python dictionaries, which is the preferred format for  parameters, since they can be easily forwarded as keyword arguments in 
        functions. For this, a convenience function is included.
      - ```json
        {
          "y0": {
              "salad": 10, 
              "rabbits": 2},
          "parameters": {
              "eating_speed": 0.2,
              "growth_rate": 0.3,
          }
        }
        ```
      - ```python
        from timepath.store_file import read_config
        def parameterize(self, input):
          parameters = read_config(input[0])
          y0 = parameters["y0"]
          params = parameters["parameters"]
          return params, y0
        ```
  - #### Model Dimensions `self.set_coordinates`
    id:: 6516a746-8021-4961-9f44-88fb76e2f0d4
    The model dimensions describe the coordinates over which the model is solved. In a simple model only one coordinate is present. This dimension can be the time dimension. A very simple example:
    ```python
    def set_coordinates(self, input)
      time = np.linspace(0, 100, 50)
      return time
    ```
    Any number of dimension can be specified as long as those dimensions are also named in the same order in the `settings.cfg` file
    ```conf
    [simulation]
    dimensions = time
    ```
    - Models can also have a batch dimension, if it is part of the model that multiple
      replicates are simulated as one trial of an experiment.
      ```python
      def set_coordinates(self, input)
        time = np.linspace(0, 100, 50)
        sample = [1,2,3]
        return time, sample
      ```
      ```conf
      [simulation]
      dimensions = time sample
      ```
  - #### Results IO
    id:: 6516bc70-66fd-4ca2-826c-b3a0087e81fc
    The task of the `ResultsIO` method is to translate the evaluation output into a common format such as `numpy.ndarray` or `xarray.Dataset`.
    Implemented as `dataset` property
    - Results transformation is implemented as the `dataset` property. This requires the run method to return a `numpy.ndarray` with the correct dimensions.
    - If the results require more complex parsing than allowed by the dataset property, of course the property can be overwritten
  - **Prameter IO** parameterize
    id:: 6519c3ea-e60e-4890-83e4-c096a83a6804
    + simulation specific. 
    + Needs to translate parameters provided by the  priors, optimization variables, simulation variables, ...) to model parameters $$\theta$$
  - #### Model Data
    -
  -
  - #### Evaluator
    id:: 6516bdaa-02cc-4ea0-a317-a17fef5dc8af
    The evaluator is the comprehensive unit which takes standardized input arguments:
    + ((6519c5af-1d32-45ee-8b60-1e48bd0e29f0))
    + ((6516b82e-5e8d-4633-9549-224dfd1b9e18))
    Translates them into formats that can be parsed by the ((6516a746-abad-4b32-b6d1-7a6978b4d1d9)), evaluates the solver at all $$x$$ and transforms the returned ((6516bc2a-d109-433a-a9d0-c5629035b75f)) into an `xarray.Dataset` ( ((6516bc70-66fd-4ca2-826c-b3a0087e81fc)) )
    
    implemented as `SimulationBase.evaluate(self, theta)`
    - This function, is not exposed to the user, but instead acts as the primary evaluator function in various modeling tasks
    - The main interface is instead the `run` method, which implements the model and returns the result
    -
  - #### The `run` method
    id:: 6519c094-d76d-479b-901d-d0c6b8c1168f
    takes no arguments, but instead accesses `Simulation` attributes.
    Most important attributes:
    + `model_parameters`: [tuple] A tuple of all necessary model parameters and additional input, as returned by the `parameterize` method.
    + `coordinates`: [dict] The input variables which define the relevant dimensions of the model. These are the varibles that the model is solved for.
    - But, for instance in the `parameterize` method, any other attribute can be set that may be necessary for the specific needs of the model, following the maximum flexibility directive.
    - #### Solver
      id:: 6516a746-abad-4b32-b6d1-7a6978b4d1d9
      solve $f(x, \theta)$ for $x$ in $X$
      The taks of the solver is to compute the model for all coordinates of the input dimensions, and the provided model parameters. The solver translates parameters, dimensions and input into values that can be understood by the model, and if necessary integrates the model over the dimensions, i.e. if the model is a differential equation model.
      Examples: 
      ```python
      # integrate a function over time
      results = odeint(lotka_volterra, t, y0, args)
      
      # solve a function at different times
      for t in np.linspace(0, 10):
          y = f(t, theta)
          results.append(y)
      ```
      - This is defined by the `run` method. It returns the output of a single simulation run and takes no arguments. Extra input to the function is specified in the `initialize` method and can then be accessed as a class attribute.
      - ```python
        def run(self):
          y0 = self.y0
          params = self.model_parameters
          t = self.coordinates["time"]
        
          model = ...  # describe the model
        
          solution = ...  # Solve the model
          return solution
        ```
        
        The solution is returned as a `list` or `np.array`. Importantly, the shape of the output must follow the order of the dimensions returned in `set_coordinates`.
      - Output: Any
    - #### Model
      id:: 6516abb9-e688-4e03-b6c7-4b73c7fb3715
      The model is the core of the simulation. It takes parameters and other input and computes the output as a function of the input variables $$X$$ and parameters $$\theta$$.
      example: `lotka_volterra(y, t, alpha, beta, gamma, delta)`
      - Input can be a time, space, ...
      - $$f(X, \theta) = ...$$
      - The model can be empirical. E.g. the linear regression
      - $$f(x, a, b) = a x + b$$
      - where $$a$$ and $$b$$ are the parameters that make up $$\theta = (a, b)$$, and $$x$$ is the input of the input matrix $$X$$
      - of course we could formulate the linear function as a differential equation $$\frac{dy}{dx} = a$$
      - This would formulate our above problem as an *initial value problem* with the starting value of $$x_0 = b$$ and we would have to solve the model until the input value $$x$$
      - $$f(x, a, b) = \int_0^x a~ dx' +  b$$
      - While this operation can be solved easily, more complicated models, usually require solves for computing the function value at a given input
      - Input: $$X, \theta$$
        id:: 6516b326-2fd4-4e45-80cb-483461d6d347
        Format: Any
      - Output: $$Y$$
        id:: 6516b5c9-3521-4a26-89a5-4c4e29409199
        Format: Any
    - #### Model results $$Y$$
      id:: 6516bc2a-d109-433a-a9d0-c5629035b75f
      Solving the model gives a function value $$y$$ for the specified input $$\theta, x$$
- #### Observations
  - Observations must have the same form as the model output
- ### Input / Output
  - As a uniform exchange format `netcdf` is enforced by the package. In python `netcdf` files are handled by the `xarray` package. For convenience, data_variables of interest can be saved as a number of output formats. However, higher dimesnional datasets with more coordinates than e.g. time, must be aggregated over the remaining dimensions, before they can be processed to .csv files
    
    TODO: make sure this works
-