<h1>Status: Initial Post-Mortem & Gap Analysis</h1>
<h1> Executive Summary </h1>
This audit identifies the structural disconnects between the current pygam implementation and the scikit-learn estimator contract. The primary goal of the GSoC project is to refactor the inheritance chain and parameter handling to ensure pyGAM models can be used in sklearn Pipelines and GridSearches without manual wrappers.

<h1>Cloned Contents Audit</h1>
<h1>###Cloned the Repo (URL: https://github.com/dswah/pyGAM.git )</h1>

<h1>### Intitial contents on cloning the repo </h1>
  <h3>    build_tools	</h3>    
  <h3>    gen_imgs.py	</h3>    
  <h3>    LICENSE</h3>    
  <h3>    pyproject.toml</h3>    
  <h3>    docs		</h3>    
  <h3>    imgs		</h3>    
  <h3>    pygam		</h3>    
  <h3>    README.md</h3>    

<h1>### Peformed editable install for dev dependencies using :    </h1>
<h2>pip install -e . </h2>

<h1>### Contents after editable install: </h1>
   <h3>build_tools	</h2>
   <h3>gen_imgs.py	</h3>
   <h3>LICENSE		</h3>
   <h3>pygam.egg-info	</h3>
   <h3>README.md</h3>
   <h3>docs		</h3>
   <h3>imgs		</h3>
   <h3>pygam		</h3>
   <h3>pyproject.toml</h3>


<h1> Current State </h1>
<h2>Test Performed after editable install using inbuilt test file : </h2>
<h2>### To verify the proper installation of all dependencies </h2>
python -m pytest pyGAM/pygam/tests


<h1>TEST RESULTS </h1>
========================================================================================== test session starts ===========================================================================================
platform darwin -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
rootdir: /Users/apple/Desktop/py_GAM_GSoC/pyGAM
configfile: pyproject.toml
plugins: cov-7.1.0
collected 163 items                                                                                                                                                                                      

pyGAM/pygam/tests/test_GAM_methods.py .........................................                                                                                                                    [ 25%]
pyGAM/pygam/tests/test_GAM_params.py .........                                                                                                                                                     [ 30%]
pyGAM/pygam/tests/test_GAMs.py ..........                                                                                                                                                          [ 36%]
pyGAM/pygam/tests/test_core.py ...                                                                                                                                                                 [ 38%]
pyGAM/pygam/tests/test_datasets.py ............                                                                                                                                                    [ 46%]
pyGAM/pygam/tests/test_gen_imgs.py .............                                                                                                                                                   [ 53%]
pyGAM/pygam/tests/test_gridsearch.py .............                                                                                                                                                 [ 61%]
pyGAM/pygam/tests/test_partial_dependence.py .........                                                                                                                                             [ 67%]
pyGAM/pygam/tests/test_penalties.py .........                                                                                                                                                      [ 73%]
pyGAM/pygam/tests/test_terms.py ...s.........................                                                                                                                                      [ 90%]
pyGAM/pygam/tests/test_utils.py ...............                                                                                                                                                    [100%]

============================================================================================ warnings summary ============================================================================================
pygam/tests/test_GAM_methods.py: 11 warnings
pygam/tests/test_GAM_params.py: 1 warning
pygam/tests/test_GAMs.py: 2 warnings
pygam/tests/test_gen_imgs.py: 24 warnings
pygam/tests/test_gridsearch.py: 8 warnings
pygam/tests/test_partial_dependence.py: 4 warnings
pygam/tests/test_terms.py: 1 warning
pygam/tests/test_utils.py: 4 warnings
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/pygam.py:1223: DeprecationWarning: Bitwise inversion '~' on bool is deprecated and will be removed in Python 3.16. This returns the bitwise inversion of the underlying int object and is usually not what you expect from negating a bool. Use the 'not' operator for boolean negation or ~int(x) if you really want the bitwise inversion of the underlying int.
    1.0 / n * dev - (~add_scale) * (scale) + 2.0 * gamma / n * edof * scale

pygam/tests/test_GAM_methods.py::test_summary
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/tests/test_GAM_methods.py:89: UserWarning: KNOWN BUG: p-values computed in this summary are likely much smaller than they should be. 
   
  Please do not make inferences based on these values! 
  
  Collaborate on a solution, and stay up to date at: 
  github.com/dswah/pyGAM/issues/163 
  
    mcycle_gam.summary()

pygam/tests/test_GAM_methods.py::test_more_splines_than_samples
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/tests/test_GAM_methods.py:107: UserWarning: KNOWN BUG: p-values computed in this summary are likely much smaller than they should be. 
   
  Please do not make inferences based on these values! 
  
  Collaborate on a solution, and stay up to date at: 
  github.com/dswah/pyGAM/issues/163 
  
    gam.summary()

pygam/tests/test_GAM_methods.py::test_summary_returns_12_lines
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/tests/test_GAM_methods.py:183: UserWarning: KNOWN BUG: p-values computed in this summary are likely much smaller than they should be. 
   
  Please do not make inferences based on these values! 
  
  Collaborate on a solution, and stay up to date at: 
  github.com/dswah/pyGAM/issues/163 
  
    mcycle_gam.summary()

pygam/tests/test_gridsearch.py::test_GCV_objective_is_for_unknown_scale
pygam/tests/test_gridsearch.py::test_GCV_objective_is_for_unknown_scale
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/links.py:181: RuntimeWarning: overflow encountered in exp
    return np.exp(lp)

pygam/tests/test_gridsearch.py::test_GCV_objective_is_for_unknown_scale
pygam/tests/test_gridsearch.py::test_GCV_objective_is_for_unknown_scale
  /Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/pygam.py:631: RuntimeWarning: invalid value encountered in multiply
    self.link.gradient(mu, self.distribution) ** 2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================================================================== short test summary info =========================================================================================
SKIPPED [1] pyGAM/pygam/tests/test_terms.py:68: failing at tolerance 1e-6
============================================================================== 162 passed, 1 skipped, 62 warnings in 22.93s ==============================================================================



<h1>Installed requirements as dev dependencies using:</h1>
pip install -r "pygam.egg-info/requires.txt"

<h2>###Current estimator and non estimator contents of the pygam library:  </h2>

['ExpectileGAM', 'GAM', 'GammaGAM', 'InvGaussGAM', 'LinearGAM', 'LogisticGAM', 'PackageNotFoundError', 'PoissonGAM', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '__version__', 'callbacks', 'core', 'distributions', 'f', 'intercept', 'l', 'links', 'penalties', 'pygam', 's', 'te', 'terms', 'utils', 'version']

<h2>*** Source of Identification: </h2>
 python -c "import pygam; print([item for item in dir(pygam)])"

<h2>### Current base class(GAM) estimators in the pygam library : </h2>
['ExpectileGAM', 'GAM', 'GammaGAM', 'InvGaussGAM', 'LinearGAM', 'LogisticGAM', 'PoissonGAM']
***Source Of Identification:
 python -c "import pygam; print([item for item in dir(pygam) if 'GAM' in item])"


 <h1>### Inheritance Audit for pygam</h1>
Inheritance for LinearGAM: ['LinearGAM', 'GAM', 'Core', 'MetaTermMixin', 'object']
Inheritance for GAM: ['GAM', 'Core', 'MetaTermMixin', 'object']
Current MRO for LinearGAM: ['LinearGAM', 'GAM', 'Core', 'MetaTermMixin', 'object']
Current MRO for GAM: ['GAM', 'Core', 'MetaTermMixin', 'object']

<h2>***Source Of Identification: </h2>
 python -c "import pygam
def print_mro(cls):
    print(f'Inheritance for {cls.__name__}: {[c.__name__ for c in cls.mro()]}')

print_mro(pygam.LinearGAM)
print_mro(pygam.GAM)
"

</h2> Conclusion </h2>
The Core class (the root of all GAMs) must inherit from BaseEstimator. Without this, sklearn tools like clone() and check_estimator fail to recognize the models as valid estimators.
and the MRO for GAM estimators show that Base Estimator from scikit-learn is missing for the pygam library




<h2>Comparision with sklearn base class estimators and contents </h2>
['BaseEstimator', 'BiclusterMixin', 'ClassNamePrefixFeaturesOutMixin', 'ClassifierMixin', 'ClassifierTags', 'ClusterMixin', 'DensityMixin', 'InconsistentVersionWarning', 'MetaEstimatorMixin', 'MultiOutputMixin', 'OneToOneFeatureMixin', 'OutlierMixin', 'ParamsDict', 'RegressorMixin', 'RegressorTags', 'ReprHTMLMixin', 'Tags', 'TargetTags', 'TransformerMixin', 'TransformerTags', '_HTMLDocumentationLinkMixin', '_IS_32BIT', '_MetadataRequester', '_SetOutputMixin', '_UnstableArchMixin', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_check_feature_names_in', '_clone_parametrized', '_fit_context', '_generate_get_feature_names_out', '_is_fitted', '_routing_enabled', 'check_array', 'check_is_fitted', 'clone', 'config_context', 'copy', 'defaultdict', 'estimator_html_repr', 'functools', 'get_config', 'get_tags', 'inspect', 'is_classifier', 'is_clusterer', 'is_outlier_detector', 'is_pandas_na', 'is_regressor', 'is_scalar_nan', 'np', 'platform', 're', 'validate_parameter_constraints', 'warnings']

<h2>Source Of Identification</h2>
python -c "import sklearn.base; print(dir(sklearn.base))"
<h1> Initialization (__init__) & Parameter Audit </h1>
<h2> The "Black Hole" Signature </h2>
<h1>According to sklearn Docs all the estimators follow the configuration of the BaseEstimator from sklearn.base</h1>
<h1>According to the MRO for GAM objects it can be found that the __init__ methods for the GAM Objects for base class can be found in pygam.core.Core</h2>
<h1>So to make verify if __init__ methods for both classes have same set of required arguements we use:</h1>

<h3>
python -c "              <br>
import inspect             <br>
import sklearn.base <br>
import pygam.core <br>

print(f'1. SKLEARN STANDARD (BaseEstimator): {inspect.signature(sklearn.base.BaseEstimator.__init__)}')
print(f'2. PYGAM REALITY (Core): {inspect.signature(pygam.core.Core.__init__)}')
"
<h3>OUTPUT </h3>
1. SKLEARN STANDARD (BaseEstimator): (self, /, *args, **kwargs)
2. PYGAM REALITY (Core): (self, name=None, line_width=70, line_offset=3)

</h3>

<h2>
Test Performed: inspect.signature(pygam.LinearGAM.__init__)

Analysis:
Violation: Use of **kwargs hides model parameters (like n_splines, lam) from sklearn introspection. <br>
the usage of **kwargs prevents the sklearn baseEstimator from reading the argumentss in the pygam __init__ method stored int the pygam.core.Core <br>
Violation: LinearGAM raises TypeError for valid parent parameters like name. <br>
</h2> 

<h1>The Underscore Storage Rule </h1>
<h1>Observation: pygam.core.Core.__init__ stores arguments with leading underscores.</h1>
Current Code: self._name = name
Sklearn Requirement: self.name = name (Parameters must be stored exactly as named in the signature).


<h1> Method Availability Audit </h1>
<h2> Missing Scikit-Learn Contract Methods </h2>
The following methods were tested using hasattr() on LinearGAM():
Method	Status	Role
get_params	Custom	Existing version is incompatible with sklearn's deep cloning.
set_params	Custom	Existing version uses non-standard force and deep logic.
__sklearn_tags__	MISSING	Required for modern sklearn compatibility (version 1.0+).


<h1> Direct Failure in Official Compatibility Test (check_estimator) </h1>
Test Command: sklearn.utils.estimator_checks.check_estimator(LinearGAM())
python -c "from sklearn.utils.estimator_checks import check_estimator; from pygam import LinearGAM; check_estimator(LinearGAM())"
<h1>OUTPUT </h1>

Traceback (most recent call last):
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/_tags.py", line 275, in get_tags
    tags = estimator.__sklearn_tags__()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/terms.py", line 494, in __getattr__
    return self._super_get(name)
           ~~~~~~~~~~~~~~~^^^^^^
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/pygam/terms.py", line 416, in _super_get
    return super(MetaTermMixin, self).__getattribute__(name)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
AttributeError: 'LinearGAM' object has no attribute '__sklearn_tags__'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from sklearn.utils.estimator_checks import check_estimator; from pygam import LinearGAM; check_estimator(LinearGAM())
                                                                                             ~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/estimator_checks.py", line 850, in check_estimator
    for estimator, check in estimator_checks_generator(
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~^
        estimator,
        ^^^^^^^^^^
    ...<3 lines>...
        mark=None,
        ^^^^^^^^^^
    ):
    ^
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/estimator_checks.py", line 569, in estimator_checks_generator
    for check in _yield_all_checks(estimator, legacy=legacy):
                 ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/estimator_checks.py", line 355, in _yield_all_checks
    tags = get_tags(estimator)
  File "/Users/apple/Desktop/py_GAM_GSoC/pyGAM/venv/lib/python3.14/site-packages/sklearn/utils/_tags.py", line 283, in get_tags
    raise AttributeError(
    ...<8 lines>...
    )
AttributeError: The following error was raised: 'LinearGAM' object has no attribute '__sklearn_tags__'. It seems that there are no classes that implement `__sklearn_tags__` in the MRO and/or all classes in the MRO call `super().__sklearn_tags__()`. Make sure to inherit from `BaseEstimator` which implements `__sklearn_tags__` (or alternatively define `__sklearn_tags__` but we don't recommend this approach). Note that `BaseEstimator` needs to be on the right side of other Mixins in the inheritance order.
<h1>Conclusion </h1>
Based On the research Upto now the reasons of incompatibility between pygam library and the sklearnAPI v1.7+ are:

1. Conclusion: Sources of Incompatibility
Based on the research and diagnostic tests performed on the pygam source code and its interaction with the scikit-learn (v1.7+) API, the following primary reasons for incompatibility have been identified:
<h1>Lack of Formal Inheritance (The MRO Gap)</h1>
The foundational class pygam.core.Core inherits directly from the Python object instead of sklearn.base.BaseEstimator.
Impact: Scikit-learn’s internal tools (like clone(), get_params(), and set_params()) rely on the "Senior Parent" to provide helper methods like _get_param_names(). Without this link, pyGAM models are not recognized as valid estimators.
<h1>Missing Modern Metadata (__sklearn_tags__)</h1>
Modern Scikit-Learn versions require an estimator to describe its capabilities (e.g., handling of NaNs, sparse data) via a __sklearn_tags__ method.
Impact: The check_estimator() suite fails immediately with an AttributeError because pyGAM does not implement or inherit this required metadata interface.
<h1>Violation of the "Explicit Signature" Rule</h1>
Scikit-Learn requires that all model "knobs" (parameters) be explicitly named in the __init__ method.
Discovery: Child classes like LinearGAM utilize **kwargs to pass parameters up to the Core parent.
Impact: Scikit-learn’s introspection "Robot" cannot see inside the **kwargs bucket. Consequently, it cannot "read" or "clone" essential parameters like n_splines or lam.
<h1>Non-Standard Parameter Storage (The Underscore Issue)</h1>
Scikit-Learn requires a 1:1 mapping where a parameter named name is stored exactly as self.name.
Discovery: In pygam/core.py, parameters are stored with leading underscores (e.g., self._name = name).
Impact: When Sklearn tries to access model.name, it finds nothing, leading to crashes during model cloning and hyperparameter tuning.
<h1>Custom Method Conflicts (get_params & set_params)</h1>
While pyGAM has methods with these names, they are custom-built and do not follow the Sklearn logic for "deep" copying or parameter validation.
Discovery: The pyGAM versions include non-standard arguments like force=False and custom filtering logic for "user-facing" vs "non-user-facing" attributes.
Impact: These custom versions override the standard Sklearn behavior, causing failures in nested objects (like Pipelines).

<h1>Proposed GSoC Technical Roadmap</h1>
Refactor pygam/core.py: <br>
Update Core to inherit from BaseEstimator. <br>
Flatten __init__ to store parameters without leading underscores. <br>
Implement a standard __sklearn_tags__ method. <br>
Explicit Signatures in pygam/pygam.py: <br>
Remove **kwargs from all child estimators (LinearGAM, etc.). <br>
Replace with explicit, default-valued arguments. <br>
Attribute Standardisation: <br>
Ensure all learned attributes (coefficients, p-values) end with a trailing underscore (e.g., self.coef_).
