"""
NumPy Docstring Templates for psr-magpy

This file contains templates for different types of objects in the codebase.
"""

def function_template(param1, param2, optional_param=None):
    """
    Short description of function purpose.

    Longer description with details about the algorithm, approach,
    or mathematical foundation. Can include equations using LaTeX:
    
    .. math::
        y = \\sin(x)
    
    Parameters
    ----------
    param1 : type
        Description of param1. Include units if applicable [units].
    param2 : type
        Description of param2. Include valid ranges if applicable.
    optional_param : type, optional
        Description of optional parameter. Default is None.
        
    Returns
    -------
    return_type
        Description of return value. Include units if applicable [units].
        
    Raises
    ------
    ExceptionType
        When and why this exception is raised.
        
    See Also
    --------
    related_function : Brief description of relationship.
    
    Notes
    -----
    Additional implementation notes, constraints, or background.
    
    Examples
    --------
    >>> result = function_template(1, 'test')
    >>> print(result)
    Expected output
    """
    pass


class ClassTemplate:
    """
    Short description of class purpose.
    
    Longer description explaining the class functionality,
    use cases, and approach.
    
    Parameters
    ----------
    param1 : type
        Description of initialization parameter.
    
    Attributes
    ----------
    attr1 : type
        Description of class attribute.
    
    Examples
    --------
    >>> obj = ClassTemplate('value')
    >>> obj.some_method()
    Expected output
    """
    
    def __init__(self, param1):
        """Initialize the class."""
        self.attr1 = param1
        
    def some_method(self, param1):
        """
        Short description of method.
        
        Parameters
        ----------
        param1 : type
            Description of parameter.
            
        Returns
        -------
        return_type
            Description of return value.
        """
        pass 