from flask import session
from functools import wraps

# Deccorator is a function
# Decorator takes the decorated function as an argument
# Decorator returns a new function
# Decorator maintains the decorated functions signature 

def checker_logged_in(func):
    @wraps(func)
    def wrapper(*args,**kargs):
        if 'logged_in' in session:
            return func(*args,**kargs)
        return "You are not logged in."
    return wrapper







