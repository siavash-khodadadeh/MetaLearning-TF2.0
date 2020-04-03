def name_repr(name):
    def wrapper(func):
        func.name = name
        return func
    return wrapper
