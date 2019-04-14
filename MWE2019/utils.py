try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm    
except NameError:
    from tqdm import tqdm