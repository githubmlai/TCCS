import sys

do_debug_key_name = 'do_debug'
def doDebugOrNot(my_dictionary):    
    print sys._getframe().f_code.co_name    
    do_debug = int(my_dictionary.get(do_debug_key_name, '0'))
    if do_debug:
        import pydevd
        pydevd.settrace()