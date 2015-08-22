import subprocess;
import debug_module
print "Hello, World! \n" 

file_name_int = 0;
do_debug = 1;

def lookupFileName( file_name_int):
    return {
        0 : 'SConstruct',
        1 : 'AmocoFdmodSConstruct.py',
    }[file_name_int]

subprocess.call('pwd',shell=True)

scons_command = "scons -f %s %s=%d" % (lookupFileName(file_name_int),
                                       debug_module.do_debug_key_name ,
                                       do_debug)
print "scons_command = \n %s \n" % scons_command
subprocess.call(scons_command,shell=True)

print "Done!\n"
 