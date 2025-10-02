import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy # <--- 1. NEW: Import NumPy

# ----------------------------------------------------------------------
# 1. PLATFORM-SPECIFIC CONFIGURATION
# ----------------------------------------------------------------------

if sys.platform == 'win32':
    # Windows MSVC compiler configuration
    # /O2 is the standard optimization flag for MSVC.
    COMPILE_ARGS = ['/O2']
    # FIX: Explicitly link the Python library (python311.lib) to resolve LNK2001 
    # for Py_Perf_Counter, which is required on Windows for certain Cython features.
    LINKER_LIBS = ['python311'] 
else:
    # Linux/macOS GCC/Clang compiler configuration
    # Use standard high optimization and architecture tuning.
    COMPILE_ARGS = ['-O3', '-march=native']
    LINKER_LIBS = [] # Not needed on non-Windows platforms

# ----------------------------------------------------------------------
# 2. SOURCE FILE DEFINITIONS (UPDATE THESE AS NEEDED)
# ----------------------------------------------------------------------

# Define the package directory (must match the namespace used in imports)
CYTHON_SOURCE_DIR = "src_shared"

# List of all .pyx files to be compiled
PYX_FILES = [
    os.path.join(CYTHON_SOURCE_DIR, 'mcts_engine.pyx'),
    os.path.join(CYTHON_SOURCE_DIR, 'mcts_node.pyx'),
    os.path.join(CYTHON_SOURCE_DIR, 'utils.pyx'),
]

# Paths containing header files (.h or .pxd files)
INCLUDE_PATHS = [CYTHON_SOURCE_DIR] 

# 2. NEW: Add NumPy's dynamic include path
# This is necessary for the C compiler to find 'numpy/arrayobject.h'
INCLUDE_PATHS.append(numpy.get_include()) # <--- 2. NEW: Add NumPy path

# ----------------------------------------------------------------------
# 3. BUILD EXTENSIONS
# ----------------------------------------------------------------------

extensions = []
for pyx_file in PYX_FILES:
    # Derive the base module name (e.g., 'mcts_engine')
    module_name_base = os.path.basename(pyx_file).replace('.pyx', '')
    
    # Create the fully qualified package name (e.g., 'src_shared.mcts_engine')
    module_name = f"{CYTHON_SOURCE_DIR}.{module_name_base}"
    
    print(f"Adding Cython extension: {module_name} from {pyx_file}")
    
    ext = Extension(
        # The fully qualified name for the module
        name=module_name,
        # The source file(s)
        sources=[pyx_file], 
        # Pass the include path to the C compiler (now includes NumPy)
        include_dirs=INCLUDE_PATHS, 
        # Use platform-aware compilation flags
        extra_compile_args=COMPILE_ARGS,
        # Use explicit linker libraries for Windows fix
        libraries=LINKER_LIBS, 
        language='c'
    )
    extensions.append(ext)

# ----------------------------------------------------------------------
# 4. SETUP CALL
# ----------------------------------------------------------------------

setup(
    name="MCTS",
    # Pass the list of extensions to cythonize
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    install_requires=[
        'chess',
        'numpy',
        'torch' 
    ]
)
