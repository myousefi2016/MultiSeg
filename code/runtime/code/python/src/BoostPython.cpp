#include "python/BoostPythonPrefix.hpp"

//
// note - for some reason, boost python's special debug build that uses the debug python interpreter
// doesn't export any symbols, so we have to explicitly include them in our project so they are visible
// at link time.
//
#include <libs/python/src/numeric.cpp>
#include <libs/python/src/list.cpp>
#include <libs/python/src/long.cpp>
#include <libs/python/src/dict.cpp>
#include <libs/python/src/tuple.cpp>
#include <libs/python/src/str.cpp>
#include <libs/python/src/slice.cpp>

#include <libs/python/src/converter/from_python.cpp>
#include <libs/python/src/converter/registry.cpp>
#include <libs/python/src/converter/type_id.cpp>
#include <libs/python/src/object/enum.cpp>
#include <libs/python/src/object/class.cpp>
#include <libs/python/src/object/function.cpp>
#include <libs/python/src/object/inheritance.cpp>
#include <libs/python/src/object/life_support.cpp>
#include <libs/python/src/object/pickle_support.cpp>
#include <libs/python/src/errors.cpp>
#include <libs/python/src/module.cpp>
#include <libs/python/src/converter/builtin_converters.cpp>
#include <libs/python/src/converter/arg_to_python_base.cpp>
#include <libs/python/src/object/iterator.cpp>
#include <libs/python/src/object/stl_iterator.cpp>
#include <libs/python/src/object_protocol.cpp>
#include <libs/python/src/object_operators.cpp>
#include <libs/python/src/wrapper.cpp>
#include <libs/python/src/import.cpp>
#include <libs/python/src/exec.cpp>
#include <libs/python/src/object/function_doc_signature.cpp>

#include "python/BoostPythonSuffix.hpp"
