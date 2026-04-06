/*
 * sparsetools.cxx
 *
 * Python module wrapping the sparsetools C++ routines.
 *
 * Each C++ routine is templated vs. an integer (I) and a data (T) parameter.
 * The `generate_sparsetools.py` script generates `*_impl.h` headers
 * that contain thunk functions with a datatype-based switch statement calling
 * each templated instantiation.
 *
 * `generate_sparsetools.py` also generates a PyMethodDef list of Python
 * routines and the corresponding functions call the thunk functions via
 * `call_thunk`.
 *
 * The `call_thunk` function below determines the templated I and T data types
 * based on the Python arguments. It then allocates arrays with pointers to
 * the raw data, with appropriate types, and calls the thunk function after
 * that.
 *
 * The types of arguments are specified by a "spec". This is given in a format
 * where one character represents one argument. The one-character values are
 * listed below in the call_spec function.
 */

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_sparse_sparsetools_ARRAY_API

#include <Python.h>

#include <string>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include "numpy/ndarrayobject.h"

#include "sparsetools.h"
#include "util.h"

#define MAX_ARGS 16

static const int supported_I_typenums[] = {NPY_INT32, NPY_INT64};
static const int n_supported_I_typenums = sizeof(supported_I_typenums) / sizeof(int);

static const int supported_T_typenums[] = {NPY_BOOL,
                                           NPY_BYTE, NPY_UBYTE,
                                           NPY_SHORT, NPY_USHORT,
                                           NPY_INT, NPY_UINT,
                                           NPY_LONG, NPY_ULONG,
                                           NPY_LONGLONG, NPY_ULONGLONG,
                                           NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                                           NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE};
static const int n_supported_T_typenums = sizeof(supported_T_typenums) / sizeof(int);

static void *_cast_arrays(PyObject *pyarg, const int cur_typenum, const int output);
static int _safe_type(
    const int arg_type, const int cur_type, const int n_valid_types, const int *valid_types
);
static PyObject *array_from_std_vector_and_free(int typenum, void *p);
static void *allocate_std_vector_typenum(int typenum);
static void free_std_vector_typenum(int typenum, void *p);
static PyObject *c_array_from_object(PyObject *obj, int typenum, int is_output);
static void *allocate_csr_array(int I_tnum, int T_tnum, void* c[]);
static void free_csr_array(int I_tnum, int T_tnumi, void *p);

/*
 * Call a thunk function, dealing with input and output arrays.
 *
 * Resolves the templated <integer> and <data> dtypes from the `args` argument
 * list.
 *
 * Parameters
 * ----------
 * ret_spec : {'i', 'v'}
 *     Return value spec. 'i' for integer, 'v' for void.
 * spec
 *     String whose each character specifies a types of an
 *     argument:
 *
 *     'i': <integer> scalar
 *     'I': <integer> array
 *     'T': <data> array
 *     'V': std::vector<integer>
 *     'W': std::vector<data>
 *     'B': npy_bool array
 *     '*': indicates that the next argument is an output argument
 * thunk : PY_LONG_LONG thunk(int I_typenum, int T_typenum, void **)
 *     Thunk function to call. It is passed a void** array of pointers to
 *     arguments, constructed according to `spec`. The types of data pointed
 *     to by each element agree with I_typenum and T_typenum, or are bools.
 * args
 *     Python tuple containing unprocessed arguments.
 *
 * Returns
 * -------
 * return_value
 *     The Python return value
 *
 */
PyObject *
call_thunk(char ret_spec, const char *spec, thunk_t *thunk, PyObject *args)
{
    void *arg_list[MAX_ARGS];
    PyObject *arg_arrays[MAX_ARGS];
    int is_output[MAX_ARGS];
    PyObject *return_value = NULL;
    int I_typenum = NPY_INT32;
    int T_typenum = -1;
    int VW_count = 0;
    int I_in_arglist = 0;
    int T_in_arglist = 0;
    bool I_is_int64;
    int next_is_output = 0;
    int j, arg_j;
    const char *p;
    PY_LONG_LONG ret;
    Py_ssize_t max_array_size = 0;
    NPY_BEGIN_THREADS_DEF;

    if (!PyTuple_Check(args)) {
        PyErr_SetString(PyExc_ValueError, "args is not a tuple");
        return NULL;
    }

    for (j = 0; j < MAX_ARGS; ++j) {
        arg_list[j] = NULL;
        arg_arrays[j] = NULL;
        is_output[j] = 0;
    }


    /*
     * Detect data types in the signature
     */
    arg_j = 0;
    j = 0;
    for (p = spec; *p != '\0'; ++p, ++j, ++arg_j) {
        int type, atype;
        PyObject *arg, *csr_part;

        if (j >= MAX_ARGS) {
            PyErr_SetString(PyExc_ValueError,
                            "internal error: too many arguments in spec");
            goto fail;
        }

        is_output[j] = next_is_output;
        next_is_output = 0;

        switch (*p) {
        case '*':
            next_is_output = 1;
            --j;
            --arg_j;
            continue;
        case 'i':
        case 'l':
            /* Integer scalars */
            arg = PyTuple_GetItem(args, arg_j);
            if (arg == NULL) goto fail;

            Py_INCREF(arg);
            arg_arrays[j] = arg;
            continue;
        case 'I':
            /* Integer arrays */
            I_in_arglist = 1;

            arg = PyTuple_GetItem(args, arg_j);
            if (arg == NULL) goto fail;

            arg_arrays[j] = c_array_from_object(arg, -1, is_output[j]);
            if (arg_arrays[j] == NULL) goto fail;

            atype = PyArray_TYPE((PyArrayObject *)arg_arrays[j]);
            type = _safe_type(atype, I_typenum, n_supported_I_typenums, supported_I_typenums);
            if (type == -1) goto fail;

            I_typenum = type;
            continue;
        case 'T':
            /* Data arrays */
            T_in_arglist = 1;

            arg = PyTuple_GetItem(args, arg_j);
            if (arg == NULL) goto fail;

            arg_arrays[j] = c_array_from_object(arg, -1, is_output[j]);
            if (arg_arrays[j] == NULL) goto fail;

            atype = PyArray_TYPE((PyArrayObject *)arg_arrays[j]);
            type = _safe_type(atype, T_typenum, n_supported_T_typenums, supported_T_typenums);
            if (type == -1) goto fail;

            T_typenum = type;
            continue;
        case 'C':
            /* csr_struct */
            arg = PyTuple_GetItem(args, arg_j);
            if (arg == NULL)  goto fail;

            // csr_struct has 5 parts: iiIIT
            for (int i = 0; i < 5; i++, j++) {
              csr_part = PyTuple_GetItem(arg, i);
              if (csr_part == NULL) goto fail;

              Py_INCREF(csr_part);
              if (i < 2) {
                arg_arrays[j] = csr_part;
              } else {
                arg_arrays[j] = c_array_from_object(csr_part, -1, is_output[j]);
                if (arg_arrays[j] == NULL) goto fail;

                atype = PyArray_TYPE((PyArrayObject *)arg_arrays[j]);

                if (i < 4) {
                  type = I_typenum;
                  type = _safe_type(atype, type, n_supported_I_typenums, supported_I_typenums);
                  if (type == -1) goto fail;
                  I_typenum = type;
                } else {
                  type = T_typenum;
                  type = _safe_type(atype, type, n_supported_T_typenums, supported_T_typenums);
                  if (type == -1) goto fail;
                  T_typenum = type;
                }
              }
            }
            continue;
        case 'B':
            /* Boolean arrays */
            arg = PyTuple_GetItem(args, arg_j);
            if (arg == NULL) {
                goto fail;
            }
            arg_arrays[j] = c_array_from_object(arg, NPY_BOOL, is_output[j]);
            if (arg_arrays[j] == NULL) {
                goto fail;
            }
            continue;
        case 'V':
            /* std::vector integer output array */
            I_in_arglist = 1;
            --arg_j;
            VW_count += 1;
            continue;
        case 'W':
            /* std::vector data output array */
            T_in_arglist = 1;
            --arg_j;
            VW_count += 1;
            continue;
        default:
            PyErr_SetString(PyExc_ValueError, "unknown character in spec");
            goto fail;
        }
    }

    if (arg_j != PyTuple_Size(args)) {
        PyErr_SetString(PyExc_ValueError, "too many arguments");
        goto fail;
    }

    if ((I_in_arglist && I_typenum == -1) ||
        (T_in_arglist && T_typenum == -1)) {
        PyErr_SetString(PyExc_ValueError,
                        "unsupported data types in input");
        goto fail;
    }

    I_is_int64 = PyArray_EquivTypenums(I_typenum, NPY_INT64);

    /*
     * Cast and extract argument arrays
     */
    j = 0;
    arg_j = 0;
    for (p = spec; *p != '\0'; ++p, ++j, ++arg_j) {

        if (*p == '*') {
            --j;
            --arg_j;
        } else if (*p == 'i' || *p == 'l') {
            /* Integer scalars */
            PY_LONG_LONG value = PyLong_AsLongLong(arg_arrays[j]);
            if (PyErr_Occurred()) goto fail;

            if ((*p == 'l' || I_is_int64) && value == (npy_int64)value) {
                arg_list[arg_j] = std::malloc(sizeof(npy_int64));
                *(npy_int64*)arg_list[arg_j] = (npy_int64)value;
            } else if (*p == 'i' && !I_is_int64 && value == (npy_int32)value) {
                arg_list[arg_j] = std::malloc(sizeof(npy_int32));
                *(npy_int32*)arg_list[arg_j] = (npy_int32)value;
            } else {
                PyErr_SetString(PyExc_ValueError, "could not convert integer scalar");
                goto fail;
            }
        } else if (*p == 'B') {
            /* Boolean arrays already cast */
        } else if (*p == 'V') {
            arg_list[arg_j] = allocate_std_vector_typenum(I_typenum);
            if (arg_list[arg_j] == NULL) goto fail;
        } else if (*p == 'W') {
            arg_list[arg_j] = allocate_std_vector_typenum(T_typenum);
            if (arg_list[arg_j] == NULL) goto fail;
        } else if (*p == 'I') {
            arg_list[arg_j] = _cast_arrays(arg_arrays[j], I_typenum, is_output[j]);
            if (arg_list[arg_j] == NULL) goto fail;
            /* Find maximum array size */
            if (PyArray_SIZE((PyArrayObject *)arg_list[arg_j]) > max_array_size) {
                max_array_size = PyArray_SIZE((PyArrayObject *)arg_list[arg_j]);
            }
        } else if (*p == 'T') {
            arg_list[arg_j] = _cast_arrays(arg_arrays[j], T_typenum, is_output[j]);
            if (arg_list[arg_j] == NULL) goto fail;
            /* Find maximum array size */
            if (PyArray_SIZE((PyArrayObject *)arg_list[arg_j]) > max_array_size) {
                max_array_size = PyArray_SIZE((PyArrayObject *)arg_list[arg_j]);
            }
        } else if (*p == 'C') {
          void* csr[5];
          // csr_struct has 5 parts: iiIIT
          /* cast two shape integers */
          for (int i = 0; i < 2; i++, j++) {
            PY_LONG_LONG MorN = PyLong_AsLongLong(arg_arrays[j]);
            if (PyErr_Occurred()) goto fail;

            if (I_is_int64 && MorN == (npy_int64)MorN) {
                csr[i] = std::malloc(sizeof(npy_int64));
                *(npy_int64*)csr[i] = (npy_int64)MorN;
            } else if (!I_is_int64 && MorN == (npy_int32)MorN) {
                csr[i] = std::malloc(sizeof(npy_int32));
                *(npy_int32*)csr[i] = (npy_int32)MorN;
            } else {
                PyErr_SetString(PyExc_ValueError, "could not convert integer scalar");
                goto fail;
            }
          }
          /* cast two index arrays */
          for (int i = 2; i < 4; i++, j++) {
              csr[i] = _cast_arrays(arg_arrays[j], I_typenum, is_output[j]);
              if (csr[i] == NULL) goto fail;

              /* Find maximum array size */
              if (PyArray_SIZE((PyArrayObject *)csr[i]) > max_array_size) {
                  max_array_size = PyArray_SIZE((PyArrayObject *)csr[i]);
              }
          }
          /* cast data array */
          csr[4] = _cast_arrays(arg_arrays[j], T_typenum, is_output[j]);
          if (csr[4] == NULL) goto fail;

          /* Find maximum array size */
          if (PyArray_SIZE((PyArrayObject *)csr[4]) > max_array_size) {
              max_array_size = PyArray_SIZE((PyArrayObject *)csr[4]);
          }
          /* combine into a csr_array struct */
          arg_list[arg_j] = allocate_csr_array(I_typenum, T_typenum, csr);
        }
    }

    /*
     * Call thunk
     */
    if (max_array_size > 100) {
        /* Threshold GIL release: it's not a free operation */
        NPY_BEGIN_THREADS;
    }
    try {
        ret = thunk(I_typenum, T_typenum, arg_list);
        NPY_END_THREADS;
    } catch (const std::bad_alloc &e) {
        NPY_END_THREADS;
        PyErr_SetString(PyExc_MemoryError, e.what());
        goto fail;
    } catch (const std::exception &e) {
        NPY_END_THREADS;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        goto fail;
    }

    /*
     * Generate return value;
     */

    switch (ret_spec) {
    case 'i':
    case 'l':
        return_value = PyLong_FromLongLong(ret);
        break;
    case 'v':
        Py_INCREF(Py_None);
        return_value = Py_None;
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
                        "internal error: invalid return value spec");
    }

    /*
     * Convert any std::vector output arrays to arrays
     */
    if (VW_count > 0) {
        PyObject *new_ret;
        PyObject *old_ret = return_value;
        int pos;

        return_value = NULL;

        new_ret = PyTuple_New(VW_count + (old_ret == Py_None ? 0 : 1));
        if (new_ret == NULL) {
            goto fail;
        }
        if (old_ret != Py_None) {
            PyTuple_SET_ITEM(new_ret, 0, old_ret);
            pos = 1;
        }
        else {
            Py_DECREF(old_ret);
            pos = 0;
        }

        j = 0;
        for (p = spec; *p != '\0'; ++p, ++j) {
            if (*p == '*') {
                --j;
                continue;
            }
            else if (*p == 'V' || *p == 'W') {
                PyObject *arg;
                if (*p == 'V') {
                    arg = array_from_std_vector_and_free(I_typenum, arg_list[j]);
                } else {
                    arg = array_from_std_vector_and_free(T_typenum, arg_list[j]);
                }
                arg_list[j] = NULL;
                if (arg == NULL) {
                    Py_XDECREF(new_ret);
                    goto fail;
                }
                PyTuple_SET_ITEM(new_ret, pos, arg);
                ++pos;
            }
        }

        return_value = new_ret;
    }


fail:
    /*
     * Cleanup
     */
    for (j = 0, p = spec; *p != '\0'; ++p, ++j) {
        if (*p == '*') {
            --j;
            continue;
        }
        if (is_output[j] && arg_arrays[j] != NULL && PyArray_Check(arg_arrays[j])) {
            PyArray_ResolveWritebackIfCopy((PyArrayObject *)arg_arrays[j]);
        }
        Py_XDECREF(arg_arrays[j]);
        if ((*p == 'i' || *p == 'l') && arg_list[j] != NULL) {
            std::free(arg_list[j]);
        }
        else if (*p == 'V' && arg_list[j] != NULL) {
            free_std_vector_typenum(I_typenum, arg_list[j]);
        }
        else if (*p == 'W' && arg_list[j] != NULL) {
            free_std_vector_typenum(T_typenum, arg_list[j]);
        }
        else if (*p == 'C' && arg_list[j] != NULL) {
            free_csr_array(I_typenum, T_typenum, arg_list[j]);
        }
    }
    return return_value;
}


/*
 * Helper functions for handling type discovery and casting
 */

static int _safe_type(const int arg_type, int cur_type, const int n_valid_types,
                      const int *valid_types) {
    int k;
    for (k = 0; k < n_valid_types; ++k) {
        if (PyArray_CanCastSafely(arg_type, valid_types[k])
            && (cur_type == -1 || PyArray_CanCastSafely(cur_type, valid_types[k]))
        ) {
            cur_type = valid_types[k];
            break;
        }
    }
    if (k == n_valid_types) {
        PyErr_SetString(PyExc_ValueError, "unsupported data types in input");
        return -1;
    }
    return cur_type;
}

static void * _cast_arrays(PyObject *pyarg, const int cur_typenum, const int output) {
    const PyObject *arg;
    int atype = PyArray_TYPE((PyArrayObject *)pyarg);
    if (!PyArray_EquivTypenums(atype, cur_typenum)) {
        if (output && !PyArray_CanCastSafely(cur_typenum, atype)) {
            PyErr_SetString(PyExc_ValueError,
                            "Output dtype not compatible with inputs.");
            return 0;
        }
        /* Cast needed and safe */
        arg = c_array_from_object(pyarg, cur_typenum, output);
        Py_DECREF(pyarg);
        if (arg == NULL) return 0;
    }

    return PyArray_DATA((PyArrayObject *)arg);
}

static PyObject *c_array_from_object(PyObject *obj, int typenum, int is_output)
{
    if (!is_output) {
        if (typenum == -1) {
            return PyArray_FROM_OF(obj, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED);
        }
        else {
            return PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_NOTSWAPPED);
        }
    }
    else {
        int flags = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_WRITEABLE|NPY_ARRAY_WRITEBACKIFCOPY|NPY_ARRAY_NOTSWAPPED;
        if (typenum == -1) {
            return PyArray_FROM_OF(obj, flags);
        }
        else {
            return PyArray_FROM_OTF(obj, typenum, flags);
        }
    }
}

/*
 * Helper functions for dealing with csr_array struct instantiation.
            csr_array<npy_int32, ctype> csr_arg = {*(npy_int32*)c[0], *(npy_int32*)c[1], \
                (npy_int32 *)c[2], (npy_int32 *)c[3], (ctype *)c[4]}; \
            return (void*)&csr_arg;                           \
            }
        if (PyArray_EquivTypenums(T_tnum, ntype)) {         \
            csr_array<npy_int64, ctype> csr_arg = {*(npy_int64*)c[0], *(npy_int64*)c[1], \
                (npy_int64 *)c[2], (npy_int64 *)c[3], (ctype *)c[4]}; \
            return (void*)&csr_arg;                           \
            }
            csr_array<npy_int32, ctype> *csr_arg = (csr_array<npy_int32, ctype> *)std::malloc(sizeof(csr_array<npy_int32, ctype>));   \
            csr_array<npy_int64, ctype> *csr_arg = (csr_array<npy_int64, ctype> *)std::malloc(sizeof(csr_array<npy_int64, ctype>));   \
            *******
            void *csr_space = std::malloc(sizeof(csr_array<npy_int32, ctype>));   \
            csr_array<npy_int32, ctype> csr_arg = *(csr_array<npy_int32, ctype> *)csr_space;  \
            *(npy_int32*)csr_arg.n_row = *(npy_int32*)c[0];               \
            *(npy_int32*)csr_arg.n_col = *(npy_int32*)c[1];               \
            csr_arg.indptr = (npy_int32 *)c[2];              \
            csr_arg.indices = (npy_int32 *)c[3];             \
            csr_arg.data = (ctype *)c[4];                    \
            csr_array<npy_int32, ctype>* csr_arg = (csr_array<npy_int32, ctype>*) std::malloc(sizeof(csr_array<npy_int32, ctype>));   \
 */

static void *allocate_csr_array(int I_tnum, int T_tnum, void* c[]) {
#define PROCESS(ntype, ctype)                              \
    if (PyArray_EquivTypenums(I_tnum, NPY_INT32)) {           \
        if (PyArray_EquivTypenums(T_tnum, ntype)) {         \
            void *csr_space = std::malloc(sizeof(csr_array<npy_int32, ctype>));   \
            csr_array<npy_int32, ctype> *csr_arg = (csr_array<npy_int32, ctype> *) csr_space;  \
            *((npy_int32*)&csr_arg->n_row) = *(npy_int32*)c[0];               \
            *((npy_int32*)&csr_arg->n_col) = *(npy_int32*)c[1];               \
            csr_arg->indptr = (npy_int32 *)c[2];              \
            csr_arg->indices = (npy_int32 *)c[3];             \
            csr_arg->data = (ctype *)c[4];                    \
            return (void *)csr_arg;                          \
        }                                                    \
    } else {                                                 \
        if (PyArray_EquivTypenums(T_tnum, ntype)) {         \
            void *csr_space = std::malloc(sizeof(csr_array<npy_int64, ctype>));   \
            csr_array<npy_int64, ctype> *csr_arg = (csr_array<npy_int64, ctype> *) csr_space;  \
            *((npy_int64*)&csr_arg->n_row) = *(npy_int64*)c[0];               \
            *((npy_int64*)&csr_arg->n_col) = *(npy_int64*)c[1];               \
            csr_arg->indptr = (npy_int64 *)c[2];              \
            csr_arg->indices = (npy_int64 *)c[3];             \
            csr_arg->data = (ctype *)c[4];                    \
            return (void *)csr_arg;                          \
        }                                                     \
    }

    try {
        PROCESS(NPY_INT, npy_int)
        //SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)
    } catch (std::exception &e) {
        /* failed */
    }
#undef PROCESS

    PyErr_SetString(PyExc_RuntimeError,
                    "failed to allocate struct csr_array");
    return NULL;
}

static void free_csr_array(int I_tnum, int T_tnum, void *p) {
#define PROCESS(ntype, ctype)                         \
    if (PyArray_EquivTypenums(I_tnum, NPY_INT32)) {           \
        if (PyArray_EquivTypenums(T_tnum, ntype)) {         \
            std::free((void*)&(((csr_array<npy_int32, ctype>*)p)->n_row));\
            std::free((void*)&(((csr_array<npy_int32, ctype>*)p)->n_col));\
            delete ((csr_array<npy_int32, ctype>*)p); return;  \
        }                                                     \
    } else {                                                  \
        if (PyArray_EquivTypenums(T_tnum, ntype)) {         \
            std::free((void*)&(((csr_array<npy_int64, ctype>*)p)->n_row));\
            std::free((void*)&(((csr_array<npy_int64, ctype>*)p)->n_col));\
            delete ((csr_array<npy_int64, ctype>*)p); return;  \
        }                                                     \
    }

    PROCESS(NPY_INT, npy_int)
    //SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)

#undef PROCESS
}

/*
 * Helper functions for dealing with std::vector templated instantiation.
 */

static void *allocate_std_vector_typenum(int typenum)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        return (void*)(new std::vector<ctype>());               \
    }

    try {
        SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)
    } catch (std::exception &e) {
        /* failed */
    }

#undef PROCESS

    PyErr_SetString(PyExc_RuntimeError,
                    "failed to allocate std::vector");
    return NULL;
}

static void free_std_vector_typenum(int typenum, void *p)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        delete ((std::vector<ctype>*)p);                        \
        return;                                                 \
    }

    SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)

#undef PROCESS
}

static PyObject *array_from_std_vector_and_free(int typenum, void *p)
{
#define PROCESS(ntype, ctype)                                   \
    if (PyArray_EquivTypenums(typenum, ntype)) {                \
        std::vector<ctype> *v = (std::vector<ctype>*)p;         \
        npy_intp length = v->size();                            \
        PyObject *obj = PyArray_SimpleNew(1, &length, typenum); \
        if (length > 0) {                                       \
            memcpy(PyArray_DATA((PyArrayObject *)obj), &((*v)[0]), \
                   sizeof(ctype)*length);                       \
        }                                                       \
        delete v;                                               \
        return obj;                                             \
    }

    SPTOOLS_FOR_EACH_DATA_TYPE_CODE(PROCESS)

#undef PROCESS

    PyErr_SetString(PyExc_RuntimeError,
                    "failed to convert std::vector output array");
    return NULL;
}


/*
 * Python module initialization
 */

/* Prevent the name mangling */
extern "C" {
#include "sparsetools_impl.h"
}


static int
_sparsetools_module_exec(PyObject *module)
{
    (void)module;  /* unused */

    if (_import_array() < 0) { return -1; }

    return 0;
}


static PyModuleDef_Slot _sparsetools_slots[] = {
    {Py_mod_exec, (void *)_sparsetools_module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#if PY_VERSION_HEX >= 0x030d00f0  /* Python 3.13+ */
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};


static struct PyModuleDef moduledef = {
    /* m_base     */ PyModuleDef_HEAD_INIT,
    /* m_name     */ "_sparsetools",
    /* m_doc      */ NULL,
    /* m_size     */ 0,
    /* m_methods  */ sparsetools_methods,
    /* m_slots    */ _sparsetools_slots,
    /* m_traverse */ NULL,
    /* m_clear    */ NULL,
    /* m_free     */ NULL
};


PyMODINIT_FUNC
PyInit__sparsetools(void)
{
    return PyModuleDef_Init(&moduledef);
}
