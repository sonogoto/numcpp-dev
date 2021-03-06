#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <set>
#include <random>


// typedef unsigned int uint;

static auto engine = std::default_random_engine(std::random_device()());

static PyObject *
_choice(PyObject *arr, long start, long stop, int n, bool replace) {
    PyObject *indices = NULL, *ret = NULL;
    if (!replace && (stop - start <= n)) {
        indices = PyArray_Arange(static_cast<double>(start), static_cast<double>(stop), 1.0, NPY_INTP);
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        Py_DECREF(indices);
        return ret;
    }
    std::uniform_int_distribution<long> dist(start, stop-1);
    npy_intp dims[] = {n};
    long *idx = new long[n];
    if (replace) {
        for (int i=0; i<n; ++i) {
            idx[i] = dist(engine);
        }
        indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
    }
    else {
        int cnt_sampled = 0;
        std::set<long> indices_sampled;
        long rand_idx = 0;
        while (cnt_sampled < n) {
            rand_idx = dist(engine);
            if (indices_sampled.count(rand_idx) == 0) {
                idx[cnt_sampled++] = rand_idx;
                indices_sampled.insert(rand_idx);
            }
        }
        indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
    }
    Py_DECREF(indices);
    delete[] idx;
    return ret;
}

static PyObject *
_choice(PyObject *arr, PyObject *p, long start, long stop, int n, bool replace) {
    PyObject *indices = PyArray_Arange(static_cast<double>(start), static_cast<double>(stop), 1.0, NPY_INTP);
    PyObject *ret = NULL;
    if (!replace && (stop - start <= n)) {
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        Py_DECREF(indices);
        return ret;
    }
    PyObject *probs =  PyArray_TakeFrom((PyArrayObject *)p, indices, 0, NULL, NPY_RAISE);
    PyObject *probs_cumsum = PyArray_CumSum((PyArrayObject *)probs, 0, NPY_DOUBLE, NULL);
    double *f = (double *)PyArray_GETPTR1((PyArrayObject *)probs_cumsum, stop-start-1);
    std::uniform_real_distribution<double> dist(0.0, *f);
    double *rand_double = new double[n];
    npy_intp dims[] = {n};
    PyObject *indices1 = NULL, *indices2 = NULL, *x = NULL;
    if (replace) {
        for (int i=0; i<n; ++i) {
            rand_double[i] = dist(engine);
        }
        x = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, static_cast<void *>(rand_double));
        indices1 = PyArray_SearchSorted((PyArrayObject *)probs_cumsum, x, NPY_SEARCHRIGHT, NULL);
        indices2 = PyArray_TakeFrom((PyArrayObject *)indices, indices1, 0, NULL, NPY_RAISE);
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices2, 0, NULL, NPY_RAISE);
        Py_DECREF(indices1);
        Py_DECREF(indices2);
        Py_DECREF(x);
    }
    else {
        int cnt_sampled = 0;
        std::set<long> indices_sampled;
        long *rand_idx, *idx = new long[n];;
        PyObject *iter = NULL;
        while (cnt_sampled < n) {
            for (int i=0; i<n-cnt_sampled; ++i) {
                rand_double[i] = dist(engine);
            }
            dims[0] = n - cnt_sampled;
            x = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, static_cast<void *>(rand_double));
            indices1 = PyArray_SearchSorted((PyArrayObject *)probs_cumsum, x, NPY_SEARCHRIGHT, NULL);
            iter = PyArray_IterNew(indices1);
            while (PyArray_ITER_NOTDONE(iter)) {
                rand_idx = (long *)PyArray_ITER_DATA(iter);
                if (indices_sampled.count(*rand_idx) == 0) {
                    indices_sampled.insert(*rand_idx);
                    idx[cnt_sampled++] = start + *rand_idx;
                }
                PyArray_ITER_NEXT(iter);
            }
            Py_DECREF(indices1);
            Py_DECREF(x);
            Py_DECREF(iter);
        }
        dims[0] = n;
        indices2 = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices2, 0, NULL, NPY_RAISE);
        Py_DECREF(indices2);
        delete[] idx;
    }
    Py_DECREF(probs);
    Py_DECREF(probs_cumsum);
    Py_DECREF(indices);
    
    delete[] rand_double;
    return ret;
}


static PyObject *
random_choice(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *x = NULL, *p = NULL, *ret = NULL;
    long start, stop;
    long n;
    bool replace;
    // ????????????????????????????????????numpy.ndarray??????????????????????????????????????????
    if (!PyArg_ParseTuple(args, "lllpO|O:random_choice", &start, &stop, &n, &replace, &x, &p)) {
        PyErr_SetString(PyExc_RuntimeError, "error parsing args");
        return NULL;
    }
    if (stop < start) {
        PyErr_SetString(PyExc_RuntimeError, "illegal args with `stop` less than `start`");
        return NULL;
    }
    if (stop == start) {
        npy_intp dims[] = {0};
        ret = PyArray_ZEROS(1, dims, PyArray_TYPE((PyArrayObject *)x), 0);
        return PyArray_Return((PyArrayObject *)ret);
    }
    if (p == NULL) {
        ret = _choice(x, start, stop, static_cast<int>(n), replace);
    }
    else {
        ret = _choice(x, p, start, stop, static_cast<int>(n), replace);
    }
    return PyArray_Return((PyArrayObject *)ret);
}



static struct PyMethodDef method_def[] = {
    {"random_choice",
    (PyCFunction)random_choice,
    METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "np",
        NULL,
        -1,
        method_def,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_np(void) {
    PyObject *m;
    /* Create the module and add the functions */
    m = PyModule_Create(&module_def);
    import_array();
    if (!m) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "cannot load np module.");
        }
        return NULL;
    }
    return m;
}