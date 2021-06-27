#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <set>
#include <random>
#include <cstdio>

// typedef unsigned int uint;

static auto engine = std::default_random_engine(std::random_device()());

static PyObject *
_choice(PyObject *arr, long start, long stop, int n, bool replace, std::uniform_int_distribution<long> &dist) {
    npy_intp dims[] = {n};
    if (arr == NULL) {
        return PyArray_Return(NULL);
    }
    PyObject *ret = NULL, *indices = NULL;
    long *idx = new long[n];
    if (replace) {
        for (int i=0; i<n; ++i) {
            idx[i] = dist(engine);
        }
        indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
    }
    else {
        if (stop - start <= n) {
            indices = PyArray_Arange(static_cast<double>(start), static_cast<double>(stop), 1.0, NPY_LONG);
            ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        }
        else {
            int cnt_sampled = 0;
            std::set<long> indices_sampled;
            long rand_idx = 0;
            while (cnt_sampled < n) {
                rand_idx = dist(engine);
                if (indices_sampled.count(rand_idx) == 0)
                    idx[cnt_sampled++] = rand_idx;
            }
            indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
            ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        }
    }
    Py_XDECREF(indices);
    Py_XINCREF(ret);
    delete[] idx;
    return PyArray_Return((PyArrayObject *)ret);
}


static PyObject *
_choice(PyObject *arr, long start, long stop, int n, bool replace, std::discrete_distribution<long> &dist) {
    npy_intp dims[] = {n};
    if (arr == NULL) {
        return PyArray_Return(NULL);
    }
    PyObject *ret = NULL, *indices = NULL;
    long *idx = new long[n];
    if (replace) {
        for (int i=0; i<n; ++i) {
            idx[i] = start + dist(engine);
        }
        indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
        ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
    }
    else {
        if (stop - start <= n) {
            indices = PyArray_Arange(static_cast<double>(start), static_cast<double>(stop), 1.0, NPY_INTP);
            ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        }
        else {
            int cnt_sampled = 0;
            std::set<long> indices_sampled;
            long rand_idx = 0;
            while (cnt_sampled < n) {
                rand_idx = dist(engine);
                if (indices_sampled.count(rand_idx) == 0)
                    idx[cnt_sampled++] = start + rand_idx;
            }
            indices = PyArray_SimpleNewFromData(1, dims, NPY_INTP, static_cast<void *>(idx));
            ret = PyArray_TakeFrom((PyArrayObject *)arr, indices, 0, NULL, NPY_RAISE);
        }
    }
    Py_XDECREF(indices);
    Py_XINCREF(ret);
    delete[] idx;
    return PyArray_Return((PyArrayObject *)ret);
}



static PyObject *
random_choice(PyObject *NPY_UNUSED(ignored), PyObject *args)
{
    PyObject *x = NULL, *p = NULL;
    long start, stop;
    long n;
    bool replace;
    // 在解析参数的时候，需要把numpy.ndarray放在最后一个参数，否则会出错
    if (!PyArg_ParseTuple(args, "lllpO|O:random_choice", &start, &stop, &n, &replace, &x, &p)) {
        return NULL;
    }
    if (p == NULL) {
        std::uniform_int_distribution<long> dist(start, stop-1);
        return _choice(x, start, stop, static_cast<int>(n), replace, dist);
    }
    else {
        PyObject *indices = PyArray_Arange(static_cast<double>(start), static_cast<double>(stop), 1.0, NPY_INTP);
        PyObject *probs =  PyArray_TakeFrom((PyArrayObject *)p, indices, 0, NULL, NPY_RAISE);
        PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)probs);
        Py_INCREF(descr);
        double *ptr = new double[stop-start];
        npy_intp dims[] = {stop-start};
        if (PyArray_AsCArray(&probs, (void *)&ptr, dims, 1, descr) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "error converting 1D array");

            Py_XDECREF(indices);
            Py_XDECREF(probs);
            Py_DECREF(descr);
            delete[] ptr;
            return NULL;

            // delete[] ptr;
            // return PyArray_Return(NULL);
        }
        // for (int i=0; i<stop-start; i++)
        //    printf("%f\n", ptr[i]);
        std::discrete_distribution<long> dist(ptr, ptr+stop-start);
        PyObject *ret = _choice(x, start, stop, static_cast<int>(n), replace, dist);
        PyArray_Free(probs, (void *)ptr);
        // PyArray_Free会清理内存，不再需要手动释放内存，或者清除计数
        // delete[] ptr;
        // Py_DECREF(probs);
        Py_DECREF(indices);
        Py_DECREF(descr);
        return ret;
    }
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
