#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <set>
#include <random>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <functional>


static auto engine = std::default_random_engine(std::random_device()());

static int
_choice(long start, long stop, int n, bool replace, long *out) {
    if (stop <= start)
        return 0;
    if (!replace && (stop - start <= n)) {
        for (int i=0; i<stop-start; ++i)
            out[i] = start + i;
        return (int)(stop-start);
    }
    std::uniform_int_distribution<long> dist(start, stop-1);
    if (replace) {
        for (int i=0; i<n; ++i) {
            out[i] = dist(engine);
        }
    }
    else {
        int cnt_sampled = 0;
        std::set<long> indices_sampled;
        long rand_idx = 0;
        while (cnt_sampled < n) {
            rand_idx = dist(engine);
            if (indices_sampled.count(rand_idx) == 0) {
                out[cnt_sampled++] = rand_idx;
                indices_sampled.insert(rand_idx);
            }
        }
    }
    return n;
}

static void cumulative_sum(PyArrayObject *arr, long start, long stop, double *out) {
    for (int i=0; i<stop-start; ++i) {
        double *x = (double *)PyArray_GETPTR1(arr, start+i);
        if (0 == i)
            out[i] = *x;
        else
            out[i] = out[i-1] + *x;
    }
}

static int
_choice(PyArrayObject *p, long start, long stop, int n, bool replace, long *out) {
    if (stop <= start)
        return 0;
    if (!replace && (stop - start <= n)) {
        for (int i=0; i<stop-start; ++i)
            out[i] = start + i;
        return (int)(stop-start);
    }
    double *cum_probs = new double[stop-start];
    cumulative_sum(p, start, stop, cum_probs);
    std::uniform_real_distribution<double> dist(0.0, cum_probs[stop-start-1]);
    double rand_x = 0.0;
    if (replace) {
        for (int i=0; i<n; ++i) {
            rand_x = dist(engine);
            out[i] = start + (std::upper_bound(cum_probs, cum_probs+stop-start, rand_x)-cum_probs);
        }
    }
    else {
        int cnt_sampled = 0;
        std::set<long> indices_sampled;
        long rand_idx = 0;
        while (cnt_sampled < n) {
            rand_x = dist(engine);
            // search sorted
            rand_idx = start + (std::upper_bound(cum_probs, cum_probs+stop-start, rand_x)-cum_probs);
            if (indices_sampled.count(rand_idx) == 0) {
                out[cnt_sampled++] = rand_idx;
                indices_sampled.insert(rand_idx);
            }
        }
    }
    delete[] cum_probs;
    return n;
}

static PyObject *
_sample_neighbors_randomly(PyObject *ids0, PyObject *ids1,
                           PyObject *nbr_ids, PyObject *nbr_ptrs,
                           PyObject *edge_ids, PyObject *edge_probs,
                           int n, bool replace, int num_threads) {
    int cnt = (int)PyArray_SIZE((PyArrayObject *)ids0);
    PyObject *ptr_start = PyArray_TakeFrom((PyArrayObject *)nbr_ptrs, ids0, 0, NULL, NPY_RAISE);
    PyObject *ptr_end = PyArray_TakeFrom((PyArrayObject *)nbr_ptrs, ids1, 0, NULL, NPY_RAISE);
    long *indices = new long[cnt*n];
    int *offset = new int[cnt];
    if (num_threads > 0) {
        omp_set_dynamic((int)true);
        omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for shared(indices, offset)
    for (int i=0; i<cnt; ++i) {
        long start, end;
        if (PyArray_TYPE((PyArrayObject *)ptr_start) == NPY_INT) {
            int *p1 = (int *)PyArray_GETPTR1((PyArrayObject *)ptr_start, i);
            int *p2 = (int *)PyArray_GETPTR1((PyArrayObject *)ptr_end, i);
            start = (long)(*p1);
            end = (long)(*p2);
        }
        else {
            start = *(long *)PyArray_GETPTR1((PyArrayObject *)ptr_start, i);
            end = *(long *)PyArray_GETPTR1((PyArrayObject *)ptr_end, i);
        }
        if (edge_probs == NULL)
            offset[i] = _choice(start, end, n, replace, indices+i*n);
        else
            offset[i] = _choice((PyArrayObject *)edge_probs, start, end, n, replace, indices+i*n);
    }
    Py_DECREF(ptr_start);
    Py_DECREF(ptr_end);

    PyObject *rets = NULL;
    if (edge_ids == NULL)
        rets = PyTuple_New(cnt+1);
    else
        rets = PyTuple_New(cnt*2+1);
    npy_intp dims[1];
    for (int i=0; i<cnt; ++i) {
        dims[0] = (npy_intp)offset[i];
        PyObject *idx = PyArray_SimpleNewFromData(1, dims, NPY_INTP, (void *)(indices+i*n));
        PyObject *ret = PyArray_TakeFrom((PyArrayObject *)nbr_ids, idx, 0, NULL, NPY_RAISE);
        PyTuple_SET_ITEM(rets, i, ret);
        if (edge_ids != NULL) {
            PyObject *eids = PyArray_TakeFrom((PyArrayObject *)edge_ids, idx, 0, NULL, NPY_RAISE);
            PyTuple_SET_ITEM(rets, i+cnt, eids);
        }
        Py_DECREF(idx);
    }

    dims[0] = cnt;
    PyObject *offset_arr = PyArray_SimpleNewFromData(1, dims, NPY_INT, (void *)offset);
    // set the NPY_ARRAY_OWNDATA flag
    // to free memory as soon as the ndarray is deallocated
    PyArray_ENABLEFLAGS((PyArrayObject *)offset_arr, NPY_ARRAY_OWNDATA);
    if (edge_ids == NULL)
        PyTuple_SET_ITEM(rets, cnt, offset_arr);
    else
        PyTuple_SET_ITEM(rets, cnt*2, offset_arr);

    delete[] indices;
    // the memory of offset will be freed
    // when offset_arr is deallocated
    // delete[] offset;
    return rets;
}


static PyObject *
sample_neighbors_randomly(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwargs)
{
    PyObject *ids0 = NULL, *ids1 = NULL, *nbr_ids = NULL, *nbr_ptrs = NULL, *edge_ids = NULL, *edge_probs = NULL;
    long n, num_threads = 0;
    bool replace;
    static char *kwlist[] = {"n", "replace", "ids0", "ids1", "nbr_ids", "nbr_ptrs",
                            "num_threads", "edge_ids", "edge_probs", NULL};
    // 在解析参数的时候，需要把numpy.ndarray放在最后一个参数，否则会出错
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lpOOOO|lOO:sample_neighbors", kwlist,
                                    &n, &replace, &ids0, &ids1, &nbr_ids, &nbr_ptrs,
                                    &num_threads, &edge_ids, &edge_probs)) {
        PyErr_SetString(PyExc_RuntimeError, "error parsing args");
        return NULL;
    }
    PyObject *ret = _sample_neighbors_randomly(ids0, ids1, nbr_ids, nbr_ptrs, edge_ids, edge_probs, (int)n, replace, (int)num_threads);
    return ret;
}


static int _topk(PyArrayObject *p, long start, long stop, int k, long *out) {
    if (stop - start <= k) {
        for (int i=0; i<stop-start; ++i)
            out[i] = start + i;
        return (int)(stop-start);
    }
    std::vector<std::pair<double, long> > vec;
    double *x;
    for (int i=0; i<k; ++i) {
        x = (double *)PyArray_GETPTR1(p, start+i);
        vec.push_back(std::pair<double, long>(*x, start+i));
    }
    std::make_heap(vec.begin(),vec.begin()+k, std::greater<std::pair<double, long> >());
    for (int i=k; i<stop-start; ++i) {
        x = (double *)PyArray_GETPTR1(p, start+i);
        if (*x > vec.begin()->first) {
            *vec.begin() = std::pair<double, long>(*x, start+i);
            std::make_heap(vec.begin(),vec.begin()+k, std::greater<std::pair<double, long> >());
        }
    }
    for (int i=0; i<k; ++i) {
        out[i] = (vec.begin()+i)->second;
    }
    return k;
}


static PyObject *
_sample_topk_neighbors(PyObject *ids0, PyObject *ids1,
                       PyObject *nbr_ids, PyObject *nbr_ptrs,
                       PyObject *edge_ids, PyObject *edge_probs,
                       int k, int num_threads) {
    int cnt = (int)PyArray_SIZE((PyArrayObject *)ids0);
    PyObject *ptr_start = PyArray_TakeFrom((PyArrayObject *)nbr_ptrs, ids0, 0, NULL, NPY_RAISE);
    PyObject *ptr_end = PyArray_TakeFrom((PyArrayObject *)nbr_ptrs, ids1, 0, NULL, NPY_RAISE);
    long *indices = new long[cnt*k];
    int *offset = new int[cnt];
    if (num_threads > 0) {
        omp_set_dynamic((int)true);
        omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for shared(indices, offset)
    for (int i=0; i<cnt; ++i) {
        long start, end;
        if (PyArray_TYPE((PyArrayObject *)ptr_start) == NPY_INT) {
            int *p1 = (int *)PyArray_GETPTR1((PyArrayObject *)ptr_start, i);
            int *p2 = (int *)PyArray_GETPTR1((PyArrayObject *)ptr_end, i);
            start = (long)(*p1);
            end = (long)(*p2);
        }
        else {
            start = *(long *)PyArray_GETPTR1((PyArrayObject *)ptr_start, i);
            end = *(long *)PyArray_GETPTR1((PyArrayObject *)ptr_end, i);
        }
        offset[i] = _topk((PyArrayObject *)edge_probs, start, end, k, indices+i*k);
    }
    Py_DECREF(ptr_start);
    Py_DECREF(ptr_end);

    PyObject *rets = NULL;
    if (edge_ids == NULL)
        rets = PyTuple_New(cnt+1);
    else
        rets = PyTuple_New(cnt*2+1);
    npy_intp dims[1];
    for (int i=0; i<cnt; ++i) {
        dims[0] = (npy_intp)offset[i];
        PyObject *idx = PyArray_SimpleNewFromData(1, dims, NPY_INTP, (void *)(indices+i*k));
        PyObject *ret = PyArray_TakeFrom((PyArrayObject *)nbr_ids, idx, 0, NULL, NPY_RAISE);
        PyTuple_SET_ITEM(rets, i, ret);
        if (edge_ids != NULL) {
            PyObject *eids = PyArray_TakeFrom((PyArrayObject *)edge_ids, idx, 0, NULL, NPY_RAISE);
            PyTuple_SET_ITEM(rets, i+cnt, eids);
        }
        Py_DECREF(idx);
    }

    dims[0] = cnt;
    PyObject *offset_arr = PyArray_SimpleNewFromData(1, dims, NPY_INT, (void *)offset);
    // set the NPY_ARRAY_OWNDATA flag
    // to free memory as soon as the ndarray is deallocated
    PyArray_ENABLEFLAGS((PyArrayObject *)offset_arr, NPY_ARRAY_OWNDATA);
    if (edge_ids == NULL)
        PyTuple_SET_ITEM(rets, cnt, offset_arr);
    else
        PyTuple_SET_ITEM(rets, cnt*2, offset_arr);

    delete[] indices;
    // the memory of offset will be freed
    // when offset_arr is deallocated
    // delete[] offset;
    return rets;
}


static PyObject *
sample_topk_neighbors(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwargs)
{
    PyObject *ids0 = NULL, *ids1 = NULL, *nbr_ids = NULL, *nbr_ptrs = NULL, *edge_ids = NULL, *edge_probs = NULL;
    long k, num_threads = 0;
    static char *kwlist[] = {"k", "ids0", "ids1", "nbr_ids", "nbr_ptrs", "edge_probs",
                            "num_threads", "edge_ids", NULL};
    // 在解析参数的时候，需要把numpy.ndarray放在最后一个参数，否则会出错
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOOOOO|lO:sample_neighbors", kwlist,
                                    &k, &ids0, &ids1, &nbr_ids, &nbr_ptrs, &edge_probs,
                                    &num_threads, &edge_ids)) {
        PyErr_SetString(PyExc_RuntimeError, "error parsing args");
        return NULL;
    }
    PyObject *ret = _sample_topk_neighbors(ids0, ids1, nbr_ids, nbr_ptrs, edge_ids, edge_probs, (int)k, (int)num_threads);
    return ret;
}

static struct PyMethodDef method_def[] = {
    {"sample_neighbors_randomly",
    (PyCFunction)sample_neighbors_randomly,
    METH_VARARGS | METH_KEYWORDS, NULL},
    {"sample_topk_neighbors",
    (PyCFunction)sample_topk_neighbors,
    METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}                /* sentinel */
};

static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "sampler",
        NULL,
        -1,
        method_def,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit_sampler(void) {
    PyObject *m;
    /* Create the module and add the functions */
    m = PyModule_Create(&module_def);
    import_array();
    if (!m) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "cannot load sampler module.");
        }
        return NULL;
    }
    return m;
}
