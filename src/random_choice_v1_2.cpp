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
#include <math.h>

static auto engine = std::default_random_engine(std::random_device()());
static std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);


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

inline static void _get_start_end(PyObject *start_arr, PyObject *end_arr,
                          npy_intp start_idx, npy_intp end_idx,
                          long *start, long *end) {
    if (PyArray_TYPE((PyArrayObject *)start_arr) == NPY_INT) {
        int *p1 = (int *)PyArray_GETPTR1((PyArrayObject *)start_arr, start_idx);
        int *p2 = (int *)PyArray_GETPTR1((PyArrayObject *)end_arr, end_idx);
        *start = (long)(*p1);
        *end = (long)(*p2);
    }
    else {
        *start = *(long *)PyArray_GETPTR1((PyArrayObject *)start_arr, start_idx);
        *end = *(long *)PyArray_GETPTR1((PyArrayObject *)end_arr, end_idx);
    }
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
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for shared(indices, offset)
    for (int i=0; i<cnt; ++i) {
        long start, end;
        _get_start_end(ptr_start, ptr_end, i, i, &start, &end);
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
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
    }
    #pragma omp parallel for shared(indices, offset)
    for (int i=0; i<cnt; ++i) {
        long start, end;
        _get_start_end(ptr_start, ptr_end, i, i, &start, &end);
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


static bool _discard_node(double freq, double freq_th) {
    return uniform_dist(engine) > (std::sqrt(freq_th/freq) + freq_th/freq);
}


static PyObject *
_random_walk(PyObject *ids, PyObject *nbr_ids, PyObject *nbr_ptrs,
             PyObject *nbr_freqs, int walk_length, double freq_th) {
    int cnt = (int)PyArray_SIZE((PyArrayObject *)ids);
    PyObject *rets = PyTuple_New(cnt);

    for (int i=0; i<cnt; ++i) {
        int j = 1, actual_len = 1, continue_flag;
        PyObject *path = PyList_New(walk_length+1);
        long start, end, nid, nbr_ptr[1];
        nid = *(long *)PyArray_GETPTR1((PyArrayObject *)ids, i);
        PyList_SetItem(path, 0, PyLong_FromLong(nid));
        for(; j<=walk_length; ++j) {
            _get_start_end(nbr_ptrs, nbr_ptrs, nid, nid+1, &start, &end);
            continue_flag = _choice(start, end, 1, true, nbr_ptr);
            if (continue_flag) {
                if (PyArray_TYPE((PyArrayObject *)nbr_ids) == NPY_INT) {
                    int *p1 = (int *)PyArray_GETPTR1((PyArrayObject *)nbr_ids, nbr_ptr[0]);
                    nid = (long)(*p1);
                }
                else
                    nid = *(long *)PyArray_GETPTR1((PyArrayObject *)nbr_ids, nbr_ptr[0]);
                // discard frequent nodes
                if (nbr_freqs != NULL &&
                    _discard_node(*(double *)PyArray_GETPTR1((PyArrayObject *)nbr_freqs, nbr_ptr[0]), freq_th))
                    continue;
                PyList_SetItem(path, actual_len++, PyLong_FromLong(nid));
//                ++actual_len;
            }
            else
                break;
        }
        PyTuple_SetItem(rets, i, PyList_GetSlice(path, 0, actual_len));
        Py_DECREF(path);
    }

    return rets;
}


static PyObject *
random_walk(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwargs)
{
    PyObject *ids = NULL, *nbr_ids = NULL, *nbr_ptrs = NULL, *nbr_freqs = NULL;
    long walk_length;
    double freq_th = 1.0;
    static char *kwlist[] = {"walk_length", "ids", "nbr_ids", "nbr_ptrs", "nbr_freqs", "freq_th", NULL};
    // 在解析参数的时候，需要把numpy.ndarray放在最后一个参数，否则会出错
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOOO|Od:random_walk", kwlist,
                                    &walk_length, &ids, &nbr_ids, &nbr_ptrs, &nbr_freqs, &freq_th)) {
        PyErr_SetString(PyExc_RuntimeError, "error parsing args");
        return NULL;
    }
    PyObject *ret = _random_walk(ids, nbr_ids, nbr_ptrs, nbr_freqs, (int)walk_length, freq_th);
    return ret;
}


static PyObject *
_node2vec_walk(PyObject *ids, PyObject *nbr_ids, PyObject *nbr_ptrs, PyObject *nbr_freqs,
             double p, double q, int walk_length, double freq_th) {
    int cnt = (int)PyArray_SIZE((PyArrayObject *)ids);
    PyObject *rets = PyTuple_New(cnt);
    double transfer_probs[3] = {1.0/p, 1.0, 1.0/q};
    npy_intp dims[1] = {3};
    PyObject *transfer_probs_arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *)(transfer_probs));

    long *path = new long[walk_length+1];
    long start, end, nid, nbr_ptr[1], next_step[1];
    for (int i=0; i<cnt; ++i) {
        int j = 1, actual_len = 1, continue_flag;
        PyObject *actual_path = PyList_New(walk_length+1);
        nid = *(long *)PyArray_GETPTR1((PyArrayObject *)ids, i);
        PyList_SetItem(actual_path, 0, PyLong_FromLong(nid));
        path[0] = nid;
        for(; j<=walk_length; ++j) {
            if (j <= 1) {
                _get_start_end(nbr_ptrs, nbr_ptrs, path[0], path[0]+1, &start, &end);
                continue_flag = _choice(start, end, 1, true, nbr_ptr);
            }
            else {
                _choice((PyArrayObject *)transfer_probs_arr, 0L, 3L, 1, true, next_step);
                switch (next_step[0]) {
                    case 0L:
                        nid = path[j-2];
                        continue_flag = 1;
                        break;
                    case 1L:
                        _get_start_end(nbr_ptrs, nbr_ptrs, path[j-2], path[j-2]+1, &start, &end);
                        continue_flag = _choice(start, end, 1, true, nbr_ptr);
                        break;
                    case 2L:
                        _get_start_end(nbr_ptrs, nbr_ptrs, path[j-1], path[j-1]+1, &start, &end);
                        continue_flag = _choice(start, end, 1, true, nbr_ptr);
                        break;
                    default:
                        continue_flag = 0;
                }
            }

            if (continue_flag) {
                if (j <= 1 || next_step[0] != 0L) {
                    if (PyArray_TYPE((PyArrayObject *)nbr_ids) == NPY_INT) {
                        int *p1 = (int *)PyArray_GETPTR1((PyArrayObject *)nbr_ids, nbr_ptr[0]);
                        nid = (long)(*p1);
                    }
                    else
                        nid = *(long *)PyArray_GETPTR1((PyArrayObject *)nbr_ids, nbr_ptr[0]);
                }
                path[j] = nid;
                // discard frequent nodes
                if (nbr_freqs != NULL &&
                    _discard_node(*(double *)PyArray_GETPTR1((PyArrayObject *)nbr_freqs, nbr_ptr[0]), freq_th))
                    continue;
                PyList_SetItem(actual_path, actual_len++, PyLong_FromLong(nid));
            }
            else
                break;
        }
        PyTuple_SetItem(rets, i, PyList_GetSlice(actual_path, 0, actual_len));
        Py_DECREF(actual_path);
    }
    delete[] path;
    Py_DECREF(transfer_probs_arr);

    return rets;
}


static PyObject *
node2vec_walk(PyObject *NPY_UNUSED(ignored), PyObject *args, PyObject *kwargs)
{
    PyObject *ids = NULL, *nbr_ids = NULL, *nbr_ptrs = NULL, *nbr_freqs = NULL;
    double p, q, freq_th = 1.0;
    long walk_length;
    static char *kwlist[] = {"walk_length", "ids", "nbr_ids", "nbr_ptrs", "p", "q",
                            "nbr_freqs", "freq_th", NULL};
    // 在解析参数的时候，需要把numpy.ndarray放在最后一个参数，否则会出错
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOOOdd|Od:node2vec_walk", kwlist,
                                    &walk_length, &ids, &nbr_ids, &nbr_ptrs, &p, &q, &nbr_freqs, &freq_th)) {
        PyErr_SetString(PyExc_RuntimeError, "error parsing args");
        return NULL;
    }
    PyObject *ret = _node2vec_walk(ids, nbr_ids, nbr_ptrs, nbr_freqs, p, q, (int)walk_length, freq_th);
    return ret;
}


static struct PyMethodDef method_def[] = {
    {"sample_neighbors_randomly",
    (PyCFunction)sample_neighbors_randomly,
    METH_VARARGS | METH_KEYWORDS, NULL},
    {"sample_topk_neighbors",
    (PyCFunction)sample_topk_neighbors,
    METH_VARARGS | METH_KEYWORDS, NULL},
    {"random_walk",
    (PyCFunction)random_walk,
    METH_VARARGS | METH_KEYWORDS, NULL},
    {"node2vec_walk",
    (PyCFunction)node2vec_walk,
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
