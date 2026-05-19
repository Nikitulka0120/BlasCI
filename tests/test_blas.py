import ctypes
import ctypes.util
import numpy as np
import pytest


# =========================
# LOAD OPENBLAS (SYSTEM)
# =========================
def getLib():
    libname = ctypes.util.find_library("openblas")

    if not libname:
        pytest.fail("OpenBLAS не найден в системе (libopenblas)")

    try:
        return ctypes.CDLL(libname)
    except Exception as e:
        pytest.fail(f"Ошибка загрузки OpenBLAS: {e}")


blas = getLib()


# =========================
# FUNCTION DEFINITIONS
# =========================

blasFunc = {
    "cblas_dnrm2": (ctypes.c_double, ctypes.c_double),
    "cblas_dasum": (ctypes.c_double, ctypes.c_double),
    "cblas_snrm2": (ctypes.c_float, ctypes.c_float),
    "cblas_sasum": (ctypes.c_float, ctypes.c_float),
}

complexBlas = {
    "cblas_cdotu_sub": (ctypes.c_float, ctypes.c_float),
    "cblas_cdotc_sub": (ctypes.c_float, ctypes.c_float),
    "cblas_zdotu_sub": (ctypes.c_double, ctypes.c_double),
    "cblas_zdotc_sub": (ctypes.c_double, ctypes.c_double),
}

dotBlas = {
    "cblas_sdot": (ctypes.c_float, ctypes.c_float),
    "cblas_ddot": (ctypes.c_double, ctypes.c_double),
    "cblas_dsdot": (ctypes.c_double, ctypes.c_float),
}


# =========================
# CONFIG FUNCTION
# =========================
def configBlas(name, restype, argtypes):
    try:
        func = getattr(blas, name)
        func.restype = restype
        func.argtypes = argtypes
        return func
    except AttributeError:
        pytest.fail(f"Функция {name} отсутствует в OpenBLAS")


# =========================
# TEST 1: NORM / SUM
# =========================
@pytest.mark.parametrize("func_name", blasFunc.keys())
def test_blas_basic(func_name):
    res_t, arg_t = blasFunc[func_name]

    blas_func = configBlas(
        func_name,
        res_t,
        [ctypes.c_int, ctypes.POINTER(arg_t), ctypes.c_int]
    )

    np_type = np.float32 if arg_t == ctypes.c_float else np.float64

    x = np.array([3.0, 4.0], dtype=np_type)

    result = blas_func(
        2,
        x.ctypes.data_as(ctypes.POINTER(arg_t)),
        1
    )

    assert result is not None
    assert result >= 0


# =========================
# TEST 2: DOT PRODUCTS
# =========================
@pytest.mark.parametrize("func_name", dotBlas.keys())
def test_dot(func_name):
    res_t, arg_t = dotBlas[func_name]

    blas_func = configBlas(
        func_name,
        res_t,
        [ctypes.c_int,
         ctypes.POINTER(arg_t), ctypes.c_int,
         ctypes.POINTER(arg_t), ctypes.c_int]
    )

    np_type = np.float32 if arg_t == ctypes.c_float else np.float64

    x = np.array([1.0, 2.0], dtype=np_type)
    y = np.array([3.0, 4.0], dtype=np_type)

    result = blas_func(
        2,
        x.ctypes.data_as(ctypes.POINTER(arg_t)), 1,
        y.ctypes.data_as(ctypes.POINTER(arg_t)), 1
    )

    assert result is not None


# =========================
# TEST 3: COMPLEX DOT
# =========================
@pytest.mark.parametrize("func_name", complexBlas.keys())
def test_complex(func_name):
    base_arg_t, base_res_t = complexBlas[func_name]

    blas_func = configBlas(
        func_name,
        None,
        [ctypes.c_int,
         ctypes.POINTER(base_arg_t), ctypes.c_int,
         ctypes.POINTER(base_arg_t), ctypes.c_int,
         ctypes.POINTER(base_res_t)]
    )

    np_type = np.float32 if base_arg_t == ctypes.c_float else np.float64

    x = np.array([1.0, 2.0], dtype=np_type)
    y = np.array([1.0, 1.0], dtype=np_type)

    result_buffer = np.zeros(2, dtype=np_type)

    blas_func(
        1,
        x.ctypes.data_as(ctypes.POINTER(base_arg_t)), 1,
        y.ctypes.data_as(ctypes.POINTER(base_arg_t)), 1,
        result_buffer.ctypes.data_as(ctypes.POINTER(base_res_t))
    )

    assert np.any(result_buffer != 0)


# =========================
# TEST 4: SPECIAL CASE (SDSDOT)
# =========================
def test_sdsdot():
    blas_func = configBlas(
        "cblas_sdsdot",
        ctypes.c_float,
        [ctypes.c_int, ctypes.c_float,
         ctypes.POINTER(ctypes.c_float), ctypes.c_int,
         ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    )

    x = np.array([1.0], dtype=np.float32)
    y = np.array([1.0], dtype=np.float32)

    result = blas_func(
        1,
        10.0,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1
    )

    assert result is not None