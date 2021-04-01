load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//google/personalization/example:__subpackages__"])

licenses(["notice"])

py_library(
    name = "p13n_utils",
    srcs = ["p13n_utils.py"],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "p13n_utils_test",
    srcs = ["p13n_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":p13n_utils"],
)

py_binary(
    name = "emnist_p13n_main",
    srcs = ["emnist_p13n_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist_p13n_main_lib"],
)

py_library(
    name = "emnist_p13n_main_lib",
    srcs = ["emnist_p13n_main.py"],
    srcs_version = "PY3",
    deps = [":p13n_utils"],
)
