load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    srcs_version = "PY3",
)

py_library(
    name = "eval_specs",
    srcs = ["eval_specs.py"],
    srcs_version = "PY3",
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:fed_avg_schedule",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)
