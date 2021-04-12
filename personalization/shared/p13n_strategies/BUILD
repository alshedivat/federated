load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "finetuning",
    srcs = ["finetuning.py"],
    srcs_version = "PY3",
)

py_library(
    name = "maml",
    srcs = ["maml.py"],
    srcs_version = "PY3",
    deps = [
        "//personalization/shared/p13n_strategies:finetuning",
        "//personalization/shared/p13n_strategies:maml_utils",
        "//posterior_averaging/shared:fed_pa_schedule",
    ],
)

py_library(
    name = "maml_utils",
    srcs = ["maml_utils.py"],
    srcs_version = "PY3",
)
