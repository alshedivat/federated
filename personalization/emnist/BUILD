load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "federated_emnist",
    srcs = ["federated_emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:training_specs",
        "//personalization/shared:evaluation",
        "//personalization/shared:eval_specs",
        "//personalization/shared:training_specs",
        "//personalization/shared:utils",
        "//personalization/shared/p13n_strategies:finetuning",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)
