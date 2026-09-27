import joblib
import pandas as pd
import numpy as np


# =========================================================
# LOAD MODEL + DATA
# =========================================================

MODEL_PATH = "models/runtime_model.pkl"
DATA_PATH = "data/Convolution.csv"

model = joblib.load(MODEL_PATH)
gpu_data = pd.read_csv(DATA_PATH)


# =========================================================
# GPU OPTIONS
# =========================================================

GPUS = sorted(gpu_data["GPU"].unique())

GPU_SPECS = (
    gpu_data
    .drop_duplicates("GPU")
    .set_index("GPU")
)


# =========================================================
# CLI DISPLAY
# =========================================================

def header():
    print()
    print("=" * 60)
    print("     COST-CARBON DRIVEN TRAINING TIME PREDICTION")
    print("=" * 60)
    print()


def choose_gpu():

    print("Select GPU")
    print("-" * 40)

    for i, gpu in enumerate(GPUS, start=1):
        print(f"{i}. {gpu}")

    while True:
        try:
            choice = int(input("\nGPU: "))

            if 1 <= choice <= len(GPUS):
                return GPUS[choice - 1]

            print("Please enter a valid option.")

        except ValueError:
            print("Please enter a number.")


def choose_padding():

    print("\nPadding")
    print("-" * 40)
    print("1. same")
    print("2. valid")

    while True:
        choice = input("Choice: ").strip()

        if choice == "1":
            return "same"

        if choice == "2":
            return "valid"

        print("Please choose 1 or 2.")


def choose_activation():

    print("\nActivation Function")
    print("-" * 40)
    print("0. 0")
    print("1. 1")
    print("2. 2")
    print("3. 3")

    while True:
        try:
            choice = int(input("Choice: "))

            if choice in [0, 1, 2, 3]:
                return choice

        except ValueError:
            pass

        print("Please enter 0, 1, 2, or 3.")


def choose_optimizer():

    print("\nOptimizer")
    print("-" * 40)

    for i in range(6):
        print(f"{i}. {i}")

    while True:
        try:
            choice = int(input("Choice: "))

            if choice in range(6):
                return choice

        except ValueError:
            pass

        print("Please enter a value from 0 to 5.")


def choose_bias():

    print("\nUse Bias")
    print("-" * 40)
    print("1. Yes")
    print("2. No")

    while True:
        choice = input("Choice: ").strip()

        if choice == "1":
            return True

        if choice == "2":
            return False

        print("Please choose 1 or 2.")


# =========================================================
# MAIN
# =========================================================

def main():

    header()

    # -----------------------------------------------------
    # GPU
    # -----------------------------------------------------

    gpu = choose_gpu()

    specs = GPU_SPECS.loc[gpu]

    print(f"\nSelected GPU: {gpu}")

    # -----------------------------------------------------
    # WORKLOAD INPUT
    # -----------------------------------------------------

    print("\nEnter Training Configuration")
    print("-" * 40)

    batchsize = int(input("Batch Size            : "))
    matsize = int(input("Matrix Size           : "))
    kernelsize = int(input("Kernel Size           : "))
    channels_in = int(input("Input Channels        : "))
    channels_out = int(input("Output Channels       : "))
    strides = int(input("Strides               : "))

    padding = choose_padding()
    activation_fct = choose_activation()
    use_bias = choose_bias()
    optimizer = choose_optimizer()

    # -----------------------------------------------------
    # WORKLOAD FEATURES
    # -----------------------------------------------------

    parameter_count = (
        kernelsize ** 2
        * channels_in
        * channels_out
    )

    output_spatial_size = matsize ** 2

    estimated_macs = (
        batchsize
        * output_spatial_size
        * channels_in
        * channels_out
        * kernelsize ** 2
    )

    estimated_flops = 2 * estimated_macs

    # -----------------------------------------------------
    # CREATE MODEL INPUT
    # -----------------------------------------------------

    data = pd.DataFrame([{

        "batchsize": batchsize,
        "matsize": matsize,
        "kernelsize": kernelsize,
        "channels_in": channels_in,
        "channels_out": channels_out,
        "strides": strides,

        "padding": padding,
        "activation_fct": activation_fct,
        "use_bias": use_bias,
        "optimizer": optimizer,

        "GPU": gpu,
        "Provisioning": specs["Provisioning"],

        "Base Clock (MHz)": specs["Base Clock (MHz)"],
        "Boost Clock (MHz)": specs["Boost Clock (MHz)"],
        "Memory Clock (MHz)": specs["Memory Clock (MHz)"],
        "Memory (GB)": specs["Memory (GB)"],
        "Memory Type": specs["Memory Type"],
        "Memory Bus (bit)": specs["Memory Bus (bit)"],
        "GPU Memory Bandwidth (GB/s)": specs["GPU Memory Bandwidth (GB/s)"],
        "Bus": specs["Bus"],

        "Cores": specs["Cores"],
        "TMUs": specs["TMUs"],
        "ROPs": specs["ROPs"],
        "SM": specs["SM"],
        "TC": specs["TC"],
        "RT": specs["RT"],
        "PR": specs["PR"],
        "TR": specs["TR"],
        "FP16": specs["FP16"],
        "FP32": specs["FP32"],
        "FP64": specs["FP64"],

        # Workload-aware features
        "parameter_count": parameter_count,
        "output_spatial_size": output_spatial_size,
        "estimated_macs": estimated_macs,
        "estimated_flops": estimated_flops

    }])

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    print("\n")
    print("-" * 60)
    print("Predicting execution time...")
    print("-" * 60)

    predicted_runtime = model.predict(data)[0]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("                 PREDICTION RESULT")
    print("=" * 60)

    print()
    print(f"GPU                    : {gpu}")
    print(f"Batch Size             : {batchsize}")
    print(f"Matrix Size            : {matsize}")
    print(f"Kernel Size            : {kernelsize}")
    print(f"Input Channels         : {channels_in}")
    print(f"Output Channels        : {channels_out}")

    print()
    print("-" * 60)

    print(
        f"Predicted Execution Time : "
        f"{predicted_runtime:.4f} seconds"
    )

    print("-" * 60)

    print()
    print("Runtime prediction completed.")
    print()


if __name__ == "__main__":
    main()