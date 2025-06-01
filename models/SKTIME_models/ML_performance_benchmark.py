import time
import model_JC_test
import file_verification
import data_preprocessing


def measure_model_execution_time():
    start_time = time.time()
    scores = model_JC_test.main()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time, scores


def measure_file_verification_time():
    start_time = time.time()

    file_verification.main()

    end_time = time.time()
    total_time = end_time - start_time
    return total_time


def measure_data_processing_time():
    start_time = time.time()
    data_preprocessing.main()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time


def benchmark_scores(scores):
    metrics = [
        'Mean Absolute Error',
        'Root Mean Squared Error',
        'Mean Absolute Percentage Error',
        'Accuracy',
        'F1-Score'
    ]

    avg_scores = {}
    for metric in metrics:
        # compute the average for that metric, ignoring NaN values
        avg_value = scores[metric].mean(skipna=True)
        avg_scores[metric] = avg_value

    return avg_scores


def main():
    # Example of a single benchmark run
    ml_time, scores = measure_model_execution_time()
    fv_time = measure_file_verification_time()
    dpp_time = measure_data_processing_time()
    bench_scores = benchmark_scores(scores)
    print(f"Total execution time for model: {ml_time:.2f} seconds")
    print(f"Total execution time for file verification: {fv_time:.2f} seconds")
    print(f"Total execution time for data preprocessing: {dpp_time:.2f} seconds")
    print(f"Determined accuracy benchmark as {bench_scores}")


if __name__ == "__main__":
    main()
