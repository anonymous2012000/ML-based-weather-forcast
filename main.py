# Runs this whole project sequentially. This file should be called upon exectuing this work
import file_verification
import data_preprocessing
from models.SKTIME_models import model_JC_test


def main():
    file_verification.main()
    data_preprocessing.main()
    model_JC_test.main()


if __name__ == '__main__':
    main()
