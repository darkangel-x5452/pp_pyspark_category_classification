from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from utils.model_utils import ModelUtils
import datetime
spark = SparkSession\
    .builder\
    .appName("multilayer_perceptron_classification_example")\
    .master('local[*]')\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "8G") \
    .getOrCreate()


if __name__ == "__main__":
    print(f'{datetime.datetime.now()}, start script')
    mu = ModelUtils()

    print(f'{datetime.datetime.now()}, load data')
    mu.test_df = mu.read_data(data_path='./data/raw/dataset_en_test.json')
    mu.train_df = mu.read_data(data_path='./data/raw/dataset_en_train.json')

    print(f'{datetime.datetime.now()}, join data')
    mu.join_all_data()
    print(f'{datetime.datetime.now()}, filter words')
    mu.filter_words(read_file=False, save_path='')
    print(f'{datetime.datetime.now()}, encode labels')
    mu.label_encode()
    print(f'{datetime.datetime.now()}, encode features')
    mu.text_vectorize()
    print(f'{datetime.datetime.now()}, make model')
    mu.make_model()
    print(f'{datetime.datetime.now()}, do predictions')
    mu.predict_data()
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    # layers = [4, 5, 4, 3]



    # create the trainer and set its parameters
    # trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

    # train the model
    # model = trainer.fit(train)
    #
    # # compute accuracy on the test set
    # result = model.transform(test)
    # predictionAndLabels = result.select("prediction", "label")
    # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    # print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    # # $example off$

    spark.stop()
