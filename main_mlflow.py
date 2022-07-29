import json
import os

import pandas as pd
import datetime
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, IDF
from pyspark.sql import SparkSession
import pyspark.sql.functions as sqlf
import nltk
from nltk.corpus import stopwords
import sklearn
import mlflow
import hyperopt
from hyperopt import pyll

from dotenv import load_dotenv

load_dotenv()
nltk.download('stopwords')
master_ip = os.getenv('master_ip', 'local[*]')
spark = SparkSession \
    .builder \
    .appName("multilayer_perceptron_classification_example") \
    .master(master_ip) \
    .config("spark.driver.memory", "4G") \
    .config("spark.driver.core", "4") \
    .config("spark.executor.core", "4") \
    .config("spark.executor.memory", "4G") \
    .config("spark.cores.max", "8") \
    .config("spark.driver.maxResultSize", "4G") \
    .config("log4j.logger.org.apache.spark.api.python.PythonGatewayServer", "warn") \
    .config("log4j.rootCategory", "warn") \
    .getOrCreate()
# funcs cause https://spark.apache.org/docs/2.0.0/programming-guide.html
spark.sparkContext.setLogLevel("WARN")


def myFunc(label_df, nfeat_vec):
    sentenceData = label_df.select('feat_review_body', 'label')
    tokenizer = Tokenizer(inputCol="feat_review_body", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    get_num_features = wordsData.select('*', sqlf.size('words').alias('words_count'))
    row1 = get_num_features.agg({"words_count": "max"}).collect()[0]
    if nfeat_vec == 0:
        nfeat_vec = row1[0]
    print(nfeat_vec)
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=nfeat_vec)
    featurized_data = hashing_tf.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
    idf_model = idf.fit(featurized_data)
    train_feats_df = idf_model.transform(featurized_data)
    result = train_feats_df.select('label', 'features')
    return result, nfeat_vec


def label_encode(input_df):
    string_index = StringIndexer(inputCol="stars", outputCol="label")
    label_model = string_index.fit(input_df)
    result = label_model.transform(input_df)
    return result


def filter_text(input_df):
    # NOTE: using the title made the prediction worse.
    # input_data['feat_review_body'] = input_data['review_title'] + ' ' + input_data['review_body']
    # TODO: add in proper contractions fix to as pyspark
    input_data = input_df.toPandas()
    input_data['feat_review_body'] = input_data['review_body'].str.replace("n't ", " not ", regex=False)
    input_data['feat_review_body'] = input_data['feat_review_body'].str.replace('[^a-zA-Z]', ' ', regex=True)
    input_data['feat_review_body'] = input_data['feat_review_body'].str.lower()
    nltk.corpus.stopwords.words('english').remove('not')
    pat = r'\b(?:{})\b'.format('|'.join(stopwords.words('english')))
    input_data['feat_review_body'] = input_data['feat_review_body'].str.replace(pat, '', regex=True)
    input_data['feat_review_body'] = input_data['feat_review_body'].str.replace(r'\s+', ' ', regex=True)
    input_data['feat_review_body'] = input_data['feat_review_body'].str.strip()
    input_data = input_data[input_data['feat_review_body'] != '']
    input_data = input_data[input_data['language'] == 'en']
    # input_data.to_parquet(save_path)
    result = spark.createDataFrame(input_data)
    return result


print(f'{datetime.datetime.now()}, load data')
data_path = './data/raw/dataset_en_test.json'
with open(data_path, 'r') as _f:
    test_data = _f.read().splitlines()
test_data = [json.loads(_) for _ in test_data]
test_pd = pd.DataFrame(test_data)
test_df = spark.createDataFrame(test_pd)
print(test_df.count())
data_path = './data/raw/dataset_en_train.json'
with open(data_path, 'r') as _f:
    test_data = _f.read().splitlines()
test_data = [json.loads(_) for _ in test_data]
train_pd = pd.DataFrame(test_data)
train_df = spark.createDataFrame(train_pd)
print(train_df.count())

print(f'{datetime.datetime.now()}, filter data')
# NOTE: using the title made the prediction worse.
# input_data['feat_review_body'] = input_data['review_title'] + ' ' + input_data['review_body']
# TODO: add in proper contractions fix to as pyspark
test_filter_df = filter_text(test_df)
train_filter_df = filter_text(train_df)

print(f'{datetime.datetime.now()}, encode label')
test_label_df = label_encode(test_filter_df)

print(f'{datetime.datetime.now()}, encode label')
train_label_df = label_encode(train_filter_df)

nfeat_vec = 0
print(f'{datetime.datetime.now()}, vectorise features')
train_feats_df_cln, nfeat_vec = myFunc(train_label_df, nfeat_vec)

print(f'{datetime.datetime.now()}, vectorise features')
test_feats_df_cln, nfeat_vec = myFunc(test_label_df, nfeat_vec)

# https://stackoverflow.com/questions/26959221/pyspark-broadcast-variables-from-local-functions
# print('replicate input_pd dataframe to workers in Spark cluster')
# train_inputs = spark.sparkContext.broadcast(train_feats_df_cln)
# test_inputs = spark.sparkContext.broadcast(test_feats_df_cln)
# nfeat_vec_inputs = spark.sparkContext.broadcast(nfeat_vec)


if __name__ == '__main__':

    # print(f'{datetime.datetime.now()}, make model')
    # layers = [nfeat_vec, int(nfeat_vec/2), int(nfeat_vec/4),int(nfeat_vec/10), int(nfeat_vec/4), int(nfeat_vec/2), int(nfeat_vec/10), 5]
    #
    # # create the trainer and set its parameters
    # # trainer = MultilayerPerceptronClassifier(maxIter=2, layers=layers, blockSize=2, seed=1)
    # trainer = MultilayerPerceptronClassifier(maxIter=50, layers=layers, blockSize=128, seed=1234)
    # # train the model
    # model = trainer.fit(train_feats_df_cln)
    # # model = trainer.fit(test_feats_df_cln)
    #
    # print(f'{datetime.datetime.now()}, make predictions')
    # result = model.transform(test_feats_df_cln.select('features', 'label'))
    # print(f'{datetime.datetime.now()}, select predictions')
    # predictionAndLabels = result.select("prediction", "label")
    #
    # print(f'{datetime.datetime.now()}, evaluation predictions')
    # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    #
    # print(f'{datetime.datetime.now()}, print evaluation')
    # print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    #
    # print(f'{datetime.datetime.now()}, save predictions')
    # result_pd = result.select("prediction", "label", "probability").toPandas()
    # result_pd['probability'] = result_pd['probability'].astype('str')
    # result_pd.to_parquet('./data/results/classification_results.parquet')

    # print(f'{datetime.datetime.now()}, save model')
    # model.write().overwrite().save('./artifacts/classification_model')
    # model.save('./artifacts/classification_model')

    # MLFLOW, HYPEROPTS SETUP ####################################################################
    # define function for model evaluation
    def evaluate_model(params):
        # train_values = train_inputs.value
        # test_values = test_inputs.value
        # nfeat_vec_values = nfeat_vec_inputs.value
        train_values = train_feats_df_cln
        test_values = test_feats_df_cln
        nfeat_vec_values = nfeat_vec
        print('evaluate_model: get data')
        # train_values = train
        # test_values = test
        print('evaluate_model: get params')
        layer_left = int(params['layer_left'])
        layer_right = int(params['layer_right'])
        print('evaluate_model: set layers')
        if layer_left==0 or layer_right == 0:
            layer_left = 10
            layer_right = 10
        layers = [nfeat_vec_values, layer_left, layer_right, 5]

        print('create the trainer and set its parameters')
        trainer = MultilayerPerceptronClassifier(maxIter=2, layers=layers, blockSize=128, seed=1234)
        print('train the model')
        model = trainer.fit(train_values)

        print('compute accuracy on the test set')
        result = model.transform(test_values)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        loss = evaluator.evaluate(predictionAndLabels)
        print("Test set accuracy = " + str(loss))
        return {'loss': -loss, 'status': hyperopt.STATUS_OK}


    print('define search space')
    search_space = {
        'layer_left': pyll.scope.int(hyperopt.hp.quniform('layer_left', int(nfeat_vec / 10), nfeat_vec, q=int(nfeat_vec / 10))),
        'layer_right': pyll.scope.int(hyperopt.hp.quniform('layer_right', int(nfeat_vec / 10), nfeat_vec, q=int(nfeat_vec / 10))),
    }

    print('select optimization algorithm')
    algo = hyperopt.tpe.suggest

    print('configure hyperopt settings to distribute to all executors on workers')
    # spark_trials = hyperopt.SparkTrials(parallelism=2, spark_session='multilayer_perceptron_classification_example')
    # spark_trials = hyperopt.SparkTrials()
    spark_trials = hyperopt.Trials()

    with mlflow.start_run(run_name='model_run'):
        print('mlflow start')
        best_params = hyperopt.fmin(
            fn=evaluate_model,
            space=search_space,
            algo=algo,
            max_evals=10,
            trials=spark_trials,
        )
    best_model_params = hyperopt.space_eval(search_space, best_params)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx best model params')
    print(best_model_params)
    ######################################################################################
    print('bye')
