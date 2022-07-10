import json
import os

import pandas as pd
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, IDF
from pyspark.sql import SparkSession
import pyspark.sql.functions as sqlf
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

spark = SparkSession\
    .builder\
    .appName("multilayer_perceptron_classification_example")\
    .master('local[*]')\
    .config("spark.driver.memory", "8G")\
    .config("spark.driver.maxResultSize", "8G") \
    .getOrCreate()



class ModelUtils():
    def __init__(self):
        self.nfeat_vec = 0
        self.full_df = spark.createDataFrame([], sqlf.StructType([]))
        self.train_df = spark.createDataFrame([], sqlf.StructType([]))
        self.test_df = spark.createDataFrame([], sqlf.StructType([]))
        self.model = None

    def read_data(self, data_path):
        with open(data_path, 'r') as _f:
            test_data = _f.read().splitlines()
        test_data = [json.loads(_) for _ in test_data]
        test_data = test_data[0:1000]
        test_pd = pd.DataFrame(test_data)

        # test_pd = test_pd.drop_duplicates(subset='stars')
        test_df = spark.createDataFrame(test_pd)
        return test_df

    def join_all_data(self):
        self.train_df = self.train_df.withColumn('model_set', sqlf.lit('train'))
        self.test_df = self.test_df.withColumn('model_set', sqlf.lit('test'))
        full_df = self.train_df.union(self.test_df)
        self.full_df = full_df

    def filter_words(self, read_file=False, save_path=''):
        if os.path.exists(save_path) and read_file:
            input_data = pd.read_parquet(save_path)
        else:
            # NOTE: using the title made the prediction worse.
            # input_data['feat_review_body'] = input_data['review_title'] + ' ' + input_data['review_body']
            # TODO: add in proper contractions fix to as pyspark
            input_data = self.full_df.toPandas()
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
        self.full_df = spark.createDataFrame(input_data)

    def label_encode(self):
        # TODO: Put this in pipeline later
        string_index = StringIndexer(inputCol="stars", outputCol="label")
        label_model = string_index.fit(self.full_df)
        self.full_df = label_model.transform(self.full_df)

    def text_vectorize(self):
        sentenceData = self.full_df.select('feat_review_body', 'label')
        tokenizer = Tokenizer(inputCol="feat_review_body", outputCol="words")
        wordsData = tokenizer.transform(sentenceData)

        if self.nfeat_vec == 0:
            get_num_features = wordsData.select('*', sqlf.size('words').alias('words_count'))
            row1 = get_num_features.agg({"words_count": "max"}).collect()[0]
            self.nfeat_vec = row1[0]

        hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=self.nfeat_vec)
        featurized_data = hashing_tf.transform(wordsData)

        idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
        idf_model = idf.fit(featurized_data)
        self.full_df = idf_model.transform(featurized_data)
        self.full_df_cln = self.full_df.select('label', 'features')

    def make_model(self):
        layers = [self.nfeat_vec, 7, 6, 5]

        # create the trainer and set its parameters
        # trainer = MultilayerPerceptronClassifier(maxIter=2, layers=layers, blockSize=2, seed=1)
        trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

        # train the model
        model = trainer.fit(self.full_df_cln)
        result = model.transform(self.full_df_cln.limit(1).select('features', 'label'))
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
        print('bye')

    def predict_data(self):
        pass
        # result = self.model.transform(self.full_df.limit(1).select('features', 'label'))
        # predictionAndLabels = result.select("prediction", "label")
        # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        # print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
