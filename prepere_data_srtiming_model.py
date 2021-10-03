"TV_TYPE",          # Тип тарелки                   (Input)
    "ls_num",           # Номер лицевого счета          (ID)
    "MON_ACTIVATION",   # Месяц подключения             (Constant)
    "CUST_ID",          # Номер клиента                 (ID)
    "ABONID",           # Номер абонента                (ID)
    "smart_card_num",   # Номер смарт-карты             (ID, Unique x18)
    "CHIP_ID",          # Номер чипа                    (ID, almost unique x18)
    "PER",              # Месяц наблюдения              (= PERIOD_, redundant)
    "_1M",              # Отток на горизонте 1 месяц    (Target)
    "_3M",              # Отток на горизонте 3 месяца   (Target)
    "_6M",              # Отток на горизонте 6 месяца   (Target)
    "TOTAL",            # Флаг расторжения контракта    (Target)
    "BASE_PACK",        # Тарифный план текущий         (Input)
    "BASE_PACK_FIRST",  # Тарифный план первый          (Input)
    "PERIOD_",          # Месяц наблюдения              (Input)
    "CURRENT_TARPLANID" # ID тарифного плана            (ID)

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, DateType, LongType, DecimalType
from pyspark.sql.functions import col, when, to_date, count, countDistinct
from pyspark.ml.feature import  StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
import json

spark = SparkSession.builder.appName("mllib_predict_app").getOrCreate()

schema = StructType.fromJson(json.loads('{"fields": [{"metadata": {}, "type": "string",  "name": "TV_TYPE", "nullable": true}, {"metadata": {}, "type": "long", "name": "ls_num", "nullable": true}, {"metadata": {}, "type": "string", "name": "MON_ACTIVATION", "nullable": true}, {"metadata": {}, "type": "integer", "name": "CUST_ID", "nullable": true}, {"metadata": {}, "type": "integer", "name": "ABONID", "nullable": true}, {"metadata": {}, "type": "long", "name": "smart_card_num", "nullable": true}, {"metadata": {}, "type": "double", "name": "CHIP_ID", "nullable": true}, {"metadata": {}, "type": "string", "name": "PER", "nullable": true}, {"metadata": {}, "type": "integer", "name": "_1M", "nullable": true}, {"metadata": {}, "type": "integer", "name": "_3M", "nullable": true}, {"metadata": {}, "type": "integer", "name": "_6M", "nullable": true}, {"metadata": {}, "type": "integer", "name": "TOTAL", "nullable": true}, {"metadata": {}, "type": "string", "name": "BASE_PACK", "nullable": true}, {"metadata": {}, "type": "string", "name": "BASE_PACK_FIRST", "nullable": true}, {"metadata": {}, "type": "string", "name": "PERIOD_", "nullable": true}, {"metadata": {}, "type": "string", "name": "CURRENT_TARPLANID", "nullable": true}], "type": "struct"}'))


source_path = 'work'
model_dir = "/tmp/student781_sss_1/models"
checkpoint_location = "work/tmp/ml_checkpoint"


raw_files = spark \
    .read \
    .format("csv") \
    .schema(schema) \
    .options(path=source_path, header=True) \
    .options(delimiter="\t") \
    .options(encoding="utf-8") \
    .load()

raw_files.printSchema()

def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq) \
        .options(truncate=False) \
        .start()


out = console_output(raw_files, 5)


select_cols = ['BASE_PACK', 'BASE_PACK_FIRST', 'day_space', '_1M', 'TV_TYPE']

def features_clearing(df, features):
    return df \
        .dropDuplicates(['ls_num']) \
        .withColumn("MON_ACTIVATION", to_date("MON_ACTIVATION", 'MM/dd/yyyy')) \
        .withColumn("PERIOD_", to_date("PERIOD_", 'MM/dd/yyyy')) \
        .withColumn("day_space", F.datediff("PERIOD_", "MON_ACTIVATION")) \
        .withColumn('BASE_PACK', when(col('BASE_PACK').isin('?'), 'Нулевой').otherwise(df.BASE_PACK)) \
        .withColumn('BASE_PACK_FIRST', when(col('BASE_PACK_FIRST').isin('?'), 'Нулевой').otherwise(df.BASE_PACK_FIRST)) \
        .select(features)


raw_files = features_clearing(raw_files, select_cols)
raw_files.show(10)

select_cols_indexer = ['BASE_PACK', 'BASE_PACK_FIRST', 'TV_TYPE']

def prepare_data(df,cols_indexer, target):
    for c in cols_indexer:
        string_indexer = StringIndexer(inputCol=c, outputCol=c + '_index').setHandleInvalid("keep")
        df = string_indexer.fit(df).transform(df)
    output_text_columns = [c + "_index" for c in cols_indexer]
    f_columns = list(filter(lambda c: c not in cols_indexer, select_cols))
    f_columns += output_text_columns
    df = df.select(f_columns)
    raw_files_OHE = OneHotEncoderEstimator(inputCols=['BASE_PACK_index', 'BASE_PACK_FIRST_index', 'TV_TYPE_index'],
                                           outputCols=['BASE_PACK_ohe', 'BASE_PACK_FIRST_ohe', 'TV_TYPE_ohe'])
    df = raw_files_OHE.fit(df).transform(df)
    features = ['BASE_PACK_ohe', 'BASE_PACK_FIRST_ohe', 'TV_TYPE_ohe', "day_space" ]
    assembler = VectorAssembler(inputCols=features, outputCol='features', handleInvalid="skip")
    df = assembler.transform(df)
    df = df.select('features', target)
    return df

target = "_1M"
raw_files = prepare_data(raw_files, select_cols_indexer, target )
raw_files.show(3)


evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction") # проверить как работает это Roc AUC



target = ['_1M']

evaluator = MulticlassClassificationEvaluator() \
        .setMetricName('f1') \
        .setLabelCol('_1M') \
        .setPredictionCol("prediction")


def train_model(df, target):
    gbt = GBTClassifier(featuresCol='features', maxIter=5, maxDepth=2, labelCol='_1M', seed=42)
    X_train, X_test = df.randomSplit([0.8, 0.2], seed=12345)
    model = gbt.fit(X_train)
    prediction = model.transform(X_test)
    evaluation_result = evaluator.evaluate(prediction)
    return  model, prediction, evaluation_result



model_1, prediction, evaluation_result = train_model(raw_files,target)
prediction.show()
print(evaluation_result)
print(model_1)
model_1.write().overwrite().save(model_dir + "/model_1")

# df2 = prediction.select(countDistinct("probability"))
#
# df2 = prediction.groupBy("probability") \
#     .agg(count("probability"))
# df2.show(truncate=False)
#


data = spark \
    .readStream \
    .format("csv") \
    .schema(schema) \
    .options(path=source_path, header=True) \
    .options(delimiter="\t") \
    .options(encoding="utf-8") \
    .load()

select_cols = ['BASE_PACK', 'BASE_PACK_FIRST', 'day_space', '_1M', 'TV_TYPE']
select_cols_indexer = ['BASE_PACK', 'BASE_PACK_FIRST', 'TV_TYPE']
target = ['_1M']


model= GBTClassifier.load(model_dir + "/model_1")

def process_batch(df, features, cols_indexer, target):
    df = features_clearing(df, features)
    df = prepare_data(df, cols_indexer, target)
    model_data = train_model(df, target)
    prediction = model.transform(model_data)
    prediction.show()

def foreach_batch_output(df, features, cols_indexer, target):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df\
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch(df, features, cols_indexer, target)) \
        .option("checkpointLocation", checkpoint_location + "/" + date)\
        .start()


stream = foreach_batch_output(data)

stream.awaitTermination()

def process_batch(df):
    model_data = train_model(df)
    prediction = model.transform(model_data)
    prediction.show()


def foreach_batch_output(df):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df\
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch(df)) \
        .option("checkpointLocation", checkpoint_location + "/" + date)\
        .start()

def prepare_data_2(df, target):
    select_cols = ['BASE_PACK', 'BASE_PACK_FIRST', 'day_space', '_1M', 'TV_TYPE']
    cols_indexer = ['BASE_PACK', 'BASE_PACK_FIRST', 'TV_TYPE']
    df = df \
        .dropDuplicates(['ls_num']) \
        .withColumn("MON_ACTIVATION", to_date("MON_ACTIVATION", 'MM/dd/yyyy')) \
        .withColumn("PERIOD_", to_date("PERIOD_", 'MM/dd/yyyy')) \
        .withColumn("day_space", F.datediff("PERIOD_", "MON_ACTIVATION")) \
        .withColumn('BASE_PACK', when(col('BASE_PACK').isin('?'), 'Нулевой').otherwise(df.BASE_PACK)) \
        .withColumn('BASE_PACK_FIRST', when(col('BASE_PACK_FIRST').isin('?'), 'Нулевой').otherwise(df.BASE_PACK_FIRST)) \
        .select(select_cols)
    for c in cols_indexer:
        string_indexer = StringIndexer(inputCol=c, outputCol=c + '_index').setHandleInvalid("keep")
        df = string_indexer.fit(df).transform(df)
    output_text_columns = [c + "_index" for c in cols_indexer]
    f_columns = list(filter(lambda c: c not in cols_indexer, select_cols))
    f_columns += output_text_columns
    df = df.select(f_columns)
    raw_files_OHE = OneHotEncoderEstimator(inputCols=['BASE_PACK_index', 'BASE_PACK_FIRST_index', 'TV_TYPE_index'],
                                           outputCols=['BASE_PACK_ohe', 'BASE_PACK_FIRST_ohe', 'TV_TYPE_ohe'])
    df = raw_files_OHE.fit(df).transform(df)
    features = ['BASE_PACK_ohe', 'BASE_PACK_FIRST_ohe', 'TV_TYPE_ohe', "day_space" ]
    assembler = VectorAssembler(inputCols=features, outputCol='features', handleInvalid="skip")
    df = assembler.transform(df)
    df = df.select('features', target)
    return df


def process_batch(df, epoch):
    model_data = prepare_data(df, target)
    prediction = model.transform(model_data)
    prediction.show()

raw_files = prepare_data_2(raw_files, target )